// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/qsa_attn_lanes_parity.cpp - STRATA_QSA_ATTN_LANES (homelab): the decode attention's chunk pass in 16-lane
// sub-groups against today's kernel, bitwise (memcmp of the outputs), then the time of both.
//
//   parity: Qwen3.8's 24/2 heads of 256 and Ornith-1.5's 16/2, every KV format (FP16 and INT8 take the lanes form,
//   Q4_0 and K8V4 keep the default kernel and must not change), 16 rows over 16 slots of their own pools under a
//   permuted page table with masked (non-resident) blocks, widths from 1 cell to the full selection (2,051) and a chunk
//   cut short; every entry point (step, batch, slots at 1..16 rows, dense step and batch).  One INT8 row is also held
//   against an FP64 host reference (1e-4 of the output's scale), so the two kernels cannot agree on a wrong answer.
//   STRATA_QSA_ATTN_FAST (not bitwise): the same calls against the default kernel within 1e-5 of the output's scale,
//   at the span the setting picks and at 1, 3 and 8 chunks a work-group; Q4_0 and K8V4 (the default kernel) by memcmp;
//   the FP64 check again.  Timed beside the other two.
//   time: INT8, the full selection a row, 1, 7, 16 and 32 rows (the batch form over one pool; the slots form, the
//   engine's STRATA_BATCH_QSA_ROWS call, up to 16), each row's K/V its own and three sets of it rotated so the L2 holds
//   none of a call's K/V.  Each time is the fastest and the median of `rounds` rounds of `reps` calls, the two kernels'
//   rounds in turns; the floor is the K/V bytes at --gbs GB/s (the B70's 456).
//
//   STRATA_QSA_SCORES_STRIDE: the block scores of 1..16 rows (each its own slot's keys, depths 1 .. 20,000 cells, the
//   grid of a 131,072-cell cache) against the default grid, memcmp, then both timed at 1, 7 and 16 rows.
//
//     build/qsa_attn_lanes_parity [--parity-only] [--rows 1,7,16,32] [--reps 20] [--rounds 7] [--gbs 456]
//                                [--depth 16384]
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
namespace P = strata::parity;
int g_fail = 0, g_checks = 0;
std::mt19937 rng(353);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.

template <typename T> T* dput(const std::vector<T>& h) {
    T* p = static_cast<T*>(P::alloc_bytes(h.size() * sizeof(T) + 64));
    P::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}
template <typename T> T* dzero(size_t n) { return dput(std::vector<T>(n, T{})); }
template <typename T> std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n);
    P::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}
std::vector<float> gauss(size_t n, float scale) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng) * scale;
    return v;
}
// width cells of [0, n_kv), ascending: the identity below the budget (as the top-k returns it), else a random subset
std::vector<int32_t> selection(int n_kv, int width, int cap) {
    std::vector<int32_t> ids((size_t) cap, 0);
    if (width == n_kv) {
        for (int c = 0; c < width; ++c) ids[(size_t) c] = c;
        return ids;
    }
    std::vector<int32_t> all((size_t) n_kv);
    for (int c = 0; c < n_kv; ++c) all[(size_t) c] = c;
    std::shuffle(all.begin(), all.end(), rng);
    std::sort(all.begin(), all.begin() + width);
    std::copy(all.begin(), all.begin() + width, ids.begin());
    return ids;
}

// One slot's pools in all four formats (random contents, finite scales), its page table, and its host copies.
struct Pools {
    int32_t* table = nullptr;
    uint16_t *kf = nullptr, *vf = nullptr, *ks = nullptr, *vs = nullptr;
    int8_t *kq = nullptr, *vq = nullptr;
    uint8_t *k4 = nullptr, *v4 = nullptr;
    std::vector<int32_t> h_table;
    std::vector<int8_t> h_kq, h_vq;
    std::vector<uint16_t> h_ks, h_vs;
    k::QsaAttnPools of(int fmt) const {
        k::QsaAttnPools p;
        p.page_table = table;
        if (fmt == 0) { p.k_pool = kf; p.v_pool = vf; }
        else if (fmt == 1) { p.k_q = kq; p.v_q = vq; p.k_scale = ks; p.v_scale = vs; }
        else if (fmt == 2) { p.k_q4 = k4; p.v_q4 = v4; }
        else { p.k_q = kq; p.k_scale = ks; p.v_q4 = v4; }
        return p;
    }
};
Pools make_pools(const k::QsaShapes& s, int cells, int masked_pages) {
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, PS = (int) s.page_size, pages = cells / PS;
    const size_t rows = (size_t) pages * H * PS, n = rows * D, ns = rows * (D / k::KV_Q8_GROUP);
    const size_t n4 = rows * (D / k::QK4_0) * sizeof(k::block_q4_0);
    Pools x;
    x.h_table.resize((size_t) pages);
    for (int i = 0; i < pages; ++i) x.h_table[(size_t) i] = i;
    std::shuffle(x.h_table.begin(), x.h_table.end(), rng);
    for (int i = 0; i < masked_pages; ++i) x.h_table[(size_t) (rng() % pages)] = -1;   // blocks the streaming left out
    std::uniform_int_distribution<int> code(-127, 127), byte(0, 255);
    std::uniform_real_distribution<float> sc(0.004f, 0.03f);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> kf(n), vf(n);
    for (auto& v : kf) v = k::f16_from_f32(nd(rng) * 1.5f);
    for (auto& v : vf) v = k::f16_from_f32(nd(rng));
    x.h_kq.resize(n); x.h_vq.resize(n); x.h_ks.resize(ns); x.h_vs.resize(ns);
    for (auto& v : x.h_kq) v = (int8_t) code(rng);
    for (auto& v : x.h_vq) v = (int8_t) code(rng);
    for (auto& v : x.h_ks) v = k::f16_from_f32(sc(rng) * 4.f);
    for (auto& v : x.h_vs) v = k::f16_from_f32(sc(rng));
    std::vector<uint8_t> k4(n4), v4(n4);
    for (auto& v : k4) v = (uint8_t) byte(rng);
    for (auto& v : v4) v = (uint8_t) byte(rng);
    for (size_t b = 0; b < n4 / sizeof(k::block_q4_0); ++b) {   // finite block scales
        const uint16_t d = k::f16_from_f32(sc(rng) * 8.f);
        std::memcpy(k4.data() + b * sizeof(k::block_q4_0), &d, 2);
        std::memcpy(v4.data() + b * sizeof(k::block_q4_0), &d, 2);
    }
    x.table = dput(x.h_table);
    x.kf = dput(kf); x.vf = dput(vf);
    x.kq = dput(x.h_kq); x.vq = dput(x.h_vq); x.ks = dput(x.h_ks); x.vs = dput(x.h_vs);
    x.k4 = dput(k4); x.v4 = dput(v4);
    return x;
}

const char* const kFmt[] = {"fp16", "int8", "q4_0", "K8V4"};
constexpr double kFastTol = 1e-5;   // the fast form against the default, of the output's scale

bool same(const char* what, const float* a, const float* b, size_t n) {
    ++g_checks;
    const auto x = fetch(a, n), y = fetch(b, n);
    if (std::memcmp(x.data(), y.data(), n * sizeof(float)) == 0) return true;
    size_t i = 0;
    while (i < n && std::memcmp(&x[i], &y[i], sizeof(float)) == 0) ++i;
    std::printf("    *** %s: the lanes form differs (first at %zu: %.9g vs %.9g) ***\n", what, i, (double) x[i], (double) y[i]);
    ++g_fail;
    return false;
}

// within tol of the output's scale (the largest |value| of a)
bool near(const char* what, const float* a, const float* b, size_t n, double tol) {
    ++g_checks;
    const auto x = fetch(a, n), y = fetch(b, n);
    double sc = 0.0, err = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        sc = std::max(sc, (double) std::fabs(x[i]));
        const double d = std::fabs((double) x[i] - (double) y[i]);
        if (d > err || std::isnan(d)) { err = std::isnan(d) ? 1e300 : d; at = i; }
    }
    if (err <= tol * sc) return true;
    std::printf("    *** %s: the fast form is %.3e of the scale off (at %zu: %.9g vs %.9g) ***\n", what, err / sc, at,
                (double) x[at], (double) y[at]);
    ++g_fail;
    return false;
}

// the FP64 attention of one query row over INT8 pools (the selection `ids`, its masked cells dropped)
double fp64_check(const Pools& x, const std::vector<float>& q, const std::vector<int32_t>& ids, int width,
                  const k::QsaShapes& s, const std::vector<float>& got) {
    const int NH = (int) s.n_head, H = (int) s.n_head_kv, D = (int) s.head_dim, PS = (int) s.page_size, G = NH / H;
    const int GS = D / k::KV_Q8_GROUP;
    double scale_out = 0.0, err = 0.0;
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / G;
        std::vector<double> sc;
        std::vector<long long> rows;
        for (int c = 0; c < width; ++c) {
            const int cell = ids[(size_t) c];
            const int page = x.h_table[(size_t) (cell / PS)];
            if (page < 0) continue;
            const long long row = ((long long) page * H + kvh) * PS + cell % PS;
            double d = 0.0;
            for (int i = 0; i < D; ++i)
                d += (double) q[(size_t) (h * D + i)] * x.h_kq[(size_t) (row * D + i)] *
                     k::f32_from_f16(x.h_ks[(size_t) (row * GS + i / k::KV_Q8_GROUP)]);
            sc.push_back(d / std::sqrt((double) D));
            rows.push_back(row);
        }
        double m = -1e300, l = 0.0;
        for (double v : sc) m = std::max(m, v);
        for (double& v : sc) { v = std::exp(v - m); l += v; }
        for (int i = 0; i < D; ++i) {
            double o = 0.0;
            for (size_t c = 0; c < rows.size(); ++c)
                o += sc[c] / l * x.h_vq[(size_t) (rows[c] * D + i)] *
                     k::f32_from_f16(x.h_vs[(size_t) (rows[c] * GS + i / k::KV_Q8_GROUP)]);
            scale_out = std::max(scale_out, std::fabs(o));
            err = std::max(err, std::fabs(o - (double) got[(size_t) (h * D + i)]));
        }
    }
    return err / scale_out;
}

void parity(const char* name, int64_t nh) {
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = nh;
    const int NH = (int) s.n_head, D = (int) s.head_dim, R = k::kQsaSlotsMax;
    const int cells = 2304;   // past the selection's 2,051: random subsets, and a chunk cut short at the end
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int pos_of[R] = {0, 2, 62, 63, 64, 299, 1000, 2049, 2050, 2051, 2303, 2303, 2200, 2303, 511, 1500};
    std::printf("%s: %d rows, widths 1 .. %lld\n", name, R, (long long) cap);
    std::vector<Pools> pl;
    for (int r = 0; r < R; ++r) pl.push_back(make_pools(s, cells, r % 3 == 1 ? 6 : 0));
    std::vector<int32_t> hstep((size_t) R * k::kStepCount), hids((size_t) R * cap);
    for (int r = 0; r < R; ++r) {
        k::qsa_step_fill(hstep.data() + (size_t) r * k::kStepCount, pos_of[r], s);
        const int n_kv = pos_of[r] + 1, width = (int) k::qsa_selection_width(n_kv, s);
        const auto ids = selection(n_kv, width, (int) cap);
        std::copy(ids.begin(), ids.end(), hids.begin() + (size_t) r * cap);
    }
    const int32_t* d_step = dput(hstep);
    const int32_t* d_ids = dput(hids);
    const std::vector<float> hq = gauss((size_t) R * NH * D, 0.5f);
    const float* d_q = dput(hq);
    const size_t sf = k::qsa_decode_attn_scratch_floats(cap, s), sfd = k::qsa_decode_attn_scratch_floats(cells, s);
    float* scr = dzero<float>((size_t) R * std::max(sf, sfd));
    const size_t out = (size_t) R * NH * D;
    float* a = dzero<float>(out);
    float* b = dzero<float>(out);
    void* cs = P::stream();
    // run `f` with the setting off into a and on into b, compare
    int fmt_now = 0;
    const auto ab = [&](const std::string& what, size_t n, auto f) {
        k::qsa_decode_attn_set_fast(0, 0);
        k::qsa_decode_attn_set_lanes(0);
        f(a);
        k::qsa_decode_attn_set_lanes(1);
        f(b);
        P::sync();
        same(what.c_str(), a, b, n);
        for (int span : {0, 1, 3, 8}) {   // the fast form against the default (a)
            k::qsa_decode_attn_set_fast(1, span);
            f(b);
            P::sync();
            const std::string w = what + " fast span " + std::to_string(span);
            if (fmt_now <= 1) near(w.c_str(), a, b, n, kFastTol);
            else same(w.c_str(), a, b, n);
        }
        k::qsa_decode_attn_set_fast(0, 0);
    };
    for (int fmt = 0; fmt < 4; ++fmt) {
        fmt_now = fmt;
        const std::string f = std::string(kFmt[fmt]) + " ";
        // step: one row over its own slot
        for (int r = 0; r < R; ++r)
            ab(f + "step row " + std::to_string(r), (size_t) NH * D, [&](float* o) {
                k::qsa_decode_attn_step(d_q + (size_t) r * NH * D, pl[(size_t) r].of(fmt), d_ids + (size_t) r * cap,
                                        d_step + (size_t) r * k::kStepCount, cap, s, scr, o, cs);
            });
        // batch: the 16 rows over slot 10's pools (each row its own selection)
        ab(f + "batch 16 rows", out, [&](float* o) {
            k::qsa_decode_attn_batch(d_q, pl[10].of(fmt), d_ids, d_step, cap, s, scr, o, R, cs);
        });
        // slots: rows 1..16 a call, each over its own slot
        k::QsaAttnSlots tab;
        for (int r = 0; r < R; ++r)
            if (!k::qsa_attn_slots_set(tab, r, pl[(size_t) r].of(fmt))) throw std::runtime_error("slots table refused");
        for (int n = 1; n <= R; ++n)
            ab(f + "slots " + std::to_string(n) + " rows", (size_t) n * NH * D, [&](float* o) {
                k::qsa_decode_attn_slots(d_q, tab, d_ids, d_step, cap, s, scr, o, n, cs);
            });
        if (fmt == 1) {   // the lanes form's and the fast form's 16-row slots output against FP64, for two rows
          for (int form = 0; form < 2; ++form) {
            k::qsa_decode_attn_set_lanes(1);
            k::qsa_decode_attn_set_fast(form, 3);
            k::qsa_decode_attn_slots(d_q, tab, d_ids, d_step, cap, s, scr, b, R, cs);
            P::sync();
            k::qsa_decode_attn_set_fast(0, 0);
            const auto got = fetch(b, out);
            for (int r : {9, 10}) {
                const int width = hstep[(size_t) r * k::kStepCount + k::kStepWidth];
                const std::vector<float> qr(hq.begin() + (size_t) r * NH * D, hq.begin() + (size_t) (r + 1) * NH * D);
                const std::vector<int32_t> ir(hids.begin() + (size_t) r * cap, hids.begin() + (size_t) (r + 1) * cap);
                const std::vector<float> gr(got.begin() + (size_t) r * NH * D, got.begin() + (size_t) (r + 1) * NH * D);
                const double rel = fp64_check(pl[(size_t) r], qr, ir, width, s, gr);
                ++g_checks;
                std::printf("  int8 %s row %d (width %d) vs FP64: %.3e of the scale %s\n", form ? "fast" : "lanes", r,
                            width, rel, rel <= 1e-4 ? "ok" : "FAIL");
                if (rel > 1e-4) ++g_fail;
            }
          }
        }
        // dense: every cell of the cache (n_kv cells), one row and the 16
        ab(f + "dense step", (size_t) NH * D, [&](float* o) {
            k::qsa_dense_attn_step(d_q + (size_t) 11 * NH * D, pl[11].of(fmt), d_step + (size_t) 11 * k::kStepCount,
                                   cells, s, scr, o, cs);
        });
        ab(f + "dense batch 16 rows", out, [&](float* o) {
            k::qsa_dense_attn_batch(d_q, pl[0].of(fmt), d_step, cells, s, scr, o, R, cs);
        });
        std::printf("  %s: done (%d checks, %d failures so far)\n", kFmt[fmt], g_checks, g_fail);
    }
    k::qsa_decode_attn_set_lanes(-1);
    k::qsa_decode_attn_set_fast(-1, 0);
}

// STRATA_QSA_SCORES_STRIDE: qsa_block_scores_slots on its fixed grid against the default, memcmp; then timed
void scores(bool time, const std::vector<int>& rows, int reps, int rounds, int depth) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int R = k::kQsaSlotsMax, ID = (int) s.idx_dim, IQ = (int) s.idx_n_head;
    const int64_t max_blocks = 131072 / s.idx_block + 2;
    const int pos_of[R] = {0, 3, 4, 5, 255, 4095, 4096, 9000, 16383, 19999, 12345, 7, 2047, 2048, 15000, 16384};
    k::QsaScoreSlots tab;
    for (int r = 0; r < R; ++r) {
        tab.pooled[r] = dput(gauss((size_t) max_blocks * ID, 1.f));
        tab.dead[r] = dput(gauss((size_t) ID, 1.f));
    }
    const float* q = dput(gauss((size_t) R * IQ * ID, 1.f));
    std::vector<int32_t> hstep((size_t) R * k::kStepCount);
    for (int r = 0; r < R; ++r) k::qsa_step_fill(hstep.data() + (size_t) r * k::kStepCount, pos_of[r], s);
    const int32_t* d_step = dput(hstep);
    const size_t n = (size_t) R * max_blocks;
    float* a = dzero<float>(n);
    float* b = dzero<float>(n);
    void* cs = P::stream();
    for (int nq = 1; nq <= R; ++nq) {
        k::qsa_block_scores_set_stride(0);
        k::qsa_block_scores_slots(tab, q, d_step, nq, max_blocks, s, a, cs);
        k::qsa_block_scores_set_stride(1);
        k::qsa_block_scores_slots(tab, q, d_step, nq, max_blocks, s, b, cs);
        P::sync();
        same(("block scores " + std::to_string(nq) + " rows").c_str(), a, b, n);
    }
    std::printf("  block scores: done (%d checks, %d failures so far)\n", g_checks, g_fail);
    if (time) {
        for (int r = 0; r < R; ++r) k::qsa_step_fill(hstep.data() + (size_t) r * k::kStepCount, depth - 1, s);
        P::copy_bytes_in(const_cast<int32_t*>(d_step), hstep.data(), hstep.size() * sizeof(int32_t));
        const double bytes = (double) depth / s.idx_block * ID * 4;   // a row's keys
        std::printf("time: block scores at depth %d (%.2f MB of keys a row), grid of %lld blocks a row\n", depth,
                    bytes / 1e6, (long long) max_blocks);
        std::printf("  %4s  %-8s %9s %9s %7s\n", "rows", "grid", "min us", "med us", "GB/s");
        for (int nq : rows) {
            if (nq > R) continue;
            std::vector<double> us[2];
            for (int round = 0; round <= rounds; ++round)
                for (int v = 0; v < 2; ++v) {
                    k::qsa_block_scores_set_stride(v);
                    k::qsa_block_scores_slots(tab, q, d_step, nq, max_blocks, s, a, cs);
                    P::sync();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) k::qsa_block_scores_slots(tab, q, d_step, nq, max_blocks, s, a, cs);
                    P::sync();
                    const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                    if (round > 0) us[v].push_back(t);
                }
            for (int v = 0; v < 2; ++v) {
                auto& u = us[v];
                std::sort(u.begin(), u.end());
                std::printf("  %4d  %-8s %9.1f %9.1f %7.0f\n", nq, v == 0 ? "default" : "stride", u[0], u[u.size() / 2],
                            nq * bytes / u[0] / 1e3);
            }
        }
        // the top-k after them (one work-group a row over the row's n_bid + 1 scores), the window's next launch
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
        int32_t* ids = dzero<int32_t>((size_t) R * cap);
        k::qsa_block_scores_set_stride(1);
        k::qsa_block_scores_slots(tab, q, d_step, R, max_blocks, s, a, cs);
        P::sync();
        std::printf("time: block top-k at depth %d (%lld scores a row, %lld cells kept)\n", depth,
                    (long long) (depth / s.idx_block + 1), (long long) cap);
        std::printf("  %4s  %-8s %9s %9s\n", "rows", "kernel", "min us", "med us");
        for (int nq : rows) {
            if (nq > R) continue;
            std::vector<double> u[1];
            for (int round = 0; round <= rounds; ++round)
                for (int v = 0; v < 1; ++v) {
                    k::qsa_block_topk(a, d_step, nq, max_blocks, cap, s, ids, cs);
                    P::sync();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) k::qsa_block_topk(a, d_step, nq, max_blocks, cap, s, ids, cs);
                    P::sync();
                    const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                    if (round > 0) u[v].push_back(t);
                }
            std::sort(u[0].begin(), u[0].end());
            std::printf("  %4d  %-8s %9.1f %9.1f\n", nq, "wide", u[0][0], u[0][u[0].size() / 2]);
        }
    }
    k::qsa_block_scores_set_stride(-1);
}

std::vector<int> parse_list(const std::string& v) {
    std::vector<int> r;
    std::stringstream ss(v);
    std::string t;
    while (std::getline(ss, t, ',')) r.push_back(std::atoi(t.c_str()));
    return r;
}

void timing(const std::vector<int>& rows, int reps, int rounds, double gbs) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int NH = (int) s.n_head, H = (int) s.n_head_kv, D = (int) s.head_dim, PS = (int) s.page_size;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int RM = *std::max_element(rows.begin(), rows.end()), SETS = 3, REGION = 4096;   // a row's resident cells
    const int cells = SETS * RM * REGION, pages = cells / PS;
    const size_t prow = (size_t) pages * H * PS;
    // one pool for every row and set (identity page table); row r of set j selects 2,051 of its own 4,096 cells
    std::vector<int32_t> table((size_t) pages);
    for (int i = 0; i < pages; ++i) table[(size_t) i] = i;
    k::QsaAttnPools p;
    p.page_table = dput(table);
    std::vector<int8_t> codes(prow * D);
    std::uniform_int_distribution<int> code(-127, 127);
    for (auto& v : codes) v = (int8_t) code(rng);
    p.k_q = dput(codes);
    for (auto& v : codes) v = (int8_t) code(rng);
    p.v_q = dput(codes);
    std::vector<uint16_t> scl(prow * (D / k::KV_Q8_GROUP));
    for (auto& v : scl) v = k::f16_from_f32(0.01f + 0.02f * (float) (rng() % 1000) / 1000.f);
    p.k_scale = dput(scl);
    p.v_scale = dput(scl);
    std::vector<int32_t> hstep((size_t) RM * k::kStepCount), hids((size_t) SETS * RM * cap);
    for (int r = 0; r < RM; ++r) k::qsa_step_fill(hstep.data() + (size_t) r * k::kStepCount, REGION - 1, s);
    for (int j = 0; j < SETS; ++j)
        for (int r = 0; r < RM; ++r) {
            auto ids = selection(REGION, (int) cap, (int) cap);
            for (auto& c : ids) c += (j * RM + r) * REGION;
            std::copy(ids.begin(), ids.end(), hids.begin() + (size_t) (j * RM + r) * cap);
        }
    const int32_t* d_step = dput(hstep);
    const int32_t* d_ids = dput(hids);
    const float* d_q = dput(gauss((size_t) RM * NH * D, 0.5f));
    const size_t sf = k::qsa_decode_attn_scratch_floats(cap, s);
    float* scr = dzero<float>((size_t) RM * sf);
    float* out = dzero<float>((size_t) RM * NH * D);
    void* cs = P::stream();
    const double kv = (double) cap * (double) k::kv_q8_bytes_per_cell(s);   // a row's K/V bytes
    std::printf("time: INT8, %lld cells a row (%.2f MB of K/V), %d reps x %d rounds, floor at %.0f GB/s\n",
                (long long) cap, kv / 1e6, reps, rounds, gbs);
    std::printf("  %-6s %4s  %-8s %9s %9s %7s %6s\n", "form", "rows", "kernel", "min us", "med us", "GB/s", "xfloor");
    for (int form = 0; form < 2; ++form)
        for (int n : rows) {
            if (form == 1 && n > k::kQsaSlotsMax) continue;
            k::QsaAttnSlots tab;
            for (int r = 0; r < std::min(n, (int) k::kQsaSlotsMax); ++r) k::qsa_attn_slots_set(tab, r, p);
            const auto call = [&](int j) {
                const int32_t* ids = d_ids + (size_t) j * RM * cap;
                if (form == 0) k::qsa_decode_attn_batch(d_q, p, ids, d_step, cap, s, scr, out, n, cs);
                else k::qsa_decode_attn_slots(d_q, tab, ids, d_step, cap, s, scr, out, n, cs);
            };
            std::vector<double> us[3];
            for (int round = 0; round <= rounds; ++round)
                for (int v = 0; v < 3; ++v) {
                    k::qsa_decode_attn_set_lanes(v >= 1);
                    k::qsa_decode_attn_set_fast(v == 2, 0);
                    call(0);
                    P::sync();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) call(i % SETS);
                    P::sync();
                    const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                    if (round > 0) us[v].push_back(t);   // round 0: the warm-up
                }
            for (int v = 0; v < 3; ++v) {
                auto& u = us[v];
                std::sort(u.begin(), u.end());
                const double floor_us = n * kv / (gbs * 1e3);
                std::printf("  %-6s %4d  %-8s %9.1f %9.1f %7.0f %6.2f\n", form == 0 ? "batch" : "slots", n,
                            v == 0 ? "default" : v == 1 ? "lanes" : "fast", u[0], u[u.size() / 2], n * kv / u[0] / 1e3, u[0] / floor_us);
            }
        }
    // the fast form's partials: the slots call at the most rows it takes, each span forced (span 1: a merge over every
    // chunk, as the default kernel's)
    const int ns = std::min(RM, (int) k::kQsaSlotsMax);
    k::QsaAttnSlots tab;
    for (int r = 0; r < ns; ++r) k::qsa_attn_slots_set(tab, r, p);
    std::printf("time: the fast form, slots %d rows, chunks a work-group forced\n  %4s %9s %9s %6s\n", ns, "span",
                "min us", "med us", "xfloor");
    k::qsa_decode_attn_set_lanes(1);
    for (int span : {1, 2, 3, 4, 6, 8}) {
        k::qsa_decode_attn_set_fast(1, span);
        std::vector<double> u;
        for (int round = 0; round <= rounds; ++round) {
            k::qsa_decode_attn_slots(d_q, tab, d_ids, d_step, cap, s, scr, out, ns, cs);
            P::sync();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i)
                k::qsa_decode_attn_slots(d_q, tab, d_ids + (size_t) (i % SETS) * RM * cap, d_step, cap, s, scr, out, ns, cs);
            P::sync();
            const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            if (round > 0) u.push_back(t);
        }
        std::sort(u.begin(), u.end());
        std::printf("  %4d %9.1f %9.1f %6.2f\n", span, u[0], u[u.size() / 2], u[0] / (ns * kv / (gbs * 1e3)));
    }
    k::qsa_decode_attn_set_lanes(-1);
    k::qsa_decode_attn_set_fast(-1, 0);
}
}  // namespace

int main(int argc, char** argv) try {
    bool parity_only = false;
    std::vector<int> rows{1, 7, 16, 32};
    int reps = 20, rounds = 7;
    double gbs = 456.0;
    int depth = 16384;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--parity-only") parity_only = true;
        else if (a == "--rows" && i + 1 < argc) rows = parse_list(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--gbs" && i + 1 < argc) gbs = std::atof(argv[++i]);
        else if (a == "--depth" && i + 1 < argc) depth = std::atoi(argv[++i]);
        else throw std::runtime_error("unknown argument " + a);
    }
    parity("Qwen3.8 QSA (24/2 x 256)", 24);
    parity("Ornith-1.5 (16/2 x 256)", 16);
    scores(!parity_only, rows, reps, rounds, depth);
    std::printf("qsa_attn_lanes: %d checks, %d failures\n", g_checks, g_fail);
    if (!parity_only && g_fail == 0) timing(rows, reps, rounds, gbs);   // a device without timing: --parity-only
    return g_fail ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "qsa_attn_lanes_parity: %s\n", e.what());
    return 1;
}
