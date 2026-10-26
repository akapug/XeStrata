// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/hc_read16_parity.cpp - STRATA_HC_Q8_WIN: fused_gr_read_multi's Q8_0 hyper-connection read of a window in
// one pass (every form) against the established path (passes of kFusedGrMaxT tokens), bitwise, and the time of each.
//
//   * the model's geometry (n_embd 2560, hc 4, hc_lr 320), synthetic Q8_0 projections in kNativeQ8Rows' layout;
//   * every window of 1..16 tokens, the previous write folded for all, none or every other token, with and without
//     the inject rows and mixed's q8_1, out of place and in place; each token's R and R_out at a scattered address
//     (only the other buffers are base + token * stride), and one window that is not a window's layout (the fallback);
//   * memcmp of R_out, lo, rs, inject_out, mixed and the q8_1;
//   * the time per call of each form at 1, 2, 4, 8 and 16 tokens, whole and per launch (down; lo; up), `reps` calls
//     in a captured graph a round, the rounds taken in turns (the card may be serving): the fastest and median round.
//
//     build/hc_read16_parity [reps] [rounds]
//
// HC16_FORMS=10,11,12 (a list) checks and times only those forms beside the reference (form 0), e.g. on the OpenCL CPU
// device, whose compiler is slow on the others
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"
#include "strata/core/gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace P = strata::parity;
namespace k = strata::kernels;

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, MT = k::kVerifyMaxT;
constexpr size_t ROW_D = (size_t) D + D / 32 * 2, ROW_U = (size_t) LR + LR / 32 * 2, Q81 = (size_t) N / 32 * 36;

template <typename T> T* up(const std::vector<T>& h) {
    T* p = static_cast<T*>(P::alloc_bytes(h.size() * sizeof(T) + 64));
    P::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}
template <typename T> T* buf(size_t n) { return static_cast<T*>(P::alloc_bytes(n * sizeof(T) + 64)); }
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    P::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}

// rows of `ne0` Q8_0 values: the int8 values, then the fp16 scales
std::vector<uint8_t> q8_rows(int rows, int ne0, std::mt19937& rng) {
    const size_t row = (size_t) ne0 + (size_t) ne0 / 32 * 2;
    std::vector<uint8_t> w((size_t) rows * row);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> sc(0.002f, 0.02f);
    for (int r = 0; r < rows; ++r) {
        uint8_t* p = w.data() + (size_t) r * row;
        for (int i = 0; i < ne0; ++i) p[i] = (uint8_t) (int8_t) q(rng);
        for (int b = 0; b < ne0 / 32; ++b) {
            const uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(sc(rng)));
            std::memcpy(p + ne0 + 2 * b, &h, 2);
        }
    }
    return w;
}

struct Bufs {
    float *R, *Rout, *R0, *bo, *inj, *lo, *rs, *io, *mixed, *xn;
    uint8_t* q8m;
};

}  // namespace

int main(int argc, char** argv) try {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20, rounds = argc > 2 ? std::atoi(argv[2]) : 7;
    void* s = P::stream();
    sycl::queue& q = strata::core::Runtime::get().compute();
    std::printf("device %s, max_parameter_size %zu bytes\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                (size_t) q.get_device().get_info<sycl::info::device::max_parameter_size>());
    std::mt19937 rng(353);
    std::normal_distribution<float> nd(0.f, 1.f);
    const uint8_t* q8_down = up(q8_rows(LR, D, rng));
    const uint8_t* q8_up = up(q8_rows(D, LR, rng));
    std::vector<float> wn(D);
    for (auto& x : wn) x = 1.0f + 0.1f * nd(rng);
    std::vector<uint16_t> wi((size_t) HC * D);
    for (auto& x : wi) x = (uint16_t) (sycl::bit_cast<uint32_t>(0.02f * nd(rng)) >> 16);
    const float* w_norm = up(wn);
    const uint16_t* w_inject = up(wi);
    std::vector<float> R((size_t) MT * D), bo((size_t) MT * N), inj((size_t) MT * HC);
    for (int t = 0; t < MT; ++t)
        for (int c = 0; c < HC; ++c)
            for (int d = 0; d < N; ++d) R[(size_t) t * D + c * N + d] = nd(rng) * (float) (1 << (2 * c));
    for (auto& x : bo) x = nd(rng);
    for (auto& x : inj) x = 3.0f * nd(rng);
    Bufs b{};
    b.R0 = up(R);
    b.R = buf<float>(R.size());
    b.Rout = buf<float>(R.size());
    b.bo = up(bo);
    b.inj = up(inj);
    b.lo = buf<float>((size_t) MT * LR);
    b.rs = buf<float>((size_t) MT * HC);
    b.io = buf<float>((size_t) MT * HC);
    b.mixed = buf<float>((size_t) MT * N);
    b.xn = buf<float>((size_t) MT * D);
    b.q8m = buf<uint8_t>((size_t) MT * Q81);
    // each token's R and R_out at a scattered slot
    std::vector<int> perm(MT), perm2(MT);
    std::iota(perm.begin(), perm.end(), 0);
    std::iota(perm2.begin(), perm2.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::shuffle(perm2.begin(), perm2.end(), rng);

    struct Cfg { int apply; bool inject, q8m, inplace, scattered; };   // apply: 0 none, 1 all, 2 every other token
    const auto args = [&](int T, const Cfg& c) {
        std::vector<k::FusedGrArgs> a((size_t) T);
        for (int t = 0; t < T; ++t) {
            k::FusedGrArgs& x = a[(size_t) t];
            x.R = b.R + (size_t) perm[t] * D;
            x.R_out = c.inplace ? b.R + (size_t) perm[t] * D : b.Rout + (size_t) perm2[t] * D;
            x.apply = c.apply == 1 || (c.apply == 2 && t % 2 == 0);
            x.bo_prev = b.bo + (size_t) t * N;
            x.inj_prev = b.inj + (size_t) t * HC;
            x.w_norm = w_norm;
            x.w_inject = c.inject ? w_inject : nullptr;
            x.q8_down = q8_down;
            x.q8_up = q8_up;
            x.lo = b.lo + (size_t) t * LR;
            x.rs = b.rs + (size_t) t * HC;
            x.inject_out = b.io + (size_t) t * HC;
            x.mixed = b.mixed + (size_t) (c.scattered ? perm[t] : t) * N;
            x.q8_mixed = c.q8m ? b.q8m + (size_t) t * Q81 : nullptr;
        }
        return a;
    };
    const auto reset = [&] {
        q.memcpy(b.R, b.R0, R.size() * 4);
        q.memset(b.Rout, 0xff, R.size() * 4);
        q.memset(b.lo, 0xff, (size_t) MT * LR * 4);
        q.memset(b.rs, 0xff, (size_t) MT * HC * 4);
        q.memset(b.io, 0xff, (size_t) MT * HC * 4);
        q.memset(b.mixed, 0xff, (size_t) MT * N * 4);
        q.memset(b.q8m, 0xff, (size_t) MT * Q81);
        q.memset(b.xn, 0xff, (size_t) MT * D * 4);
        P::sync();
    };
    struct Out { std::vector<float> R, Rout, lo, rs, io, mixed; std::vector<uint8_t> q8m; };
    const auto outputs = [&] {
        Out o;
        o.R = down(b.R, R.size()); o.Rout = down(b.Rout, R.size()); o.lo = down(b.lo, (size_t) MT * LR);
        o.rs = down(b.rs, (size_t) MT * HC); o.io = down(b.io, (size_t) MT * HC); o.mixed = down(b.mixed, (size_t) MT * N);
        o.q8m = down(b.q8m, (size_t) MT * Q81);
        return o;
    };
    const auto first_diff = [](const void* x, const void* y, size_t n) -> long {
        const auto* p = static_cast<const uint8_t*>(x);
        const auto* r = static_cast<const uint8_t*>(y);
        for (size_t i = 0; i < n; ++i) if (p[i] != r[i]) return (long) i;
        return -1;
    };
    const char* const names[] = {"R", "R_out", "lo", "rs", "inject_out", "mixed", "q8_mixed"};
    std::vector<bool> want(k::kFusedGrQ8WinForms, true);
    if (const char* f = std::getenv("HC16_FORMS")) {
        std::fill(want.begin(), want.end(), false);
        want[0] = true;
        for (const char* p = f; *p;) {
            char* e = nullptr;
            const long v = std::strtol(p, &e, 10);
            if (e == p) break;
            if (v >= 0 && v < k::kFusedGrQ8WinForms) want[(size_t) v] = true;
            p = *e == ',' ? e + 1 : e;
        }
    }
    const auto run = [&](int form, int T, const Cfg& c) {
        k::fused_gr_set_q8_win(form);
        const auto a = args(T, c);
        k::fused_gr_read_multi(a.data(), T, b.xn, s);
    };

    namespace G = strata::gpu;
    G::Stream gst = G::stream_create();
    int checks = 0, failures = 0;
    std::vector<Cfg> cfgs;
    for (int ap = 0; ap < 3; ++ap)
        for (int inject = 0; inject < 2; ++inject)
            for (int q8m = 0; q8m < 2; ++q8m)
                for (int inplace = 0; inplace < 2; ++inplace) cfgs.push_back({ap, inject != 0, q8m != 0, inplace != 0, false});
    for (int T = 1; T <= MT; ++T) {
        std::vector<Cfg> cs = cfgs;
        if (T == MT) cs.push_back({1, true, true, true, true});   // mixed not base + t * stride: the fallback
        for (const Cfg& c : cs) {
            reset();
            run(0, T, c);
            P::sync();
            const Out ref = outputs();
            // each form called directly, and (8 and 16 tokens) replayed from a captured graph as the engine runs it
            for (int fg = 2; fg < 2 * k::kFusedGrQ8WinForms; ++fg) {
                const int form = fg / 2;
                const bool graphed = fg % 2 == 1;
                if (!want[(size_t) form] || (graphed && T != 8 && T != 16 && T != MT)) continue;
                reset();
                if (graphed) {
                    k::fused_gr_set_q8_win(form);
                    const auto a = args(T, c);
                    G::Graph* g = nullptr;
                    if (!G::begin_capture(gst)) throw std::runtime_error("begin_capture");
                    k::fused_gr_read_multi(a.data(), T, b.xn, gst);
                    if (!G::end_capture(gst, &g)) throw std::runtime_error("end_capture");
                    G::graph_launch(g, gst);
                    G::stream_sync(gst);
                    G::graph_destroy(g);
                } else {
                    run(form, T, c);
                    P::sync();
                }
                const Out o = outputs();
                const std::pair<const void*, const void*> cmp[] = {
                    {ref.R.data(), o.R.data()}, {ref.Rout.data(), o.Rout.data()}, {ref.lo.data(), o.lo.data()},
                    {ref.rs.data(), o.rs.data()}, {ref.io.data(), o.io.data()}, {ref.mixed.data(), o.mixed.data()},
                    {ref.q8m.data(), o.q8m.data()}};
                const size_t bytes[] = {ref.R.size() * 4, ref.Rout.size() * 4, ref.lo.size() * 4, ref.rs.size() * 4,
                                        ref.io.size() * 4, ref.mixed.size() * 4, ref.q8m.size()};
                ++checks;
                bool ok = true;
                for (int i = 0; i < 7; ++i) {
                    const long d = first_diff(cmp[i].first, cmp[i].second, bytes[i]);
                    if (d < 0) continue;
                    if (ok) ++failures;
                    ok = false;
                    std::printf("FAIL T %2d form %d%s apply %d inject %d q8m %d inplace %d scattered %d: %s differs at byte %ld\n",
                                T, form, graphed ? " (graph)" : "", c.apply, (int) c.inject, (int) c.q8m, (int) c.inplace, (int) c.scattered, names[i], d);
                }
            }
        }
        std::printf("hc_read16_parity: window of %d tokens done, %d checks so far\n", T, checks);
    }
    std::printf("hc_read16_parity: %d checks, %d failures\n", checks, failures);
    if (reps <= 0) {   // parity alone
        G::stream_destroy(gst);
        return failures ? 1 : 0;
    }

    // times: apply all, inject rows, in place (verify's), with and without mixed's q8_1; `reps` calls captured in one
    // graph as the engine runs them (no host submission in the time)
    G::Stream st = gst;
    const int parts[] = {7, 1, 2, 4};
    const char* const pname[] = {"whole", "down", "lo", "up"};
    for (int q8m = 0; q8m < 2; ++q8m) {
        std::printf("\ntimes (us a call, min/median of %d rounds of %d calls in a graph), q8_mixed %s:\n", rounds, reps,
                    q8m ? "on" : "off");
        std::printf("  rows form   %-15s %-15s %-15s %-15s\n", pname[0], pname[1], pname[2], pname[3]);
        for (const int T : {1, 2, 4, 8, 16}) {
            const Cfg c{1, true, q8m != 0, true, false};
            reset();
            G::Graph* graph[k::kFusedGrQ8WinForms][4] = {};
            for (int form = 0; form < k::kFusedGrQ8WinForms; ++form)
                for (int p = 0; p < 4; ++p) {
                    if (!want[(size_t) form] || (p == 2 && form < 2)) continue;   // the established up adds the partials itself
                    k::fused_gr_set_q8_win(form);
                    const auto a = args(T, c);
                    k::fused_gr_q8_set_parts(7);
                    k::fused_gr_read_multi(a.data(), T, b.xn, st);   // the partials the later launches read
                    G::stream_sync(st);
                    k::fused_gr_q8_set_parts(parts[p]);
                    if (!G::begin_capture(st)) throw std::runtime_error("begin_capture");
                    for (int i = 0; i < reps; ++i) k::fused_gr_read_multi(a.data(), T, b.xn, st);
                    if (!G::end_capture(st, &graph[form][p])) throw std::runtime_error("end_capture");
                    k::fused_gr_q8_set_parts(7);
                }
            std::vector<double> us[k::kFusedGrQ8WinForms][4];
            for (int round = 0; round <= rounds; ++round)
                for (int form = 0; form < k::kFusedGrQ8WinForms; ++form)
                    for (int p = 0; p < 4; ++p) {
                        if (graph[form][p] == nullptr) continue;
                        G::graph_launch(graph[form][p], st);
                        G::stream_sync(st);
                        const auto t0 = std::chrono::steady_clock::now();
                        G::graph_launch(graph[form][p], st);
                        G::stream_sync(st);
                        const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                        if (round > 0) us[form][p].push_back(t);
                    }
            for (int form = 0; form < k::kFusedGrQ8WinForms; ++form) {
                std::printf("  %4d  %d   ", T, form);
                for (int p = 0; p < 4; ++p) {
                    G::graph_destroy(graph[form][p]);
                    auto& v = us[form][p];
                    if (v.empty()) { std::printf(" %-15s", "-"); continue; }
                    std::sort(v.begin(), v.end());
                    char cell[32];
                    std::snprintf(cell, sizeof cell, "%6.1f/%6.1f", v[0], v[v.size() / 2]);
                    std::printf(" %-15s", cell);
                }
                std::printf("\n");
            }
        }
    }
    G::stream_destroy(gst);
    k::fused_gr_set_q8_win(0);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "hc_read16_parity: %s\n", e.what());
    return 1;
}
