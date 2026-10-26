// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/prefill_split_parity.cpp - homelab STRATA_BATCH_READ_PIPE: a prompt read in many calls of smaller chunks
// leaves every per-token output and every piece of carried state exactly as one call over the whole prompt does.
//
// The lever changes only where the prompt path's chunks end (a quarter of the read instead of one 32768-token chunk)
// and how many chunks a call takes; turn checkpoints already cut reads at arbitrary positions.  What carries from one
// chunk to the next is the recurrent state: the GDN state and conv history, the QSA K/V cells (written before the
// chunk's queries read them), the indexer's tail, dead key and pooled blocks, and, on a streamed layer, the host copy
// the next chunk stages from.  This program runs those stateful steps of prefill.cpp's GDN and QSA layers - the same
// kernels with the same arguments, in the same order - on synthetic inputs (the projections are row-local: random
// rows stand in for them), once over the whole prompt and again over cut patterns: the read's chunk grid, and cuts at
// awkward places (a single token, inside an indexer block, either side of the 2,051-cell point where the selection
// becomes sparse, the GDN recurrence's 64-token form switch).  Every output row and the final state must match bit
// for bit (memcmp).  K/V K8V4 (`--kv k8v4`, as served), resident and streamed (the staging pool path).
//
// Not covered here (row-local, no state across chunks): the projections, hc, routing and the experts.  The expert
// products choose INT8 or FP16 by the chunk's mean rows an expert (prefill.cpp, `use_mmq`), so a chunk of 18.5k tokens
// or more can round its experts differently from a smaller one; that is the lever's one chunk-size-dependent
// arithmetic, and a card check, not this program, measures it.
// Runs on the OpenCL CPU device (STRATA_TEST_CPU_DEVICE=1) or a GPU.  Usage: prefill_split_parity [qsa_tokens=2304]
#include "strata/prefill/kernels.hpp"

#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include "../kernels/parity_device.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace pf = strata::prefill;
namespace par = strata::parity;

namespace {
constexpr float EPS = 1e-6f;   // prefill.cpp's
int g_checks = 0, g_fail = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::printf("  MISMATCH %s\n", what.c_str());
    }
}

std::vector<float> normal(size_t n, uint32_t seed, float mean = 0.0f, float sd = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    par::copy_out(h.data(), d, n);
    return h;
}
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
template <class T> int64_t first_diff(const std::vector<T>& a, const std::vector<T>& b, size_t row) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) return (int64_t) (i / row);
    return -1;
}

// liveness: most output elements written and nonzero (a kernel that ran nothing would compare equal too)
template <class T> bool live(const std::vector<T>& v) {
    size_t nz = 0;
    for (const T& x : v) nz += x != T{};
    return nz * 2 > v.size();
}

// the [begin, end) calls of a cut pattern over n tokens (cuts strictly inside (0, n), any order)
std::vector<std::pair<int64_t, int64_t>> parts(int64_t n, std::vector<int64_t> cuts) {
    cuts.push_back(n);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    std::vector<std::pair<int64_t, int64_t>> v;
    int64_t at = 0;
    for (const int64_t c : cuts)
        if (c > at && c <= n) { v.emplace_back(at, c); at = c; }
    return v;
}
std::string show(const std::vector<std::pair<int64_t, int64_t>>& v) {
    std::string s;
    for (const auto& [a, b] : v) s += (s.empty() ? "" : ",") + std::to_string(b - a);
    return s;
}

// ---------------------------------------------------------------- GDN (prefill.cpp's GDN layer, from the gates on)
struct GdnRun {
    std::vector<uint16_t> y16;   // [n, 6144]: what the out projection reads
    std::vector<float> state;    // the recurrent state and the conv history after the read
};

constexpr int64_t HK = 16, HV = 48, S = 128, C = 2 * HK * S + HV * S, ZV = HV * S, DCONV = 4;

// forget: the negative control - the carried state is wiped before every call but the first (must differ)
GdnRun gdn_read(int64_t n, const std::vector<std::pair<int64_t, int64_t>>& calls, bool forget = false) {
    par::Device dev;
    void* cs = par::stream();
    const std::vector<float> qkv = normal((size_t) (n * C), 11, 0.0f, 1.0f), z = normal((size_t) (n * ZV), 12),
                             ab = normal((size_t) (n * 2 * HV), 13, 0.0f, 2.0f), dt = normal(HV, 14, -1.0f, 0.5f),
                             sa = normal(HV, 15, -0.5f, 0.2f), wc = normal((size_t) (C * DCONV), 16, 0.0f, 0.5f),
                             gamma = normal(S, 17, 1.0f, 0.1f);
    const size_t state_n = (size_t) (S * HV * S + C * (DCONV - 1));
    float* state = dev.upload(std::vector<float>(state_n, 0.0f));
    float* conv = state + S * HV * S;
    const float *d_qkv = dev.upload(qkv), *d_z = dev.upload(z), *d_ab = dev.upload(ab), *d_dt = dev.upload(dt),
                *d_sa = dev.upload(sa), *d_wc = dev.upload(wc), *d_gamma = dev.upload(gamma);
    int64_t T = 0;
    for (const auto& [a, b] : calls) T = std::max(T, b - a);
    float *gate = dev.alloc<float>((size_t) (T * HV)), *beta = dev.alloc<float>((size_t) (T * HV)),
          *hbuf = dev.alloc<float>((size_t) (T * C)), *y = dev.alloc<float>((size_t) (T * ZV));
    uint16_t* y16 = dev.alloc<uint16_t>((size_t) (n * ZV));
    const std::vector<float> zero(state_n, 0.0f);
    for (const auto& [a, b] : calls) {
        const int64_t t = b - a;
        if (forget && a > 0) par::copy_in(state, zero.data(), state_n);
        pf::gdn_gates(d_ab + a * 2 * HV, d_dt, d_sa, gate, beta, t, cs);
        pf::gdn_conv(conv, d_qkv + a * C, d_wc, hbuf, t, EPS, cs);
        pf::gdn_recurrence(state, hbuf, gate, beta, d_z + a * ZV, d_gamma, EPS, y, y16 + a * ZV, t, cs);
    }
    par::sync();
    return GdnRun{down(y16, (size_t) (n * ZV)), down(state, state_n)};
}

void gdn_case(int64_t n, const std::vector<int64_t>& cuts, const GdnRun& ref) {
    const auto calls = parts(n, cuts);
    const GdnRun r = gdn_read(n, calls);
    const std::string w = "gdn [" + show(calls) + "]";
    check(same(r.y16, ref.y16), w + ": outputs (first differing token " + std::to_string(first_diff(r.y16, ref.y16, ZV)) + ")");
    check(same(r.state, ref.state), w + ": state and conv history");
    std::printf("  %-70s %s\n", w.c_str(), same(r.y16, ref.y16) && same(r.state, ref.state) ? "same bits" : "DIFFERS");
}

// ---------------------------------------------------------------- QSA (prefill.cpp's QSA layer, K8V4)
struct QsaRun {
    std::vector<uint16_t> out16;   // [n, 6144]: the gated attention the out projection reads
    std::vector<int32_t> ids;      // [n, cap]: every query's selected cells
    std::vector<int8_t> k_q;       // the cells' K codes, scales and V blocks [0, n)
    std::vector<uint16_t> k_scale;
    std::vector<uint8_t> v_q4;
    std::vector<float> idx;        // the indexer: tail, dead, pooled
    std::vector<int8_t> host_k;    // streamed: the host copy the next chunk stages from
    std::vector<uint8_t> host_v;
};

QsaRun qsa_read(int64_t n, const std::vector<std::pair<int64_t, int64_t>>& calls, bool streamed, bool forget = false) {
    par::Device dev;
    void* cs = par::stream();
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_cells = (n + 1023) / 1024 * 1024 + 1024, pages = max_cells / s.page_size;
    const int64_t rows = pages * s.n_head_kv * s.page_size;
    const int64_t q4 = k::kv_q4_bytes_per_head((int) s.head_dim);
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s), max_blocks = max_cells / s.idx_block + 2;
    const int64_t pooled_rows = max_cells / s.idx_block + 2;
    constexpr int64_t sel_batch = 256, attn_batch = 32;   // prefill.cpp's
    // the cells' pools (identity page table), the host copy and the staging pool of a streamed layer
    std::vector<int32_t> ident((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
    const int32_t* table = dev.upload(ident);
    auto pools = [&] {
        k::KvHostPools p;
        p.k_q = dev.upload(std::vector<int8_t>((size_t) (rows * s.head_dim), 0));
        p.k_scale = dev.upload(std::vector<uint16_t>((size_t) (rows * (s.head_dim / 64)), 0));
        p.v_q4 = dev.upload(std::vector<uint8_t>((size_t) (rows * q4), 0));
        return p;
    };
    const k::KvHostPools vram = pools(), host = streamed ? pools() : k::KvHostPools{},
                         stage = streamed ? pools() : k::KvHostPools{};
    auto attn_pools = [&](const k::KvHostPools& h) {
        k::QsaAttnPools p;
        p.k_q = h.k_q; p.k_scale = h.k_scale; p.v_q4 = h.v_q4; p.page_table = table;
        return p;
    };
    float* tail = dev.upload(std::vector<float>((size_t) ((s.idx_block - 1) * s.idx_dim), 0.0f));
    float* dead = dev.upload(std::vector<float>((size_t) s.idx_dim, 0.0f));
    float* pooled = dev.upload(std::vector<float>((size_t) (pooled_rows * s.idx_dim), 0.0f));
    int32_t* block_pos = dev.upload(std::vector<int32_t>(1, 0));
    const k::QsaIndexerBuffers ib{tail, dead, pooled, block_pos};
    // the projections' outputs (row-local, so random rows stand in for them) and the norms' weights
    const std::vector<float> Kh = normal((size_t) (n * 512), 21), Vh = normal((size_t) (n * 512), 22),
                             Qh = normal((size_t) (n * 12288), 23), Ih = normal((size_t) (n * 128), 24),
                             QIh = normal((size_t) (n * 512), 25), wkn = normal(256, 26, 1.0f, 0.1f),
                             wqn = normal(256, 27, 1.0f, 0.1f), wiqn = normal(128, 28, 1.0f, 0.1f),
                             wikn = normal(128, 29, 1.0f, 0.1f);
    const float *d_wkn = dev.upload(wkn), *d_wqn = dev.upload(wqn), *d_wiqn = dev.upload(wiqn),
                *d_wikn = dev.upload(wikn), *d_Qf = dev.upload(Qh);
    int64_t T = 0;
    for (const auto& [a, b] : calls) T = std::max(T, b - a);
    float *Kc = dev.alloc<float>((size_t) (T * 512)), *Vc = dev.alloc<float>((size_t) (T * 512)),
          *q = dev.alloc<float>((size_t) (T * 6144)), *idx_raw = dev.alloc<float>((size_t) (T * 128)),
          *q_idx = dev.alloc<float>((size_t) (T * 512)), *attn = dev.alloc<float>((size_t) (T * 6144)),
          *sel_scores = dev.alloc<float>((size_t) (sel_batch * max_blocks)),
          *scratch = dev.alloc<float>((size_t) (attn_batch * k::qsa_decode_attn_scratch_floats(cap, s)));
    int32_t *steps_dev = dev.alloc<int32_t>((size_t) (T * k::kStepCount)),
            *sel = dev.alloc<int32_t>((size_t) (n * cap));
    uint16_t* out16 = dev.alloc<uint16_t>((size_t) (n * 6144));
    std::vector<int32_t> steps_host((size_t) (T * k::kStepCount));
    const k::KvHostPools hk = k::kv_hybrid_k_half(host), hv = k::kv_hybrid_v_half(host),
                         sk = k::kv_hybrid_k_half(stage), sv = k::kv_hybrid_v_half(stage);
    const std::vector<float> zero((size_t) (pooled_rows * s.idx_dim), 0.0f);
    for (const auto& [p0, b] : calls) {
        const int64_t t = b - p0;
        if (forget && p0 > 0) {   // the indexer's carried keys wiped
            par::copy_in(tail, zero.data(), (size_t) ((s.idx_block - 1) * s.idx_dim));
            par::copy_in(dead, zero.data(), (size_t) s.idx_dim);
            par::copy_in(pooled, zero.data(), (size_t) (pooled_rows * s.idx_dim));
        }
        par::copy_in(Kc, Kh.data() + p0 * 512, (size_t) (t * 512));
        par::copy_in(Vc, Vh.data() + p0 * 512, (size_t) (t * 512));
        par::copy_in(idx_raw, Ih.data() + p0 * 128, (size_t) (t * 128));
        par::copy_in(q_idx, QIh.data() + p0 * 512, (size_t) (t * 512));
        for (int64_t i = 0; i < t; ++i) k::qsa_step_fill(steps_host.data() + i * k::kStepCount, p0 + i, s);
        par::copy_in(steps_dev, steps_host.data(), (size_t) (t * k::kStepCount));
        const float* Qf = d_Qf + p0 * 12288;
        pf::rms_rows(Kc, d_wkn, t * 2, 256, 256, EPS, cs);
        pf::rope(Kc, t, 2, 256, 512, p0, k::rope_scaling(), cs);
        if (streamed)   // this layer's cells [0, p0) from the host copy into the staging pool
            k::kv_stage_from_host(attn_pools(stage), host, k::kKvHybrid, (p0 + s.page_size - 1) / s.page_size, s, cs);
        k::fwht256_inplace_cuda(Vc, t * 2, cs);   // K8V4: V rotated, K not
        pf::kv_append(Kc, Kc, t, p0, table, s.page_size, nullptr, nullptr, vram.k_q, vram.k_q, vram.k_scale,
                      vram.k_scale, cs, streamed ? &hk : nullptr, streamed ? &sk : nullptr);
        k::kv_append_q4(vram.v_q4, vram.v_q4, table, p0, t, Vc, Vc, s, cs, streamed ? &hv : nullptr,
                        streamed ? &sv : nullptr);
        pf::split_q(Qf, q, t, cs);
        pf::rms_rows(q, d_wqn, t * 24, 256, 256, EPS, cs);
        pf::rope(q, t, 24, 256, 6144, p0, k::rope_scaling(), cs);
        pf::rms_rows(q_idx, d_wiqn, t * 4, 128, 128, EPS, cs);
        pf::rope(q_idx, t, 4, 128, 512, p0, k::rope_scaling(), cs);
        k::native_qsa_indexer_append_batch(idx_raw, t, p0, 0, d_wikn, EPS, ib, s, max_cells, k::rope_scaling(), cs);
        int32_t* ids = sel + p0 * cap;
        for (int64_t t0 = 0; t0 < t; t0 += sel_batch) {
            const int64_t nb = std::min(sel_batch, t - t0);
            const int32_t* steps0 = steps_dev + t0 * k::kStepCount;
            const int64_t active = (int64_t) steps_host[(size_t) ((t0 + nb - 1) * k::kStepCount + k::kStepNBid)] + 1;
            if (!k::qsa_block_scores_tc(pooled, dead, q_idx + t0 * 512, steps0, nb, max_blocks, s, sel_scores, cs,
                                        active))
                k::qsa_block_scores(pooled, dead, q_idx + t0 * 512, steps0, nb, max_blocks, s, sel_scores, cs, active);
            k::qsa_block_topk(sel_scores, steps0, nb, max_blocks, cap, s, ids + t0 * cap, cs);
        }
        const k::QsaAttnPools rd = attn_pools(streamed ? stage : vram);
        if (!k::qsa_prompt_attn_batch(q, rd, ids, steps_dev, cap, s, attn, t, cs))
            for (int64_t t0 = 0; t0 < t; t0 += attn_batch) {
                const int64_t nb = std::min(attn_batch, t - t0);
                k::qsa_decode_attn_batch(q + t0 * 6144, rd, ids + t0 * cap, steps_dev + t0 * k::kStepCount, cap, s,
                                         scratch, attn + t0 * 6144, nb, cs);
            }
        k::fwht256_inplace_cuda(attn, t * 24, cs);
        pf::gate_attn(attn, Qf, out16 + p0 * 6144, t, cs);
        par::sync();   // the host buffers above are overwritten by the next call
    }
    par::sync();
    QsaRun r;
    r.out16 = down(out16, (size_t) (n * 6144));
    // a query's selection: its first `width` ids (the rest of the row is not read)
    const std::vector<int32_t> all = down(sel, (size_t) (n * cap));
    for (int64_t i = 0; i < n; ++i) {
        int32_t st[k::kStepCount];
        k::qsa_step_fill(st, i, s);
        r.ids.insert(r.ids.end(), all.begin() + i * cap, all.begin() + i * cap + st[k::kStepWidth]);
    }
    const int64_t used = (n + s.page_size - 1) / s.page_size * s.n_head_kv * s.page_size;   // the pages of [0, n)
    r.k_q = down(vram.k_q, (size_t) (used * s.head_dim));
    r.k_scale = down(vram.k_scale, (size_t) (used * (s.head_dim / 64)));
    r.v_q4 = down(vram.v_q4, (size_t) (used * q4));
    r.idx = down(tail, (size_t) ((s.idx_block - 1) * s.idx_dim));
    const std::vector<float> dk = down(dead, (size_t) s.idx_dim),
                             pk = down(pooled, (size_t) ((n / s.idx_block + 1) * s.idx_dim));
    r.idx.insert(r.idx.end(), dk.begin(), dk.end());
    r.idx.insert(r.idx.end(), pk.begin(), pk.end());
    if (streamed) {
        r.host_k = down(host.k_q, (size_t) (used * s.head_dim));
        r.host_v = down(host.v_q4, (size_t) (used * q4));
    }
    return r;
}

void qsa_case(int64_t n, const std::vector<int64_t>& cuts, bool streamed, const QsaRun& ref) {
    const auto calls = parts(n, cuts);
    const QsaRun r = qsa_read(n, calls, streamed);
    const std::string w = std::string("qsa k8v4 ") + (streamed ? "streamed" : "resident") + " [" + show(calls) + "]";
    const bool out = same(r.out16, ref.out16), ids = same(r.ids, ref.ids),
               kv = same(r.k_q, ref.k_q) && same(r.k_scale, ref.k_scale) && same(r.v_q4, ref.v_q4),
               idx = same(r.idx, ref.idx), hostc = same(r.host_k, ref.host_k) && same(r.host_v, ref.host_v);
    check(out, w + ": attention outputs (first differing token " + std::to_string(first_diff(r.out16, ref.out16, 6144)) + ")");
    check(ids, w + ": selections");
    check(kv, w + ": K/V cells");
    check(idx, w + ": indexer tail, dead key, pooled blocks");
    check(hostc, w + ": host copy");
    std::printf("  %-70s %s\n", w.c_str(), out && ids && kv && idx && hostc ? "same bits" : "DIFFERS");
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t nq = argc > 1 ? std::strtoll(argv[1], nullptr, 10) : 2304;   // past 2,051: sparse selections too
    const int64_t ng = 1000;
    std::printf("prefill_split_parity: GDN %lld tokens, QSA %lld tokens (selection sparse from 2,052 cells)\n",
                (long long) ng, (long long) nq);
    // the GDN layer: the whole read against the read's chunk grid and awkward cuts
    const GdnRun g0 = gdn_read(ng, parts(ng, {}));
    check(live(g0.y16) && live(g0.state), "gdn: the whole read wrote its outputs and state");
    {   // negative control: the same cuts with the state wiped between calls must not match
        const GdnRun x = gdn_read(ng, parts(ng, {256, 512, 768}), true);
        check(!same(x.y16, g0.y16), "gdn control: a read that forgets its state between calls differs");
        std::printf("  %-70s %s\n", "gdn control (state wiped between calls)", same(x.y16, g0.y16) ? "SAME (bad)" : "differs (good)");
    }
    gdn_case(ng, {256, 512, 768}, g0);                        // a 256-token chunk grid (multi-chunk calls the same)
    gdn_case(ng, {1, 2, 63, 64, 128, 129, 517, 999}, g0);     // single tokens, under and over the 64-token form switch
    // the QSA layer, resident and streamed
    for (const bool streamed : {false, true}) {
        const QsaRun q0 = qsa_read(nq, parts(nq, {}), streamed);
        check(live(q0.out16) && live(q0.k_q) && live(q0.idx), "qsa: the whole read wrote its outputs, cells and keys");
        if (!streamed) {   // negative control: the indexer's keys wiped between calls must change the read
            const QsaRun x = qsa_read(nq, parts(nq, {1024, 2048}), streamed, true);
            const bool eq = same(x.out16, q0.out16) && same(x.ids, q0.ids);
            check(!eq, "qsa control: a read that forgets its indexer keys between calls differs");
            std::printf("  %-70s %s\n", "qsa control (indexer keys wiped between calls)", eq ? "SAME (bad)" : "differs (good)");
        }
        qsa_case(nq, {1024, 2048}, streamed, q0);             // the read's chunk grid
        qsa_case(nq, {1, 3, 517, 1029, 2050, 2051, 2052, nq - 1}, streamed, q0);   // inside blocks, the sparse switch
    }
    std::printf("prefill_split_parity: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
