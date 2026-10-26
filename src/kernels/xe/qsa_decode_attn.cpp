// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/qsa_decode_attn.cpp - the Xe port of Strata's src/kernels/cuda/qsa_decode_attn.cu (see the header): split-K
// decode attention over the selected cells for fp16, INT8 and Q4_0 KV, then a merge of the chunk partials.
//
// The head size HD and the query heads per KV head G are template parameters: (256, 12) is Qwen3.8's QSA, (256, 8)
// Ornith-1.5's and (128, 10) llm-jp-4.1's full attention, which read EVERY cell (`qsa_dense_attn_*`: no selection,
// cell c of the cache for slot c).  A work-group has HD work-items, one per dimension in the value loop; a lane holds
// HD/32 dimensions in the score loop.  For (256, 12) every expression is the one this file had with HD and G fixed.
//
// A warp is a sub-group of 32.  CUDA used the __expf intrinsic here; this uses the precise exp.  Explicit fmaf calls
// stay fused (sycl::fma), and the library's -ffp-contract=off leaves every other multiply-add as written.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_qsa_decode_attn
#include "strata/kernels/device_arch_tag.hpp"
#include "param_table.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace strata::kernels {
namespace {

constexpr int CHUNK = 64;        // cells per work-group
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) {
    auto& queue = core::Runtime::get().stream(stream);
    return queue;
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float warp_max(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}
inline float h2f(uint16_t bits) { return (float) sycl::bit_cast<sycl::half>(bits); }

// The G <= 16 heads' lane sums (`part[G..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5 (upstream fd95405).  Lane l ends holding head head_of_lane(l); lanes l and l^1
// agree.
inline float reduce12(const sycl::sub_group& sg, const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + sycl::permute_group_by_xor(sg, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + sycl::permute_group_by_xor(sg, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + sycl::permute_group_by_xor(sg, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + sycl::permute_group_by_xor(sg, hi2 ? r2[0] : r2[1], 2);
    return r1 + sycl::permute_group_by_xor(sg, r1, 1);
}
inline int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own work-item), per KV format.
template <int HD, int KV_MODE>
inline float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return h2f(p.v_pool[row * HD + d]);
    } else if constexpr (KV_MODE == 1) {
        const float sc = h2f(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]);
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = h2f(blk->d);
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

// N = HD / 32 consecutive values of one cell's key or value row, dimensions [d0, d0+N).  KV_MODE 0: FP16, 1: INT8,
// 2: rotated Q4_0, 3: the hybrid K8V4 (K as INT8, V as rotated Q4_0; upstream 2aa8f72).  N divides 16, so the run
// stays inside one Q4_0 half (the low nibbles hold elements 0-15 of a block, the high ones 16-31).
template <int HD, int KV_MODE>
inline void loadn(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int N = HD / WARP;
    if constexpr (KV_MODE == 3) {
        if (value) loadn<HD, 2>(p, true, row, d0, out);
        else loadn<HD, 1>(p, false, row, d0, out);
    } else if constexpr (KV_MODE == 0) {
        const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
        for (int j = 0; j < N; ++j) out[j] = h2f(base[j]);
    } else if constexpr (KV_MODE == 1) {
        const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
        const float sc = h2f((value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]);
        for (int j = 0; j < N; ++j) out[j] = (float) codes[j] * sc;
    } else {
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d0 / QK4_0;
        const int rem = d0 % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
        const float d = h2f(blk->d);
        const uint8_t* bytes = blk->qs + (rem & 15);
        if (rem < 16) {
            for (int k = 0; k < N; ++k) out[k] = (float) ((int) (bytes[k] & 0x0F) - 8) * d;
        } else {
            for (int k = 0; k < N; ++k) out[k] = (float) ((int) (bytes[k] >> 4) - 8) * d;
        }
    }
}

// One lane's share of q.k: the N products summed left to right, which for N = 8 is the expression this file wrote out.
template <int N>
inline float dotn(const float* k, const float* q) {
    float s = k[0] * q[0];
#pragma unroll
    for (int e = 1; e < N; ++e) s = s + k[e] * q[e];
    return s;
}

// Query z's pools: the one state's (every query of a sequence), or row z's table entry (a batch window's slots).
template <int KV_MODE>
inline QsaAttnPools pools_of(const QsaAttnPools& p, size_t) { return p; }
template <int KV_MODE>
inline QsaAttnPools pools_of(const QsaAttnSlots& t, size_t z) {
    QsaAttnPools p;
    p.page_table = t.page_table[z];
    if constexpr (KV_MODE == 0) {
        p.k_pool = static_cast<const uint16_t*>(t.k[z]);
        p.v_pool = static_cast<const uint16_t*>(t.v[z]);
    } else if constexpr (KV_MODE == 1) {
        p.k_q = static_cast<const int8_t*>(t.k[z]);
        p.v_q = static_cast<const int8_t*>(t.v[z]);
        p.k_scale = t.k_scale[z];
        p.v_scale = t.v_scale[z];
    } else if constexpr (KV_MODE == 2) {
        p.k_q4 = static_cast<const uint8_t*>(t.k[z]);
        p.v_q4 = static_cast<const uint8_t*>(t.v[z]);
    } else {
        p.k_q = static_cast<const int8_t*>(t.k[z]);
        p.k_scale = t.k_scale[z];
        p.v_q4 = static_cast<const uint8_t*>(t.v[z]);
    }
    return p;
}

template <int HD, int G, int KV_MODE, class Src>
sycl::event launch_chunk(sycl::queue& queue, const float* q_all, const Src src, const int32_t* ids_all,
                         const int32_t* step_all, int n_kv_heads, int page_size, float scale, float* part_acc_all,
                         float* part_m_all, float* part_l_all, int n_chunks, size_t n_q, int cap,
                         long long scratch_stride) {
    static_assert(G >= 1 && G <= 16, "reduce12 sums at most 16 heads");
    static_assert(HD % 128 == 0 && HD <= 256, "a lane holds HD/32 dimensions inside one Q4_0 half");
    constexpr int THREADS = HD;           // one work-item per dimension in the value loop
    constexpr int WARPS = THREADS / WARP;
    constexpr int N = HD / WARP;          // dimensions a lane holds in the score loop
    constexpr int GP = (G + 3) / 4 * 4;   // a cell's weights padded to whole float4s
    // ids_all null: every cell, slot c of the chunk is cell c0 + c, and the count is the cache's (step[kStepNKv])
    const bool dense = ids_all == nullptr;
    return queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(G * HD), hd);          // this KV head's query heads
        sycl::local_accessor<float, 1> sp(sycl::range<1>((size_t) G * CHUNK), hd);       // scores
        sycl::local_accessor<sycl::float4, 1> spt(sycl::range<1>(CHUNK * GP / 4), hd);  // probabilities, cell-major
        sycl::local_accessor<long long, 1> srow(sycl::range<1>(CHUNK), hd);     // pool row of each cell
        hd.parallel_for(sycl::nd_range<3>({n_q, (size_t) n_kv_heads, (size_t) n_chunks * THREADS}, {1, 1, THREADS}),
                        xe::fits("qsa_decode_attn chunk", [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const size_t z = it.get_group(0);
            const QsaAttnPools p = pools_of<KV_MODE>(src, z);
            // batched form: query z, with its own q row, selection, step and scratch
            const float* q = q_all + z * (size_t) (n_kv_heads * G) * HD;
            const int32_t* ids = dense ? nullptr : ids_all + z * (size_t) cap;
            const int32_t* step = step_all + z * kStepCount;
            float* part_acc = part_acc_all + z * (size_t) scratch_stride;
            float* part_m = part_m_all + z * (size_t) scratch_stride;
            float* part_l = part_l_all + z * (size_t) scratch_stride;
            const int n_ids = dense ? step[kStepNKv] : step[kStepWidth];
            const int chunk = (int) it.get_group(2), kvh = (int) it.get_group(1);
            const int t = (int) it.get_local_id(2);
            const int lane = (int) sg.get_local_linear_id(), warp = (int) sg.get_group_linear_id();
            const int c0 = chunk * CHUNK;
            const int n_here = sycl::min(CHUNK, n_ids - c0);
            const int slot = kvh * n_chunks + chunk;
            if (n_here <= 0) {                                    // uniform across the group
                if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
                return;
            }
            for (int i = t; i < G * HD; i += THREADS) sq[i] = q[(size_t) (kvh * G) * HD + i];
            if (t < CHUNK) {
                long long r = -1;
                if (t < n_here) {
                    const int cell = dense ? c0 + t : ids[c0 + t];
                    const long long page = (long long) p.page_table[cell / page_size];
                    // a block the KV streaming could not make resident keeps page -1; its cells are masked (score
                    // -FLT_MAX, weight 0) instead of being read from before the pool (upstream f3925cf)
                    if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
                }
                srow[t] = r;
            }
            sycl::group_barrier(grp);
            // scores: each sub-group takes cells warp, warp+WARPS, ...; each lane holds N of the HD dimensions.  NC
            // cells per step (warp + WARPS * (i + j)): a lane's q slice comes out of local memory once for all of them
            // (upstream fd95405); every dot product is the same expression as for a single cell, so the scores are
            // unchanged.
            constexpr int NC = 2;
            const int hd_lane = head_of_lane(lane);
            for (int i = 0; i < CHUNK / WARPS; i += NC) {
                int cc[NC];
                bool all_ok = true;
                for (int j = 0; j < NC; ++j) {
                    cc[j] = warp + WARPS * (i + j);
                    all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
                }
                if (all_ok) {
                    float kk[NC][N];
                    for (int j = 0; j < NC; ++j) loadn<HD, KV_MODE>(p, false, srow[cc[j]], lane * N, kk[j]);
                    float pp[NC][16];
#pragma unroll
                    for (int h = 0; h < G; ++h) {
                        const float* qa = &sq[h * HD + lane * N];
#pragma unroll
                        for (int j = 0; j < NC; ++j) pp[j][h] = dotn<N>(kk[j], qa);
                    }
#pragma unroll
                    for (int j = 0; j < NC; ++j) {
                        for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                        const float sj = reduce12(sg, pp[j], lane);
                        if ((lane & 1) == 0 && hd_lane < G) sp[hd_lane * CHUNK + cc[j]] = sj * scale;
                    }
                } else {
                    for (int j = 0; j < NC; ++j) {   // a masked or out-of-range cell in the step: one cell at a time
                        const int c = cc[j];
                        if (c >= n_here || srow[c] < 0) {
                            if (lane < G) sp[lane * CHUNK + c] = -FLT_MAX;
                            continue;
                        }
                        float kn[N];
                        loadn<HD, KV_MODE>(p, false, srow[c], lane * N, kn);
                        float part[16];
                        for (int h = 0; h < G; ++h) part[h] = dotn<N>(kn, &sq[h * HD + lane * N]);
                        for (int h = G; h < 16; ++h) part[h] = 0.0f;
                        const float sc = reduce12(sg, part, lane);
                        if ((lane & 1) == 0 && hd_lane < G) sp[hd_lane * CHUNK + c] = sc * scale;
                    }
                }
            }
            sycl::group_barrier(grp);
            // per-head chunk max and exp-sum: sub-group w handles heads w, w+WARPS, ...
            for (int h = warp; h < G; h += WARPS) {
                const float a = sp[h * CHUNK + lane], b = sp[h * CHUNK + lane + 32];
                const float m = warp_max(sg, sycl::fmax(a, b));
                const float ea = (lane < n_here && srow[lane] >= 0) ? sycl::exp(a - m) : 0.0f;
                const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? sycl::exp(b - m) : 0.0f;
                float* pt = reinterpret_cast<float*>(&spt[0]);
                pt[lane * GP + h] = ea;
                pt[(lane + 32) * GP + h] = eb;
                const float l = warp_sum(sg, ea + eb);
                if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
            }
            sycl::group_barrier(grp);
            // values: work-item t owns dimension t for all G heads.
            float acc[GP];
            for (int h = 0; h < GP; ++h) acc[h] = 0.0f;
            // one cell's weights (GP/4 float4 loads from local memory) into the accumulators; the cells are folded in
            // ascending order, so the sums are the plain loop's, bit for bit (upstream fd95405).  The padding slots
            // past G accumulate whatever local memory holds and are never stored.
            const auto fold = [&](int c, float v) {
#pragma unroll
                for (int g4 = 0; g4 < GP / 4; ++g4) {
                    const sycl::float4 w = spt[(size_t) c * (GP / 4) + g4];
                    acc[4 * g4] = sycl::fma(w.x(), v, acc[4 * g4]);
                    acc[4 * g4 + 1] = sycl::fma(w.y(), v, acc[4 * g4 + 1]);
                    acc[4 * g4 + 2] = sycl::fma(w.z(), v, acc[4 * g4 + 2]);
                    acc[4 * g4 + 3] = sycl::fma(w.w(), v, acc[4 * g4 + 3]);
                }
            };
            int c = 0;
            // four cells at a time with their V loads issued together; a group holding a masked cell (page -1,
            // rare) leaves this loop and the one below finishes the chunk
            for (; c + 4 <= n_here; c += 4) {
                const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
                if ((r0 | r1 | r2 | r3) < 0) break;
                const float v0 = load_v1<HD, KV_MODE>(p, r0, t), v1 = load_v1<HD, KV_MODE>(p, r1, t);
                const float v2 = load_v1<HD, KV_MODE>(p, r2, t), v3 = load_v1<HD, KV_MODE>(p, r3, t);
                fold(c, v0);
                fold(c + 1, v1);
                fold(c + 2, v2);
                fold(c + 3, v3);
            }
            for (; c < n_here; ++c) {
                if (srow[c] < 0) continue;   // masked above, weight 0
                fold(c, load_v1<HD, KV_MODE>(p, srow[c], t));
            }
            for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
        }));
    });
}

// ---- STRATA_QSA_ATTN_LANES=1 (homelab): launch_chunk's chunk in 16-lane sub-groups with one CELL a lane in the score
// pass, the same bits.  Measured on the B70 at 16 rows (batch_window_ubench, int8): launch_chunk takes 270 us for 35 MB
// of K/V (~3x the 456 GB/s floor).  Its score pass splits a cell's 256 dimensions over 32 lanes, so every lane reads
// its 8 dims x 12 heads of the query out of local memory for every cell pair (384 KB of local reads a 64-cell chunk
// against 67 KB of K/V) and reduces 12 heads across the lanes; its value pass reads the 12 weights of every cell out of
// local memory in every work-item (786 KB a chunk).  Here a lane holds its cell's 12 scores in registers, takes the
// query as a 16-wide window a load and broadcasts each dimension from the lane that holds it (a register region, no
// local memory), and builds each score with the summation tree reduce12 performs across lanes (leaf B = dims
// [8B, 8B+8) summed left to right; then pairs B ^ 16, ^ 8, ^ 4, ^ 2, ^ 1), so every score is launch_chunk's bit for
// bit.  The value pass gives sub-group w the dims [64w, 64w+64) (4 a lane, one 64-byte line a cell) and broadcasts
// the weights of 16 cells from the lanes that computed them; each accumulator folds the cells in ascending order with
// launch_chunk's fma, so the partials, and so the merged output, are its bits.  The chunks (CHUNK cells), the scratch
// layout and the merge are launch_chunk's.  FP16 and INT8 K/V (the Q4_0 V of modes 2 and 3 keeps launch_chunk);
// head_dim 256 only.
constexpr int kLanesSg = 16;
constexpr int kLanesVPrefetch = 8;   // value-pass cells whose loads are issued before their folds

// K row `row`, dims [d0, d0+8): loadn's values (KV modes 0 and 1) in one vector load; `ks` the row's four scales.
template <int KV_MODE>
inline void lanes_k8(const QsaAttnPools& p, long long row, int d0, const float (&ks)[4], float (&out)[8]) {
    if constexpr (KV_MODE == 0) {
        const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.k_pool + row * 256 + d0);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = h2f(v[j]);
    } else {
        const sycl::vec<int8_t, 8> v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.k_q + row * 256 + d0);
        const float sc = ks[d0 / KV_Q8_GROUP];
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) v[j] * sc;
    }
}

// V row `row`, dims [d0, d0+4): the raw codes (one load), then load_v1's values given the row's scale for d0's group.
template <int KV_MODE>
using LanesRaw = std::conditional_t<KV_MODE == 0, sycl::vec<uint16_t, 4>, sycl::vec<int8_t, 4>>;
template <int KV_MODE>
inline LanesRaw<KV_MODE> lanes_v4_raw(const QsaAttnPools& p, long long row, int d0) {
    if constexpr (KV_MODE == 0) return *reinterpret_cast<const sycl::vec<uint16_t, 4>*>(p.v_pool + row * 256 + d0);
    else return *reinterpret_cast<const sycl::vec<int8_t, 4>*>(p.v_q + row * 256 + d0);
}
template <int KV_MODE>
inline float lanes_v_scale(const QsaAttnPools& p, long long row, int d0) {
    if constexpr (KV_MODE == 0) return 1.0f;
    else return h2f(p.v_scale[row * (256 / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]);
}
template <int KV_MODE>
inline float lanes_v1(const LanesRaw<KV_MODE>& raw, int k, float sc) {
    if constexpr (KV_MODE == 0) return h2f(raw[k]);
    else return (float) raw[k] * sc;
}

// The score tree of HG heads: leaf B is dotn<8> over dims [8B, 8B+8); level L pairs B with B + 2^L (L = 4 first, as
// reduce12's xor 16), so tree<0>(0) is the sum reduce12 leaves in every lane.
template <int G, int L, class Leaf>
inline void lanes_tree(int b, float (&out)[G], const Leaf& leaf) {
    if constexpr (L == 5) {
        leaf(b, out);
    } else {
        float lo[G], hi[G];
        lanes_tree<G, L + 1>(b, lo, leaf);
        lanes_tree<G, L + 1>(b + (1 << L), hi, leaf);
#pragma unroll
        for (int h = 0; h < G; ++h) out[h] = lo[h] + hi[h];
    }
}

template <int HD, int G, int KV_MODE, class Src>
sycl::event launch_chunk_lanes(sycl::queue& queue, const float* q_all, const Src src, const int32_t* ids_all,
                               const int32_t* step_all, int n_kv_heads, int page_size, float scale, float* part_acc_all,
                               float* part_m_all, float* part_l_all, int n_chunks, size_t n_q, int cap,
                               long long scratch_stride) {
    static_assert(HD == 256 && (KV_MODE == 0 || KV_MODE == 1), "the lanes form: head_dim 256, FP16 or INT8 K/V");
    static_assert(G >= 1 && G <= 16, "launch_chunk's head bound");
    constexpr int SG = kLanesSg;
    constexpr int NSG = CHUNK / SG;          // 4 sub-groups: a cell a lane in the score pass
    constexpr int THREADS = CHUNK;
    constexpr int VD = HD / (NSG * SG);      // 4 dims a lane in the value pass
    constexpr int GP = (G + 3) / 4 * 4;
    static_assert(VD == 4 && HD / NSG == KV_Q8_GROUP, "a value-pass sub-group spans one scale group");
    const bool dense = ids_all == nullptr;
    return queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<sycl::float4, 1> sp(sycl::range<1>(CHUNK * GP / 4), hd);   // scores, then weights, cell-major
        sycl::local_accessor<long long, 1> srow(sycl::range<1>(CHUNK), hd);
        hd.parallel_for(sycl::nd_range<3>({n_q, (size_t) n_kv_heads, (size_t) n_chunks * THREADS}, {1, 1, THREADS}),
                        xe::fits("qsa_decode_attn lanes", [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(kLanesSg)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const size_t z = it.get_group(0);
            const QsaAttnPools p = pools_of<KV_MODE>(src, z);
            const float* q = q_all + z * (size_t) (n_kv_heads * G) * HD;
            const int32_t* ids = dense ? nullptr : ids_all + z * (size_t) cap;
            const int32_t* step = step_all + z * kStepCount;
            float* part_acc = part_acc_all + z * (size_t) scratch_stride;
            float* part_m = part_m_all + z * (size_t) scratch_stride;
            float* part_l = part_l_all + z * (size_t) scratch_stride;
            const int n_ids = dense ? step[kStepNKv] : step[kStepWidth];
            const int chunk = (int) it.get_group(2), kvh = (int) it.get_group(1);
            const int t = (int) it.get_local_id(2);
            const int lane = (int) sg.get_local_linear_id(), w = (int) sg.get_group_linear_id();
            const int c0 = chunk * CHUNK;
            const int n_here = sycl::min(CHUNK, n_ids - c0);
            const int slot = kvh * n_chunks + chunk;
            if (n_here <= 0) {                                    // uniform across the group
                if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
                return;
            }
            float* spf = reinterpret_cast<float*>(&sp[0]);
            // ---- scores: lane = cell c of the chunk
            const int c = w * SG + lane;
            long long r = -1;
            if (c < n_here) {
                const int cell = dense ? c0 + c : ids[c0 + c];
                const long long page = (long long) p.page_table[cell / page_size];
                if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);   // else masked
            }
            srow[c] = r;
            const long long rr = r >= 0 ? r : 0;   // a masked lane reads row 0 and drops its scores
            float ks[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            if constexpr (KV_MODE == 1) {
#pragma unroll
                for (int g = 0; g < 4; ++g) ks[g] = h2f(p.k_scale[rr * (HD / KV_Q8_GROUP) + g]);
            }
            const float* qk = q + (size_t) (kvh * G) * HD;
            // HG heads a pass (the K row read again from the cache for the next ones): G heads at once hold G x 4
            // pending tree sums and spill.  Lane l loads dim 8B + (l & 7) of the query for leaf B, broadcast from lanes
            // 0..7; a window of its own a leaf, so no load is shared between leaves far apart in the tree.
            constexpr int HG = G / 2;
            static_assert(G % 2 == 0, "two passes of G / 2 heads");
#pragma unroll 1
            for (int h0 = 0; h0 < G; h0 += HG) {
                const auto leaf = [&](int b, float (&out)[HG]) {
                    float kk[8];
                    lanes_k8<KV_MODE>(p, rr, b * 8, ks, kk);
#pragma unroll
                    for (int h = 0; h < HG; ++h) {
                        const float qw = qk[(h0 + h) * HD + b * 8 + (lane & 7)];
                        float s = kk[0] * sycl::select_from_group(sg, qw, 0);
#pragma unroll
                        for (int e = 1; e < 8; ++e) s = s + kk[e] * sycl::select_from_group(sg, qw, e);
                        out[h] = s;
                    }
                };
                // the tree's top two levels as a loop over its four subtrees of level 2 (leaves B = base + 4 j, the
                // K row's four 64-byte lines each), so the compiler cannot hoist all 32 leaves' loads and spill them:
                // e = (T(0) + T(2)) + (T(1) + T(3)), T(base) = lanes_tree<HG, 2>(base)
                float sc[HG], hi[HG];
#pragma unroll 1
                for (int k = 0; k < 4; ++k) {
                    float t[HG];
                    lanes_tree<HG, 2>(2 * (k & 1) + (k >> 1), t, leaf);
#pragma unroll
                    for (int h = 0; h < HG; ++h) {
                        if (k == 0) sc[h] = t[h];
                        else if (k == 1) sc[h] = sc[h] + t[h];
                        else if (k == 2) hi[h] = t[h];
                        else sc[h] = sc[h] + (hi[h] + t[h]);
                    }
                }
#pragma unroll
                for (int h = 0; h < HG; ++h) spf[c * GP + h0 + h] = r >= 0 ? sc[h] * scale : -FLT_MAX;
            }
            sycl::group_barrier(grp);
            // ---- per-head chunk max and exp-sum (launch_chunk's: its lane l held cells l and l + 32)
            for (int h = w; h < G; h += NSG) {
                float a[4], e[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) a[k] = spf[(lane + SG * k) * GP + h];
                float m = sycl::fmax(sycl::fmax(a[0], a[2]), sycl::fmax(a[1], a[3]));
                for (int o = SG / 2; o > 0; o >>= 1) m = sycl::fmax(m, sycl::permute_group_by_xor(sg, m, o));
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int cc = lane + SG * k;
                    e[k] = (cc < n_here && srow[cc] >= 0) ? sycl::exp(a[k] - m) : 0.0f;
                    spf[cc * GP + h] = e[k];
                }
                // warp_sum's tree: its lane l held e[l] + e[l + 32]; the xor-16 pair is this lane's two
                float l = (e[0] + e[2]) + (e[1] + e[3]);
                for (int o = SG / 2; o > 0; o >>= 1) l += sycl::permute_group_by_xor(sg, l, o);
                if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
            }
            sycl::group_barrier(grp);
            // ---- values: sub-group w owns dims [64 w, 64 w + 64), this lane d0 .. d0 + 3, every cell in order
            const int d0 = w * KV_Q8_GROUP + lane * VD;
            float acc[G][VD];
#pragma unroll
            for (int h = 0; h < G; ++h)
#pragma unroll
                for (int k = 0; k < VD; ++k) acc[h][k] = 0.0f;
            for (int cb = 0; cb < CHUNK; cb += SG) {
                const long long rl = srow[cb + lane];
                const long long rlc = rl >= 0 ? rl : 0;
                const float svl = lanes_v_scale<KV_MODE>(p, rlc, d0);
                float pl[GP];
#pragma unroll
                for (int g4 = 0; g4 < GP / 4; ++g4) {
                    const sycl::float4 v = sp[(size_t) (cb + lane) * (GP / 4) + g4];
                    pl[4 * g4] = v.x(); pl[4 * g4 + 1] = v.y(); pl[4 * g4 + 2] = v.z(); pl[4 * g4 + 3] = v.w();
                }
#pragma unroll
                for (int i0 = 0; i0 < SG; i0 += kLanesVPrefetch) {
                    LanesRaw<KV_MODE> raw[kLanesVPrefetch];   // these cells' loads issued together
#pragma unroll
                    for (int i = 0; i < kLanesVPrefetch; ++i)
                        raw[i] = lanes_v4_raw<KV_MODE>(p, sycl::select_from_group(sg, rlc, i0 + i), d0);
#pragma unroll
                    for (int i = 0; i < kLanesVPrefetch; ++i) {
                        if (sycl::select_from_group(sg, rl, i0 + i) < 0) continue;   // masked or past n_here: weight 0
                        const float svi = sycl::select_from_group(sg, svl, i0 + i);
                        float v[VD];
#pragma unroll
                        for (int k = 0; k < VD; ++k) v[k] = lanes_v1<KV_MODE>(raw[i], k, svi);
#pragma unroll
                        for (int h = 0; h < G; ++h) {
                            const float wt = sycl::select_from_group(sg, pl[h], i0 + i);
#pragma unroll
                            for (int k = 0; k < VD; ++k) acc[h][k] = sycl::fma(wt, v[k], acc[h][k]);
                        }
                    }
                }
            }
#pragma unroll
            for (int h = 0; h < G; ++h)
                *reinterpret_cast<sycl::float4*>(part_acc + ((size_t) slot * G + h) * HD + d0) =
                    sycl::float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
        }));
    });
}

// STRATA_QSA_ATTN_LANES (default off); qsa_decode_attn_set_lanes overrides it for tests (-1: the setting again).
constexpr bool kQsaAttnLanesDefault = false;
int g_lanes_forced = -1;
bool lanes_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_QSA_ATTN_LANES");
        return v == nullptr ? kQsaAttnLanesDefault : std::strtol(v, nullptr, 10) != 0;
    }();
    return g_lanes_forced >= 0 ? g_lanes_forced != 0 : env;
}

// ---- STRATA_QSA_ATTN_FAST=1 (homelab): the lanes form without launch_chunk's bits.  (1) A score is one fma chain a
// scale group (INT8: the codes times the query, then the group's sum times its scale once; launch_chunk scales every
// code and sums in reduce12's tree, so 2 ops a dimension become 1).  (2) A work-group walks `span` chunks of CHUNK
// cells in order with an online softmax (a running max and sum a head, the accumulators rescaled when the max moves),
// so the partials (12 x 256 floats a chunk, written and read back by the merge) shrink span times: at 16 rows of the
// full selection they are ~13 MB each way against 35 MB of K/V.  `span` keeps the grid at about kFastTargetGroups
// work-groups (STRATA_QSA_ATTN_FAST_GROUPS).  The merge is launch_merge over the work-groups.  Matches the default to
// float rounding (qsa_attn_lanes_parity: 1e-5 of the output's scale, and FP64).
constexpr bool kQsaAttnFastDefault = false;
constexpr int kFastTargetGroups = 512;   // one wave of work-groups (the B70: 32 Xe cores x 16 of 64 work-items)
constexpr int kFastSpanMax = 8;
int g_fast_forced = -1, g_fast_span_forced = 0;
bool fast_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_QSA_ATTN_FAST");
        return v == nullptr ? kQsaAttnFastDefault : std::strtol(v, nullptr, 10) != 0;
    }();
    return g_fast_forced >= 0 ? g_fast_forced != 0 : env;
}
int fast_span(size_t n_q, int n_kv_heads, int n_chunks) {
    static const long target = [] {
        const char* v = std::getenv("STRATA_QSA_ATTN_FAST_GROUPS");
        const long t = v == nullptr ? 0 : std::strtol(v, nullptr, 10);
        return t > 0 ? t : (long) kFastTargetGroups;
    }();
    if (g_fast_span_forced > 0) return std::min(g_fast_span_forced, n_chunks);
    // the smallest span whose grid fits `target` work-groups (one wave of the card: a grid a little past it runs a
    // second, nearly empty wave; B70 16 rows x 33 chunks, slots: span 1 226.5 us, 2 234.7, 3 181.6, 4 201.4, 6 262.3)
    const int top = std::min(kFastSpanMax, n_chunks);
    int span = 1;
    while (span < top && (long) n_q * n_kv_heads * ((n_chunks + span - 1) / span) > target) ++span;
    return span;
}

// STRATA_QSA_ATTN_LANES_MIN_ROWS: the lanes and fast forms only for calls of at least this many rows (B70, all3
// image, batch_window_ubench stage totals 1/2/4/8/16 rows: lanes +0.69/+0.74/-0.05/-0.19/-0.20 ms, fast +0.48/+0.43/
// -0.01/-0.01/-0.21: a cell a lane loses at a few work-groups, where each lane's row arrives in serial round trips).
// A forced setting (the test hooks) takes every call.
constexpr int kLanesMinRows = 4;
bool lanes_rows_ok(size_t n_q) {
    static const long min_rows = [] {
        const char* v = std::getenv("STRATA_QSA_ATTN_LANES_MIN_ROWS");
        return v == nullptr ? (long) kLanesMinRows : std::strtol(v, nullptr, 10);
    }();
    return g_lanes_forced >= 0 || g_fast_forced >= 0 || (long) n_q >= min_rows;
}

// K row `row`, dims [d0, d0+8) as floats before the scale (INT8: the codes; FP16: the values)
template <int KV_MODE>
inline void fast_k8(const QsaAttnPools& p, long long row, int d0, float (&out)[8]) {
    if constexpr (KV_MODE == 0) {
        const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(p.k_pool + row * 256 + d0);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = h2f(v[j]);
    } else {
        const sycl::vec<int8_t, 8> v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(p.k_q + row * 256 + d0);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) v[j];
    }
}

template <int HD, int G, int KV_MODE, class Src>
sycl::event launch_chunk_fast(sycl::queue& queue, const float* q_all, const Src src, const int32_t* ids_all,
                              const int32_t* step_all, int n_kv_heads, int page_size, float scale, float* part_acc_all,
                              float* part_m_all, float* part_l_all, int n_groups, int span, size_t n_q, int cap,
                              long long scratch_stride) {
    static_assert(HD == 256 && (KV_MODE == 0 || KV_MODE == 1), "the fast form: head_dim 256, FP16 or INT8 K/V");
    constexpr int SG = kLanesSg;
    constexpr int NSG = CHUNK / SG;
    constexpr int THREADS = CHUNK;
    constexpr int VD = HD / (NSG * SG);
    constexpr int GP = (G + 3) / 4 * 4;
    constexpr int HJ = (G + NSG - 1) / NSG;   // heads a sub-group keeps the running max and sum of
    constexpr int HG = G / 2;
    static_assert(VD == 4 && HD / NSG == KV_Q8_GROUP && G % 2 == 0, "the lanes form's geometry");
    const bool dense = ids_all == nullptr;
    return queue.submit([&](sycl::handler& hd) {
        sycl::local_accessor<sycl::float4, 1> sp(sycl::range<1>(CHUNK * GP / 4), hd);
        sycl::local_accessor<long long, 1> srow(sycl::range<1>(CHUNK), hd);
        sycl::local_accessor<float, 1> salpha(sycl::range<1>(GP), hd);   // this chunk's rescale of each head
        hd.parallel_for(sycl::nd_range<3>({n_q, (size_t) n_kv_heads, (size_t) n_groups * THREADS}, {1, 1, THREADS}),
                        xe::fits("qsa_decode_attn fast", [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(kLanesSg)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const size_t z = it.get_group(0);
            const QsaAttnPools p = pools_of<KV_MODE>(src, z);
            const float* q = q_all + z * (size_t) (n_kv_heads * G) * HD;
            const int32_t* ids = dense ? nullptr : ids_all + z * (size_t) cap;
            const int32_t* step = step_all + z * kStepCount;
            float* part_acc = part_acc_all + z * (size_t) scratch_stride;
            float* part_m = part_m_all + z * (size_t) scratch_stride;
            float* part_l = part_l_all + z * (size_t) scratch_stride;
            const int n_ids = dense ? step[kStepNKv] : step[kStepWidth];
            const int gi = (int) it.get_group(2), kvh = (int) it.get_group(1);
            const int t = (int) it.get_local_id(2);
            const int lane = (int) sg.get_local_linear_id(), w = (int) sg.get_group_linear_id();
            const int slot = kvh * n_groups + gi;
            const int n_blk = sycl::min(span, (n_ids + CHUNK - 1) / CHUNK - gi * span);
            if (n_blk <= 0) {                                     // uniform across the group
                if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
                return;
            }
            float* spf = reinterpret_cast<float*>(&sp[0]);
            const float* qk = q + (size_t) (kvh * G) * HD;
            float mrun[HJ], lrun[HJ];
#pragma unroll
            for (int j = 0; j < HJ; ++j) { mrun[j] = -FLT_MAX; lrun[j] = 0.0f; }
            const int d0 = w * KV_Q8_GROUP + lane * VD;
            float acc[G][VD];
#pragma unroll
            for (int h = 0; h < G; ++h)
#pragma unroll
                for (int k = 0; k < VD; ++k) acc[h][k] = 0.0f;
            for (int blk = 0; blk < n_blk; ++blk) {
                const int c0 = (gi * span + blk) * CHUNK;
                const int n_here = sycl::min(CHUNK, n_ids - c0);
                sycl::group_barrier(grp);                         // the last chunk's value pass is done with sp
                // ---- scores: lane = cell c of the chunk
                const int c = w * SG + lane;
                long long r = -1;
                if (c < n_here) {
                    const int cell = dense ? c0 + c : ids[c0 + c];
                    const long long page = (long long) p.page_table[cell / page_size];
                    if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
                }
                srow[c] = r;
                const long long rr = r >= 0 ? r : 0;
#pragma unroll 1
                for (int h0 = 0; h0 < G; h0 += HG) {
                    float s[HG];
#pragma unroll
                    for (int h = 0; h < HG; ++h) s[h] = 0.0f;
#pragma unroll 1
                    for (int g = 0; g < HD / KV_Q8_GROUP; ++g) {
                        float part[HG];
#pragma unroll
                        for (int h = 0; h < HG; ++h) part[h] = 0.0f;
#pragma unroll
                        for (int j = 0; j < KV_Q8_GROUP / 8; ++j) {
                            const int b = g * (KV_Q8_GROUP / 8) + j;
                            float kk[8];
                            fast_k8<KV_MODE>(p, rr, b * 8, kk);
#pragma unroll
                            for (int h = 0; h < HG; ++h) {
                                const float qw = qk[(h0 + h) * HD + b * 8 + (lane & 7)];
#pragma unroll
                                for (int e = 0; e < 8; ++e)
                                    part[h] = sycl::fma(kk[e], sycl::select_from_group(sg, qw, e), part[h]);
                            }
                        }
                        if constexpr (KV_MODE == 1) {
                            const float ksg = h2f(p.k_scale[rr * (HD / KV_Q8_GROUP) + g]);
#pragma unroll
                            for (int h = 0; h < HG; ++h) s[h] = sycl::fma(part[h], ksg, s[h]);
                        } else {
#pragma unroll
                            for (int h = 0; h < HG; ++h) s[h] += part[h];
                        }
                    }
#pragma unroll
                    for (int h = 0; h < HG; ++h) spf[c * GP + h0 + h] = r >= 0 ? s[h] * scale : -FLT_MAX;
                }
                sycl::group_barrier(grp);
                // ---- online softmax: sub-group w keeps heads w, w + 4, ...
#pragma unroll
                for (int j = 0; j < HJ; ++j) {
                    const int h = w + NSG * j;
                    if (h >= G) continue;
                    float a[4], e[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) a[k] = spf[(lane + SG * k) * GP + h];
                    float m = sycl::fmax(sycl::fmax(a[0], a[1]), sycl::fmax(a[2], a[3]));
                    m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                    const float mn = sycl::fmax(mrun[j], m);
                    float l = 0.0f;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int cc = lane + SG * k;
                        e[k] = (cc < n_here && srow[cc] >= 0) ? sycl::exp(a[k] - mn) : 0.0f;
                        spf[cc * GP + h] = e[k];
                        l += e[k];
                    }
                    l = sycl::reduce_over_group(sg, l, sycl::plus<float>());
                    const float alpha = mrun[j] == -FLT_MAX ? 0.0f : sycl::exp(mrun[j] - mn);
                    lrun[j] = sycl::fma(lrun[j], alpha, l);
                    mrun[j] = mn;
                    if (lane == 0) salpha[h] = alpha;
                }
                sycl::group_barrier(grp);
                // ---- values: rescale, then fold this chunk's cells in order
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float al = salpha[h];
#pragma unroll
                    for (int k = 0; k < VD; ++k) acc[h][k] *= al;
                }
                for (int cb = 0; cb < CHUNK; cb += SG) {
                    const long long rl = srow[cb + lane];
                    const long long rlc = rl >= 0 ? rl : 0;
                    const float svl = lanes_v_scale<KV_MODE>(p, rlc, d0);
                    float pl[GP];
#pragma unroll
                    for (int g4 = 0; g4 < GP / 4; ++g4) {
                        const sycl::float4 v = sp[(size_t) (cb + lane) * (GP / 4) + g4];
                        pl[4 * g4] = v.x(); pl[4 * g4 + 1] = v.y(); pl[4 * g4 + 2] = v.z(); pl[4 * g4 + 3] = v.w();
                    }
#pragma unroll
                    for (int i0 = 0; i0 < SG; i0 += kLanesVPrefetch) {
                        LanesRaw<KV_MODE> raw[kLanesVPrefetch];
#pragma unroll
                        for (int i = 0; i < kLanesVPrefetch; ++i)
                            raw[i] = lanes_v4_raw<KV_MODE>(p, sycl::select_from_group(sg, rlc, i0 + i), d0);
#pragma unroll
                        for (int i = 0; i < kLanesVPrefetch; ++i) {
                            if (sycl::select_from_group(sg, rl, i0 + i) < 0) continue;
                            const float svi = sycl::select_from_group(sg, svl, i0 + i);
                            float v[VD];
#pragma unroll
                            for (int k = 0; k < VD; ++k) v[k] = lanes_v1<KV_MODE>(raw[i], k, svi);
#pragma unroll
                            for (int h = 0; h < G; ++h) {
                                const float wt = sycl::select_from_group(sg, pl[h], i0 + i);
#pragma unroll
                                for (int k = 0; k < VD; ++k) acc[h][k] = sycl::fma(wt, v[k], acc[h][k]);
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < HJ; ++j) {
                const int h = w + NSG * j;
                if (h < G && lane == 0) { part_m[slot * G + h] = mrun[j]; part_l[slot * G + h] = lrun[j]; }
            }
#pragma unroll
            for (int h = 0; h < G; ++h)
                *reinterpret_cast<sycl::float4*>(part_acc + ((size_t) slot * G + h) * HD + d0) =
                    sycl::float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
        }));
    });
}

template <int HD, int G>
sycl::event launch_merge(sycl::queue& queue, const float* part_acc_all, const float* part_m_all, const float* part_l_all,
                         int n_chunks, float* attn_all, size_t n_head, size_t n_q, long long scratch_stride) {
    return queue.parallel_for(sycl::nd_range<2>({n_q, n_head * HD}, {1, HD}), [=](sycl::nd_item<2> it) {
        const size_t y = it.get_group(0);
        const float* part_acc = part_acc_all + y * (size_t) scratch_stride;
        const float* part_m = part_m_all + y * (size_t) scratch_stride;
        const float* part_l = part_l_all + y * (size_t) scratch_stride;
        float* attn = attn_all + y * n_head * HD;
        const int h = (int) it.get_group(1);                 // global query head
        const int kvh = h / G, hl = h % G;
        const int d = (int) it.get_local_id(1);
        float M = -FLT_MAX;
        for (int c = 0; c < n_chunks; ++c) M = sycl::fmax(M, part_m[(kvh * n_chunks + c) * G + hl]);
        float L = 0.0f, acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const int slot = kvh * n_chunks + c;
            const float m = part_m[slot * G + hl];
            if (m == -FLT_MAX) continue;
            const float w = sycl::exp(m - M);
            L = sycl::fma(part_l[slot * G + hl], w, L);
            acc = sycl::fma(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
        }
        attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
    });
}

int kv_mode_of(const QsaAttnPools& pools) {
    return pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3 : (pools.k_q != nullptr ? 1 : 0));
}

int kv_mode_of(const QsaAttnSlots& tab) { return tab.mode; }

template <int HD, int G, class Src>
void run_shape(const float* q, const Src& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
               const QsaShapes& s, float* scratch, float* attn, size_t n_q, long long stride, void* stream) {
    const int kv_mode = kv_mode_of(pools);
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / std::sqrt((float) HD);
    auto& queue = queue_for(stream);
    const int nkv = (int) s.n_head_kv, ps = (int) s.page_size;
    if constexpr (HD == 256) {
        if (fast_on() && kv_mode <= 1 && lanes_rows_ok(n_q)) {
            const int span = fast_span(n_q, nkv, n_chunks), n_groups = (n_chunks + span - 1) / span;
            if (kv_mode == 1) launch_chunk_fast<HD, G, 1>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_groups, span, n_q, (int) cap, stride);
            else launch_chunk_fast<HD, G, 0>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_groups, span, n_q, (int) cap, stride);
            const auto e = launch_merge<HD, G>(queue, part_acc, part_m, part_l, n_groups, attn, (size_t) s.n_head, n_q, stride);
            if (!stream) core::Runtime::get().wait(e, "qsa_decode_attn");
            return;
        }
        if (lanes_on() && kv_mode <= 1 && lanes_rows_ok(n_q)) {
            if (kv_mode == 1) launch_chunk_lanes<HD, G, 1>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
            else launch_chunk_lanes<HD, G, 0>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
            const auto e = launch_merge<HD, G>(queue, part_acc, part_m, part_l, n_chunks, attn, (size_t) s.n_head, n_q, stride);
            if (!stream) core::Runtime::get().wait(e, "qsa_decode_attn");
            return;
        }
    }
    if (kv_mode == 3) launch_chunk<HD, G, 3>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else if (kv_mode == 2) launch_chunk<HD, G, 2>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else if (kv_mode == 1) launch_chunk<HD, G, 1>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    else launch_chunk<HD, G, 0>(queue, q, pools, ids, steps, nkv, ps, scale, part_acc, part_m, part_l, n_chunks, n_q, (int) cap, stride);
    const auto e = launch_merge<HD, G>(queue, part_acc, part_m, part_l, n_chunks, attn, (size_t) s.n_head, n_q, stride);
    if (!stream) core::Runtime::get().wait(e, "qsa_decode_attn");
}

// The shapes this file is built for: (head_dim, query heads per KV head).  A model with another shape needs its line
// here; the kernels' local arrays are sized by G and HD.
bool shape_ok(const QsaShapes& s) {
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv) return false;
    const int64_t g = s.n_head / s.n_head_kv;
    return (s.head_dim == 256 && (g == 12 || g == 8)) || (s.head_dim == 128 && g == 10);
}

template <class Src>
void run(const float* q, const Src& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
         const QsaShapes& s, float* scratch, float* attn, size_t n_q, long long stride, void* stream) {
    const int64_t g = s.n_head / s.n_head_kv;
    if (s.head_dim == 256 && g == 12) run_shape<256, 12>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream);
    else if (s.head_dim == 256 && g == 8) run_shape<256, 8>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream);
    else run_shape<128, 10>(q, pools, ids, steps, cap, s, scratch, attn, n_q, stride, stream);
}

bool pools_ok(const QsaAttnPools& pools) {
    const int kv_mode = kv_mode_of(pools);
    return kv_mode == 3 ? pools.k_scale != nullptr
         : kv_mode == 2 ? pools.v_q4 != nullptr
         : kv_mode == 1 ? (pools.v_q && pools.k_scale && pools.v_scale) : (pools.k_pool && pools.v_pool);
}

std::string shape_name(const QsaShapes& s) {
    return std::to_string(s.n_head) + "/" + std::to_string(s.n_head_kv) + " heads of " + std::to_string(s.head_dim);
}

}  // namespace

void qsa_decode_attn_set_lanes(int on) { g_lanes_forced = on < 0 ? -1 : (on != 0); }

void qsa_decode_attn_set_fast(int on, int span) {
    g_fast_forced = on < 0 ? -1 : (on != 0);
    g_fast_span_forced = span > 0 ? span : 0;
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (!shape_ok(s) || cap <= 0 || !scratch || !ids || !steps || !pools.page_table || n_q > 65535)
        throw core::DeviceError("qsa_decode_attn_batch: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    run(q, pools, ids, steps, cap, s, scratch, attn, (size_t) n_q, (long long) qsa_decode_attn_scratch_floats(cap, s), stream);
}

bool qsa_attn_slots_set(QsaAttnSlots& tab, int r, const QsaAttnPools& pools) {
    const int mode = kv_mode_of(pools);
    if (r < 0 || r >= kQsaSlotsMax || !pools_ok(pools) || !pools.page_table || (tab.mode >= 0 && tab.mode != mode))
        return false;
    tab.mode = mode;
    tab.page_table[r] = pools.page_table;
    tab.k[r] = mode == 0 ? (const void*) pools.k_pool : mode == 2 ? (const void*) pools.k_q4 : (const void*) pools.k_q;
    tab.v[r] = mode == 0 ? (const void*) pools.v_pool : mode == 1 ? (const void*) pools.v_q : (const void*) pools.v_q4;
    tab.k_scale[r] = mode == 1 || mode == 3 ? pools.k_scale : nullptr;
    tab.v_scale[r] = mode == 1 ? pools.v_scale : nullptr;
    return true;
}

void qsa_decode_attn_slots(const float* q, const QsaAttnSlots& tab, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (!shape_ok(s) || cap <= 0 || !scratch || !ids || !steps || n_q > kQsaSlotsMax || tab.mode < 0 || tab.mode > 3)
        throw core::DeviceError("qsa_decode_attn_slots: unsupported geometry (" + shape_name(s) + "), rows or format");
    for (int64_t r = 0; r < n_q; ++r)
        if (!tab.page_table[r] || !tab.k[r] || !tab.v[r])
            throw core::DeviceError("qsa_decode_attn_slots: a row's pools are missing");
    // qsa_decode_attn_batch's strides: query r's scratch at r * the single-query size
    run(q, tab, ids, steps, cap, s, scratch, attn, (size_t) n_q, (long long) qsa_decode_attn_scratch_floats(cap, s), stream);
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (uint64_t) (s.head_dim + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (!shape_ok(s) || cap <= 0 || !scratch || !ids || !step || !pools.page_table)
        throw core::DeviceError("qsa_decode_attn: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_decode_attn: incomplete KV pools");
    run(q, pools, ids, step, cap, s, scratch, attn, 1, 0, stream);
}

void qsa_dense_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* step, int64_t max_cells,
                         const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (!shape_ok(s) || max_cells <= 0 || !scratch || !step || !pools.page_table)
        throw core::DeviceError("qsa_dense_attn: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_dense_attn: incomplete KV pools");
    run(q, pools, nullptr, step, max_cells, s, scratch, attn, 1, 0, stream);
}

void qsa_dense_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* steps, int64_t max_cells,
                          const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (!shape_ok(s) || max_cells <= 0 || !scratch || !steps || !pools.page_table || n_q > 65535)
        throw core::DeviceError("qsa_dense_attn_batch: unsupported geometry (" + shape_name(s) + ") or missing buffers");
    if (!pools_ok(pools)) throw core::DeviceError("qsa_dense_attn_batch: incomplete KV pools");
    run(q, pools, nullptr, steps, max_cells, s, scratch, attn, (size_t) n_q,
        (long long) qsa_decode_attn_scratch_floats(max_cells, s), stream);
}

}  // namespace strata::kernels
