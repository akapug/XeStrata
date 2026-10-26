// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/hc_xmx.cpp - see include/strata/kernels/hc_xmx.hpp.
//
// A product step is a sub-group of 16 outputs (a lane its weight row, Intel's accumulator column) against every row of
// the window, a block of 32 inputs at a time: the lane loads its row's 32 int8 weights and their fp16 scale, and
//   * the XMX body hands the block to the matrix engines through 512 bytes of the sub-group's local memory (Intel's
//     VNNI B layout, as iq_mmq_decode) and takes 8 x 16 x 32 int8 tiles of the activations from local memory;
//   * the portable body (the OpenCL CPU device, the tests) adds the same int8 products with dp4a.
// Either gives the block's exact int32 sum c (two slices: 128 c1 + c2), and both then run the same
// acc = fma(w_scale * a_scale, float(c), acc) over the blocks in order, so they agree bit for bit.  An 8-row accumulator
// gives each lane its element i in row i (Xe's DPAS layout; iq_mmq_decode_ok's probe checks it).
#include "strata/kernels/hc_xmx.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#include "cuda_intrinsics.hpp"
#include "q8_1_finite.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_hc_xmx
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>

// Intel's joint_matrix bodies only in Intel's device compile (and the host pass); the run-time probe is the only
// thing that launches them
#if STRATA_DEVICE_NOT_INTEL
#define STRATA_HCX_INTEL 0
#else
#define STRATA_HCX_INTEL 1
#endif

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;

constexpr int N = kHcxN, HC = kHcxHC, D = kHcxD, LR = kHcxLR, QB = kHcxBlock, PR = kHcxPR;
constexpr int SG = 16;                                    // a sub-group: an accumulator's 16 columns, a lane each
constexpr int TM = 8, TK = QB;                            // an XMX tile: 8 rows, K 32
constexpr int MTP = (kHcxMaxRows + TM - 1) / TM * TM;     // rows in whole tiles
constexpr int MTILES = MTP / TM;
constexpr int KCH = kHcxKch, NCH = kHcxChunks, NBC = KCH / QB, CPS = N / KCH;
constexpr int NSGD = LR / SG, WGD = NSGD * SG;            // down: a sub-group 16 outputs
constexpr int NBL = LR / QB, WGL = (NBL + 1) * SG;        // lo: a sub-group a block, one more for rs and the inject
constexpr int UPC = kHcxUpCols, UPH = UPC / SG, NSGU = HC * UPH, WGU = NSGU * SG;   // up: a sub-group 16 columns
constexpr int LDA = KCH + 16, LDU = LR + 16;              // local int8 row strides (16-byte multiples)
constexpr int UNR = 5;                                    // weight blocks a lane loads ahead
constexpr int BT = TK * SG;                               // a sub-group's B tile in bytes
constexpr size_t ROW_D = (size_t) D / QB * (QB + 2), ROW_U = (size_t) LR / QB * (QB + 2);
static_assert(D % KCH == 0 && N % KCH == 0 && KCH % QB == 0 && NBC % UNR == 0 && NBL % UNR == 0, "chunks");
static_assert(LR % SG == 0 && LR % QB == 0 && N % UPC == 0 && UPC % SG == 0 && QB == 2 * SG, "a block, two a lane");
static_assert(32 * 127 * 127 * 128 + 32 * 127 * 127 < (1u << 31), "128 c1 + c2 fits an int32");

struct Win {
    float* R;
    const float* bo;
    const float* ip;
    const float* wn;
    const uint16_t* wi;
    const uint8_t* qd;
    const uint8_t* qu;
    float* lo;
    float* rs;
    float* io;
    float* mixed;
    uint8_t* q8m;
    float eps;
    int apply;
    int T;
};

inline float sg_sum(const sycl::sub_group& sg, float v) {
    for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float sg_max(const sycl::sub_group& sg, float v) {
    for (int o = SG / 2; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float bf16f(uint16_t v) { return sycl::bit_cast<float>((uint32_t) v << 16); }
inline float halff(uint16_t v) { return (float) sycl::bit_cast<sycl::half>(v); }

template <typename T>
inline auto local_as(sycl::multi_ptr<uint8_t, sycl::access::address_space::local_space, sycl::access::decorated::no> p) {
    using V = sycl::multi_ptr<void, sycl::access::address_space::local_space, sycl::access::decorated::no>;
    return static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space, sycl::access::decorated::no>>(V(p));
}

// A block of 32 values, two a lane (v0 at 2 lane, v1 at 2 lane + 1), as int8: slice one round(v / d) with d = amax /
// 127, slice two (S = 2) the residual at d / 128; `scale` is what the block's int32 sum (c1, or 128 c1 + c2) is
// multiplied by.  hcx_parity's host quantizer is this arithmetic.
template <int S>
inline void quant_pair(const sycl::sub_group& sg, float v0, float v1, int8_t* q1, int8_t* q2, float& scale) {
    const float amax = sg_max(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)));
    const float d = amax / 127.0f, id = amax > 0.0f ? 1.0f / d : 0.0f;
    const float a0 = sycl::round(v0 * id), a1 = sycl::round(v1 * id);
    q1[0] = (int8_t) (int) a0;
    q1[1] = (int8_t) (int) a1;
    scale = d;
    if constexpr (S == 2) {
        const float id2 = id * 128.0f;
        const float b0 = sycl::round(sycl::fma(-a0, d, v0) * id2), b1 = sycl::round(sycl::fma(-a1, d, v1) * id2);
        q2[0] = (int8_t) (int) sycl::clamp(b0, -127.0f, 127.0f);
        q2[1] = (int8_t) (int) sycl::clamp(b1, -127.0f, 127.0f);
        scale = d * (1.0f / 128.0f);
    }
}

// One block (`kb` of the activations in local memory, `lda` their row stride, `as` their scales [row][nb]) times the
// lane's weights `wv` (8 words) with scale `ws`, into acc[tile][i] (row tile * TM + i), rows below T.  `bt`: the
// sub-group's B tile (XMX).
template <bool X, int S>
inline void block_dot([[maybe_unused]] const sycl::sub_group& sg, [[maybe_unused]] int lane, const uint32_t (&wv)[8],
                      float ws, int kb, int T,
                      sycl::multi_ptr<int8_t, sycl::access::address_space::local_space, sycl::access::decorated::no> a,
                      int lda, const float* as, int nb,
                      [[maybe_unused]] sycl::multi_ptr<int8_t, sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no> bt,
                      float (&acc)[MTILES][TM]) {
    if constexpr (X) {
#if STRATA_HCX_INTEL
        uint32_t* bw = reinterpret_cast<uint32_t*>(bt.get());
#pragma unroll
        for (int j = 0; j < 8; ++j) bw[j * SG + lane] = wv[j];
        sycl::group_barrier(sg);
        mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, TK, SG, mx::layout::ext_intel_packed> bm;
        mx::joint_matrix_load(sg, bm, bt, SG * 4);
#pragma unroll
        for (int t = 0; t < MTILES; ++t) {
            if (t * TM >= T) break;
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, TM, SG> ci;
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, TM, TK, mx::layout::row_major> am;
            mx::joint_matrix_fill(sg, ci, 0);
            mx::joint_matrix_load(sg, am, a + (t * TM * lda + kb * QB), lda);
            mx::joint_matrix_mad(sg, ci, am, bm, ci);
            if constexpr (S == 2) {
                mx::joint_matrix_apply(sg, ci, [](int32_t& c) { c *= 128; });
                mx::joint_matrix_load(sg, am, a + ((MTP + t * TM) * lda + kb * QB), lda);
                mx::joint_matrix_mad(sg, ci, am, bm, ci);
            }
            int i = 0;
            mx::joint_matrix_apply(sg, ci, [&](int32_t& c) {
                acc[t][i] = sycl::fma(ws * as[(t * TM + i) * nb + kb], (float) c, acc[t][i]);
                ++i;
            });
        }
        sycl::group_barrier(sg);   // the tile is read before the next block overwrites it
#endif
    } else {
#pragma unroll
        for (int t = 0; t < MTILES; ++t)
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = t * TM + i;
                if (row >= T) continue;
                const int* p1 = reinterpret_cast<const int*>((a + (row * lda + kb * QB)).get());
                int c = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) c = xe::dp4a(p1[j], (int) wv[j], c);
                if constexpr (S == 2) {
                    const int* p2 = reinterpret_cast<const int*>((a + ((MTP + row) * lda + kb * QB)).get());
                    int c2 = 0;
#pragma unroll
                    for (int j = 0; j < 8; ++j) c2 = xe::dp4a(p2[j], (int) wv[j], c2);
                    c = c * 128 + c2;
                }
                acc[t][i] = sycl::fma(ws * as[row * nb + kb], (float) c, acc[t][i]);
            }
    }
}

// The lane's weight row `wr` (int8 values of `cols`, then their fp16 scales) over blocks kb0 .. kb0 + nblk of it,
// against activation blocks 0 .. nblk in local memory
template <bool X, int S, int NBLK>
inline void row_dots(const sycl::sub_group& sg, int lane, const uint8_t* wr, int cols, int kb0, int T,
                     sycl::multi_ptr<int8_t, sycl::access::address_space::local_space, sycl::access::decorated::no> a,
                     int lda, const float* as,
                     sycl::multi_ptr<int8_t, sycl::access::address_space::local_space, sycl::access::decorated::no> bt,
                     float (&acc)[MTILES][TM]) {
    const uint16_t* wsc = reinterpret_cast<const uint16_t*>(wr + cols);
    for (int b0 = 0; b0 < NBLK; b0 += UNR) {
        uint32_t wv[UNR][8];
        float ws[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(wr + (size_t) (kb0 + b0 + u) * QB);
#pragma unroll
            for (int j = 0; j < 8; ++j) wv[u][j] = p[j];
            ws[u] = halff(wsc[kb0 + b0 + u]);
        }
#pragma unroll
        for (int u = 0; u < UNR; ++u) block_dot<X, S>(sg, lane, wv[u], ws[u], b0 + u, T, a, lda, as, NBLK, bt, acc);
    }
}

// the XMX bodies (never launched off Intel's GPUs) in the large register file: a lane holds 32 accumulators and UNR
// blocks of weights; the portable ones in the default file (the OpenCL CPU device refuses the large one's option)
template <bool X>
constexpr auto kernel_props() {
    if constexpr (X)
        return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>,
                                  sycl::ext::intel::experimental::grf_size<256>};
    else
        return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
}

// down: a work-group a chunk of KCH inputs (stream c = chunk / CPS), every output
template <bool X, int S>
struct Down {
    static constexpr int A_BYTES = S * MTP * LDA, YS_OFF = A_BYTES, SS_OFF = YS_OFF + MTP * NBC * 4,
                         IJ_OFF = SS_OFF + MTP * NBC * 4, BT_OFF = IJ_OFF + MTP * NBC * HC * 4,
                         LOCAL = BT_OFF + (X ? NSGD * BT : 0);
    static_assert(A_BYTES % 16 == 0 && BT_OFF % 16 == 0, "local alignment");
    Win w;
    float* part;
    float* ssg;
    sycl::local_accessor<uint8_t, 1> lm;

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), sgid = (int) sg.get_group_linear_id();
        const int lid = (int) it.get_local_id(0), ch = (int) it.get_group(0), c = ch / CPS, col0 = ch * KCH;
        const int T = w.T, TP = (T + TM - 1) / TM * TM;
        const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
        const auto a = local_as<int8_t>(base);
        int8_t* al = a.get();
        float* ys = local_as<float>(base + YS_OFF).get();
        float* bss = local_as<float>(base + SS_OFF).get();
        float* bij = local_as<float>(base + IJ_OFF).get();
        // stage: (row, block) a sub-group, two inputs a lane: R' (written back), y = R' w_norm, the block's sum of
        // squares and inject partials, y quantized; rows T .. TP zeros
        for (int j = sgid; j < TP * NBC; j += NSGD) {
            const int t = j / NBC, b = j - t * NBC, k = col0 + b * QB + 2 * lane;
            float y0 = 0.0f, y1 = 0.0f, ss = 0.0f, p[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (t < T) {
                float* r = w.R + (size_t) t * D + k;
                float r0 = r[0], r1 = r[1];
                if (w.apply) {
                    const float gw = 2.0f * sigmoidf_(w.ip[t * HC + c] / (float) HC);
                    const float* bo = w.bo + (size_t) t * N + (k - c * N);
                    r0 = sycl::fma(bo[0], gw, r0);
                    r1 = sycl::fma(bo[1], gw, r1);
                    r[0] = r0;
                    r[1] = r1;
                }
                ss = r0 * r0 + r1 * r1;
                y0 = r0 * w.wn[k];
                y1 = r1 * w.wn[k + 1];
                if (w.wi != nullptr) {
#pragma unroll
                    for (int q = 0; q < HC; ++q)
                        p[q] = sycl::fma(y1, bf16f(w.wi[(size_t) q * D + k + 1]), y0 * bf16f(w.wi[(size_t) q * D + k]));
                }
            }
            float sc = 0.0f;
            int8_t* q1 = al + t * LDA + b * QB + 2 * lane;
            quant_pair<S>(sg, y0, y1, q1, q1 + MTP * LDA, sc);
            ss = sg_sum(sg, ss);
#pragma unroll
            for (int q = 0; q < HC; ++q) p[q] = sg_sum(sg, p[q]);
            if (lane == 0) {
                ys[t * NBC + b] = sc;
                bss[t * NBC + b] = ss;
#pragma unroll
                for (int q = 0; q < HC; ++q) bij[(t * NBC + b) * HC + q] = p[q];
            }
        }
        sycl::group_barrier(it.get_group());
        // the chunk's sums of squares and inject partials, the blocks in order
        for (int i = lid; i < T * (1 + HC); i += WGD) {
            const int t = i / (1 + HC), q = i - t * (1 + HC);
            float s = 0.0f;
            for (int b = 0; b < NBC; ++b) s += q == 0 ? bss[t * NBC + b] : bij[(t * NBC + b) * HC + q - 1];
            if (q == 0) ssg[t * NCH + ch] = s;
            else if (w.wi != nullptr) part[((size_t) t * NCH + ch) * PR + LR + q - 1] = s;
        }
        // the products: sub-group sgid outputs 16 sgid .., the chunk's blocks
        const int r = sgid * SG + lane;
        float acc[MTILES][TM];
#pragma unroll
        for (int t = 0; t < MTILES; ++t)
#pragma unroll
            for (int i = 0; i < TM; ++i) acc[t][i] = 0.0f;
        row_dots<X, S, NBC>(sg, lane, w.qd + (size_t) r * ROW_D, D, col0 / QB, T, a, LDA, ys,
                            local_as<int8_t>(base + BT_OFF + sgid * BT), acc);
#pragma unroll
        for (int t = 0; t < MTILES; ++t)
#pragma unroll
            for (int i = 0; i < TM; ++i)
                if (t * TM + i < T) part[((size_t) (t * TM + i) * NCH + ch) * PR + r] = acc[t][i];
    }
    auto get(syclex::properties_tag) const { return kernel_props<X>(); }
};

// lo: a work-group a row: rs, lo, the inject, lo quantized
template <int S>
struct Lo {
    Win w;
    const float* part;
    const float* ssg;
    int8_t* lq;
    float* lsc;

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), sgid = (int) sg.get_group_linear_id();
        const int t = (int) it.get_group(0);
        float rs[HC];
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float ss = 0.0f;
            for (int j = 0; j < CPS; ++j) ss += ssg[t * NCH + c * CPS + j];
            rs[c] = sycl::rsqrt(ss / (float) N + w.eps);
        }
        const auto sum_of = [&](int row) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float pp = 0.0f;
                for (int j = 0; j < CPS; ++j) pp += part[((size_t) t * NCH + c * CPS + j) * PR + row];
                sum = sycl::fma(rs[c], pp, sum);
            }
            return sum;
        };
        if (sgid < NBL) {
            float v[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int row = sgid * QB + 2 * lane + e;
                const float x = sum_of(row) / (float) HC;
                v[e] = x / (1.0f + sycl::exp(-x));
                w.lo[(size_t) t * LR + row] = v[e];
            }
            float sc = 0.0f;
            int8_t* q1 = lq + (size_t) t * kHcxMaxSlices * LR + sgid * QB + 2 * lane;
            quant_pair<S>(sg, v[0], v[1], q1, q1 + LR, sc);
            if (lane == 0) lsc[t * NBL + sgid] = sc;
            return;
        }
        if (lane >= HC) return;
        if (w.wi != nullptr) w.io[t * HC + lane] = sum_of(LR + lane);
        w.rs[t * HC + lane] = lane == 0 ? rs[0] : lane == 1 ? rs[1] : lane == 2 ? rs[2] : rs[3];
    }
    auto get(syclex::properties_tag) const { return kernel_props<false>(); }
};

// up: a work-group UPC columns of every stream; sub-group s stream s / UPH, columns d0 + 16 (s % UPH) ..
template <bool X, int S>
struct Up {
    // the gates and mixed reuse the activations' bytes once every sub-group's products are done
    static constexpr int A_BYTES = S * MTP * LDU, G_OFF = 0, M_OFF = G_OFF + MTP * HC * UPC * 4,
                         A_REGION = std::max(A_BYTES, M_OFF + MTP * UPC * 4), LS_OFF = A_REGION,
                         RS_OFF = LS_OFF + MTP * NBL * 4, BT_OFF = RS_OFF + MTP * HC * 4,
                         LOCAL = BT_OFF + (X ? NSGU * BT : 0);
    static_assert(A_BYTES % 16 == 0 && BT_OFF % 16 == 0, "local alignment");
    Win w;
    const int8_t* lq;
    const float* lsc;
    float* u_out;
    sycl::local_accessor<uint8_t, 1> lm;

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), sgid = (int) sg.get_group_linear_id();
        const int lid = (int) it.get_local_id(0), d0 = (int) it.get_group(0) * UPC;
        const int T = w.T, TP = (T + TM - 1) / TM * TM;
        const auto base = lm.template get_multi_ptr<sycl::access::decorated::no>();
        const auto a = local_as<int8_t>(base);
        uint32_t* aw = local_as<uint32_t>(base).get();
        float* ls = local_as<float>(base + LS_OFF).get();
        float* rss = local_as<float>(base + RS_OFF).get();
        float* g = local_as<float>(base + G_OFF).get();
        float* m = local_as<float>(base + M_OFF).get();
        constexpr int WR = LR / 4, WA = LDU / 4;   // words a row of lq, of local memory
        for (int i = lid; i < S * TP * WR; i += WGU) {
            const int sl = i / (TP * WR), rem = i - sl * TP * WR, t = rem / WR, wd = rem - t * WR;
            aw[(sl * MTP + t) * WA + wd] =
                t < T ? reinterpret_cast<const uint32_t*>(lq + ((size_t) t * kHcxMaxSlices + sl) * LR)[wd] : 0u;
        }
        for (int i = lid; i < TP * NBL; i += WGU) ls[i] = i < T * NBL ? lsc[i] : 0.0f;
        for (int i = lid; i < T * HC; i += WGU) rss[i] = w.rs[i];
        sycl::group_barrier(it.get_group());
        const int c = sgid / UPH, dd = (sgid - c * UPH) * SG + lane, oi = c * N + d0 + dd;
        float acc[MTILES][TM];
#pragma unroll
        for (int t = 0; t < MTILES; ++t)
#pragma unroll
            for (int i = 0; i < TM; ++i) acc[t][i] = 0.0f;
        row_dots<X, S, NBL>(sg, lane, w.qu + (size_t) oi * ROW_U, LR, 0, T, a, LDU, ls,
                            local_as<int8_t>(base + BT_OFF + sgid * BT), acc);
        const float wn = w.wn[oi];
        sycl::group_barrier(it.get_group());   // the activations are read; the gates take their bytes
#pragma unroll
        for (int t = 0; t < MTILES; ++t)
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = t * TM + i;
                if (row >= T) continue;
                const float u = acc[t][i];
                if (u_out != nullptr) u_out[(size_t) row * D + oi] = u;
                const float x = w.R[(size_t) row * D + oi] * wn * rss[row * HC + c];
                g[(row * HC + c) * UPC + dd] = x * sigmoidf_(u);
            }
        sycl::group_barrier(it.get_group());
        for (int i = lid; i < T * UPC; i += WGU) {
            const int t = i / UPC, col = i - t * UPC;
            float sum = 0.0f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) sum += g[(t * HC + cc) * UPC + col];
            const float xi = sum / (float) HC;
            w.mixed[(size_t) t * N + d0 + col] = xi;
            m[t * UPC + col] = xi;
        }
        if (w.q8m == nullptr) return;
        sycl::group_barrier(it.get_group());
        // mixed's q8_1 block (native_quantize_q8_1's arithmetic; its 32-lane sum's first step is lane + lane 16)
        for (int t = sgid; t < T; t += NSGU) {
            const float v0 = m[t * UPC + lane], v1 = m[t * UPC + lane + SG];
            const float amax = sg_max(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)));
            const float bsum = sg_sum(sg, v0 + v1);
            const float d = xe::q8_1_finite(amax / 127.0f);
            uint8_t* blk = w.q8m + (size_t) t * (N / 32) * 36 + (size_t) (d0 / 32) * 36;
            blk[4 + lane] = (uint8_t) xe::q8_1_quant(v0, d, amax);
            blk[4 + lane + SG] = (uint8_t) xe::q8_1_quant(v1, d, amax);
            if (lane == 0) *reinterpret_cast<sycl::half2*>(blk) = sycl::half2(sycl::half(d), sycl::half(xe::q8_1_finite(bsum)));
        }
    }
    auto get(syclex::properties_tag) const { return kernel_props<X>(); }
};
static_assert(UPC == 32, "an up work-group writes whole q8_1 blocks of mixed");

template <bool X, int S>
sycl::event launch(sycl::queue& q, const Win& w, const HcxScratch& s, unsigned long long* sb, int si) {
    q.submit([&](sycl::handler& h) {
        using K = Down<X, S>;
        h.parallel_for(sycl::nd_range<1>((size_t) NCH * WGD, WGD),
                       K{w, s.part, s.ssg, sycl::local_accessor<uint8_t, 1>(sycl::range<1>(K::LOCAL), h)});
    });
    if (sb) {   // the norm is in the down launch: both of the BF16 read's stamps after it
        gpu_stamp(sb, si, &q);
        gpu_stamp(sb, si + 1, &q);
    }
    q.parallel_for(sycl::nd_range<1>((size_t) w.T * WGL, WGL), Lo<S>{w, s.part, s.ssg, s.lq, s.lsc});
    return q.submit([&](sycl::handler& h) {
        using K = Up<X, S>;
        h.parallel_for(sycl::nd_range<1>((size_t) (N / UPC) * WGU, WGU),
                       K{w, s.lq, s.lsc, s.u, sycl::local_accessor<uint8_t, 1>(sycl::range<1>(K::LOCAL), h)});
    });
}

size_t al256(size_t b) { return (b + 255) & ~(size_t) 255; }

int env_int(const char* name, int def) {
    const char* e = std::getenv(name);
    return e != nullptr && *e != 0 ? (int) std::strtol(e, nullptr, 10) : def;
}

}  // namespace

size_t hcx_scratch_bytes(int rows) {
    const size_t r = (size_t) std::max(rows, 1);
    return al256(r * NCH * PR * 4) + al256(r * NCH * 4) + al256(r * kHcxMaxSlices * LR) + al256(r * NBL * 4);
}

HcxScratch hcx_carve(void* base, int rows) {
    const size_t r = (size_t) std::max(rows, 1);
    auto* p = static_cast<uint8_t*>(base);
    HcxScratch s;
    s.part = reinterpret_cast<float*>(p);
    p += al256(r * NCH * PR * 4);
    s.ssg = reinterpret_cast<float*>(p);
    p += al256(r * NCH * 4);
    s.lq = reinterpret_cast<int8_t*>(p);
    p += al256(r * kHcxMaxSlices * LR);
    s.lsc = reinterpret_cast<float*>(p);
    return s;
}

int hcx_min() {
    static const int v = std::max(0, env_int("STRATA_BATCH_HCX", kHcxMinDefault));
    return v;
}
int hcx_slices() {
    static const int v = [] {
        const int s = env_int("STRATA_BATCH_HCX_SLICES", kHcxSlicesDefault);
        return s >= 1 && s <= kHcxMaxSlices ? s : kHcxSlicesDefault;
    }();
    return v;
}

bool hcx_engine() { return iq_mmq_decode_ok(core::Runtime::get().compute(), kNativeQ8Rows, LR, D, 1); }

bool hcx_window_ok(const FusedGrArgs* a, int n) noexcept {
    if (a == nullptr || n < 1 || n > kHcxMaxRows) return false;
    const FusedGrArgs& a0 = a[0];
    if (!a0.R || !a0.w_norm || !a0.q8_down || !a0.q8_up || !a0.lo || !a0.rs || !a0.mixed ||
        ((uintptr_t) a0.q8_down & 3) != 0 || ((uintptr_t) a0.q8_up & 3) != 0 || ((uintptr_t) a0.q8_mixed & 3) != 0 ||
        (a0.w_inject && !a0.inject_out) || (a0.apply && (!a0.bo_prev || !a0.inj_prev)))
        return false;
    for (int t = 0; t < n; ++t) {
        const FusedGrArgs& x = a[t];
        const auto at = [t](const auto* b, size_t per) { return b == nullptr ? nullptr : b + (size_t) t * per; };
        if (x.R != at(a0.R, D) || x.R_out != x.R || x.apply != a0.apply || x.eps != a0.eps || x.lo != at(a0.lo, LR) ||
            x.rs != at(a0.rs, HC) || x.mixed != at(a0.mixed, N) || x.w_norm != a0.w_norm ||
            x.w_inject != a0.w_inject || x.q8_down != a0.q8_down || x.q8_up != a0.q8_up ||
            (x.w_inject && x.inject_out != at(a0.inject_out, HC)) ||
            (x.apply && (x.bo_prev != at(a0.bo_prev, N) || x.inj_prev != at(a0.inj_prev, HC) ||
                         x.inj_prev == x.inject_out)) ||
            static_cast<const uint8_t*>(x.q8_mixed) != at(static_cast<const uint8_t*>(a0.q8_mixed), (size_t) N / 32 * 36))
            return false;
    }
    return true;
}

bool hcx_take(const FusedGrArgs* a, int n) {
    return hcx_min() > 0 && n >= hcx_min() && hcx_window_ok(a, n) && hcx_engine();
}

void hcx_read(const FusedGrArgs* a, int n, const HcxScratch& s, void* stream, int slices, HcxBody body,
              unsigned long long* stamp_buf, int stamp_i0) {
    if (!hcx_window_ok(a, n) || !s.part || !s.ssg || !s.lq || !s.lsc || slices < 1 || slices > kHcxMaxSlices)
        throw core::DeviceError("hcx_read: invalid arguments");
    auto& q = core::Runtime::get().stream(stream);
    const bool engine = iq_mmq_decode_ok(q, kNativeQ8Rows, LR, D, 1);
    if (body == HcxBody::xmx && !engine) throw core::DeviceError("hcx_read: the device has no int8 XMX tiles");
    const bool x = body == HcxBody::xmx || (body == HcxBody::automatic && engine);
    const FusedGrArgs& a0 = a[0];
    const Win w{const_cast<float*>(a0.R), a0.bo_prev, a0.inj_prev, a0.w_norm, a0.w_inject, a0.q8_down, a0.q8_up,
                a0.lo, a0.rs, a0.inject_out, a0.mixed, static_cast<uint8_t*>(a0.q8_mixed), a0.eps, a0.apply ? 1 : 0, n};
    const sycl::event e = x ? (slices == 2 ? launch<true, 2>(q, w, s, stamp_buf, stamp_i0)
                                           : launch<true, 1>(q, w, s, stamp_buf, stamp_i0))
                            : (slices == 2 ? launch<false, 2>(q, w, s, stamp_buf, stamp_i0)
                                           : launch<false, 1>(q, w, s, stamp_buf, stamp_i0));
    if (!stream) core::Runtime::get().wait(e, "hcx_read");
}

}  // namespace strata::kernels
