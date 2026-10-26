// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/hc_mmq.cpp - see include/strata/kernels/hc_mmq.hpp.  fused_gr.cpp's geometry and element arithmetic;
// the products go to xmx_gemm (BF16) or iq_mmq_grouped / opt-in decode (native Q8_0 rows, kNativeQ8Rows).
#include "strata/kernels/hc_mmq.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/runtime.hpp"
#include "q8_1_finite.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_hc_mmq
#include "strata/kernels/device_arch_tag.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::kernels {
namespace {

constexpr int N = 2560;   // n_embd
constexpr int HC = 4;     // streams
constexpr int D = N * HC;
constexpr int LR = 320;   // hc_lr
constexpr int THREADS = 256;
constexpr int WARP = 32;
constexpr int WARPS = THREADS / WARP;
constexpr size_t Q8_ROW_D = (size_t) D / 32 * 34;   // a kNativeQ8Rows down row
// iq_mmq takes the outputs 128 at a time: the down product's 320 rows as rows 0-255, then 192-319 (64 computed twice,
// the same values)
constexpr int DOWN_A = 256, DOWN_B0 = LR - 128;
static_assert(LR % 32 == 0 && LR > DOWN_A && DOWN_B0 % 32 == 0 && N % THREADS == 0 && N % (WARPS * WARP) == 0,
              "the read's geometry");

using float4 = sycl::float4;
using uint4 = sycl::uint4;

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float warp_max(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float4 load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
inline float dot8(const uint4 w, const float* x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x(), w.y(), w.z(), w.w()};
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(v[j] << 16), x[2 * j], acc);
        acc = sycl::fma(sycl::bit_cast<float>(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

// One value of a 32-value block a lane, the block a sub-group: iq_mmq_quantize's bytes (scale amax / 127, values
// round(x / scale); max is order-free, so the butterfly gives its amax).  `x8` is the block's row of int8 values,
// `xs` its scale.
inline void quant_lane(const sycl::sub_group& sg, float v, int8_t* x8, float* xs) {
    const float amax = warp_max(sg, sycl::fabs(v));
    const float d = amax / 127.0f, id = amax > 0.0f ? 1.0f / d : 0.0f;
    *x8 = (int8_t) (int) sycl::round(v * id);
    if (sg.get_local_linear_id() == 0) *xs = d;
}

// The products' inputs: BF16 copies (xb, lb) or iq_mmq's activations (x8 then the scales: rows * cols bytes, then
// rows * cols / 32 floats).
struct Act {
    uint16_t* bf = nullptr;
    int8_t* x8 = nullptr;
    float* xs = nullptr;
};
Act act(const HcMmqScratch& s, bool q8, int n, int cols, uint16_t* bf) {
    if (!q8) return Act{bf, nullptr, nullptr};
    auto* x8 = reinterpret_cast<int8_t*>(s.xq);
    return Act{nullptr, x8, reinterpret_cast<float*>(x8 + (size_t) n * cols)};
}

}  // namespace

size_t hc_mmq_xq_bytes(int64_t rows) { return iq_mmq_act_bytes(rows, D); }

bool hc_mmq_engine(bool q8) {
    if (q8) return iq_mmq_usable(core::Runtime::get().compute());
    return std::strcmp(gemm_path(XmxType::bf16), "DP4a") != 0;
}

bool hc_mmq_weights_ok(const FusedGrArgs& w, const float* lo, const HcMmqScratch& s) noexcept {
    if (!w.w_norm || !s.up || !s.bounds || !s.ids || !lo) return false;
    if (w.q8_down != nullptr && w.q8_up != nullptr)
        return s.xq != nullptr && iq_mmq_type_ok(kNativeQ8Rows) && ((uintptr_t) w.q8_down & 3) == 0 &&
               ((uintptr_t) w.q8_up & 3) == 0;
    return w.w_down && w.w_up && s.xb && s.lb && xmx_gemm_ok(s.xb, w.w_down, lo, LR, D, LR) &&
           xmx_gemm_ok(s.lb, w.w_up, s.up, D, LR, D);
}

// Keep the established prompt-MMQ opt-in independent of the decode-only HC lever.
constexpr int kHcMmqMinDefault = 0;
constexpr int kHcXmxMinDefault = 0;
int hc_mmq_min() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_BATCH_HC_MMQ_MIN");
        return e && *e ? std::atoi(e) : kHcMmqMinDefault;
    }();
    return v;
}
int hc_xmx_min() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_BATCH_HC_XMX");
        return dmmq::dispatch_min(e && *e ? std::atoi(e) : kHcXmxMinDefault,
                                  std::getenv("STRATA_DMMQ_MIN_ROWS"));
    }();
    return v;
}

dmmq::HcPath hc_mmq_path(const FusedGrArgs& w, const float* lo, const HcMmqScratch& s, int rows) {
    const bool q8 = w.q8_down && w.q8_up;
    auto& q = core::Runtime::get().compute();
    const bool decode = q8 && hc_xmx_min() > 0 &&
        iq_mmq_decode_ok(q, kNativeQ8Rows, LR, D, rows) &&
        iq_mmq_decode_ok(q, kNativeQ8Rows, D, LR, rows);
    return dmmq::hc_path(rows, hc_mmq_min(), hc_xmx_min(), q8, hc_mmq_weights_ok(w, lo, s),
                         hc_mmq_min() > 0 && hc_mmq_engine(q8), decode);
}

void fused_gr_read_mmq(const FusedGrArgs* a, int n, float* xn_scratch, const HcMmqScratch& s, void* stream,
                       unsigned long long* stamp_buf, int stamp_i0) {
    if (n < 1 || a == nullptr || xn_scratch == nullptr || !hc_mmq_weights_ok(a[0], a[0].lo, s))
        throw core::DeviceError("fused_gr_read_mmq: invalid arguments");
    const FusedGrArgs& a0 = a[0];
    for (int t = 0; t < n; ++t) {
        const FusedGrArgs& x = a[t];
        const auto at = [t](const auto* base, size_t per) { return base == nullptr ? nullptr : base + (size_t) t * per; };
        if (x.R != at(a0.R, D) || x.R_out != x.R || x.apply != a0.apply || x.eps != a0.eps ||
            x.lo != at(a0.lo, LR) || x.rs != at(a0.rs, HC) || x.mixed != at(a0.mixed, N) || !x.mixed ||
            !x.rs || x.w_norm != a0.w_norm || x.w_down != a0.w_down || x.w_up != a0.w_up ||
            x.w_inject != a0.w_inject || x.q8_down != a0.q8_down || x.q8_up != a0.q8_up ||
            (x.w_inject && (!x.inject_out || x.inject_out != at(a0.inject_out, HC))) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || x.bo_prev != at(a0.bo_prev, N) ||
                         x.inj_prev != at(a0.inj_prev, HC) || x.inj_prev == x.inject_out)) ||
            static_cast<const uint8_t*>(x.q8_mixed) != at(static_cast<const uint8_t*>(a0.q8_mixed), (size_t) N / 32 * 36))
            throw core::DeviceError("fused_gr_read_mmq: row " + std::to_string(t) + " is not the window's layout");
    }
    auto& q = core::Runtime::get().stream(stream);
    const bool q8 = a0.q8_down != nullptr && a0.q8_up != nullptr;
    const bool decode = hc_mmq_path(a0, a0.lo, s, n) == dmmq::HcPath::decode;
    const bool apply = a0.apply;
    const float eps = a0.eps;
    float* const R = const_cast<float*>(a0.R);
    const float *bo = a0.bo_prev, *ip = a0.inj_prev, *wn = a0.w_norm;
    float *rs = a0.rs, *lo = a0.lo, *mixed = a0.mixed, *inj = a0.inject_out, *xn = xn_scratch, *up = s.up;
    const uint16_t* wi = a0.w_inject;
    auto* q8m = static_cast<uint8_t*>(a0.q8_mixed);

    // norm: fused_gr_read_multi's split norm (a group a row and stream, its visiting and reduction order), R' written
    // back where it was read, and xn as the down product's input
    const Act ax = act(s, q8, n, D, s.xb);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(WARPS), h);
        sycl::local_accessor<float, 1> s_rs(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n * HC * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int k = (int) it.get_group(0) / HC, c = (int) it.get_group(0) % HC;
            const int t = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id();
            const int warp = (int) sg.get_group_linear_id();
            float* r0 = R + (size_t) k * D;
            float* x = xn + (size_t) k * D;
            const float gw = apply ? 2.0f * sigmoidf_(ip[k * HC + c] / (float) HC) : 0.0f;
            float ss = 0.0f;
            for (int i = t * 4; i < D; i += THREADS * 4) {
                if (i / N != c) continue;
                const int d = i - c * N;
                float4 r = load4(r0 + i);
                if (apply) {
                    const float4 b = load4(bo + (size_t) k * N + d);
                    r.x() = sycl::fma(b.x(), gw, r.x()); r.y() = sycl::fma(b.y(), gw, r.y());
                    r.z() = sycl::fma(b.z(), gw, r.z()); r.w() = sycl::fma(b.w(), gw, r.w());
                    *reinterpret_cast<float4*>(r0 + i) = r;
                }
                const float4 g = load4(wn + i);
                ss += r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
                x[i] = r.x() * g.x(); x[i + 1] = r.y() * g.y(); x[i + 2] = r.z() * g.z(); x[i + 3] = r.w() * g.w();
            }
            const float v = warp_sum(sg, ss);
            if (lane == 0) part[warp] = v;
            sycl::group_barrier(it.get_group());
            if (t == 0) {
                float sum = 0.0f;
                for (int w = 0; w < WARPS; ++w) sum += part[w];
                s_rs[0] = sycl::rsqrt(sum / (float) N + eps);
                rs[k * HC + c] = s_rs[0];
            }
            sycl::group_barrier(it.get_group());
            const float sr = s_rs[0];
            // a sub-group a block of 32 (the quantized input's), every lane on every step
            for (int j = 0; j < N / THREADS; ++j) {
                const int i = c * N + j * THREADS + warp * WARP + lane;
                const float y = x[i] * sr;
                x[i] = y;
                const size_t o = (size_t) k * D + i;
                if (ax.bf != nullptr) ax.bf[o] = bf16_from_f32(y);
                else quant_lane(sg, y, ax.x8 + o, ax.xs + o / 32);
            }
        });
    });
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, &q);

    // down: lo = xn . w_down^T
    if (decode) {
        iq_mmq_decode(q, kNativeQ8Rows, a0.q8_down, LR, D, s.xq, n, lo, LR);
    } else if (q8) {
        iq_mmq_grouped(q, kNativeQ8Rows, a0.q8_down, 0, DOWN_A, D, 1, s.xq, n, s.bounds, n, s.ids, lo, LR);
        iq_mmq_grouped(q, kNativeQ8Rows, a0.q8_down + (size_t) DOWN_B0 * Q8_ROW_D, 0, LR - DOWN_B0, D, 1, s.xq, n,
                       s.bounds, n, s.ids, lo + DOWN_B0, LR);
    } else {
        xmx_gemm(XmxType::bf16, s.xb, a0.w_down, lo, n, LR, D, LR, stream);
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, &q);

    // mid: a sub-group a (row, j): j < hc the inject row j (the multi read's lane order and xor tree), j = hc lo's silu
    // and the up product's input
    const Act al = act(s, q8, n, LR, s.lb);
    q.parallel_for(sycl::nd_range<1>((size_t) n * (HC + 1) * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int k = (int) it.get_group(0) / (HC + 1), j = (int) it.get_group(0) % (HC + 1);
        if (j < HC) {
            if (wi == nullptr) return;
            const uint16_t* wrow = wi + (size_t) j * D;
            const float* x = xn + (size_t) k * D;
            float acc = 0.0f;
#pragma unroll 8
            for (int qq = 0; qq < D / 8 / WARP; ++qq) {
                const int jj = lane + WARP * qq;
                const uint4 w = reinterpret_cast<const uint4*>(wrow)[jj];
                const float4 x0 = load4(x + (size_t) jj * 8), x1 = load4(x + (size_t) jj * 8 + 4);
                const float xv[8] = {x0.x(), x0.y(), x0.z(), x0.w(), x1.x(), x1.y(), x1.z(), x1.w()};
                acc += dot8(w, xv);
            }
            const float sum = warp_sum(sg, acc);
            if (lane == 0) inj[k * HC + j] = sum;
            return;
        }
        for (int r = lane; r < LR; r += WARP) {
            const size_t o = (size_t) k * LR + r;
            const float v = lo[o] / (float) HC;
            const float y = v / (1.0f + sycl::exp(-v));
            lo[o] = y;
            if (al.bf != nullptr) al.bf[o] = bf16_from_f32(y);
            else quant_lane(sg, y, al.x8 + o, al.xs + o / 32);
        }
    });

    // up: u = lo . w_up^T
    if (decode) iq_mmq_decode(q, kNativeQ8Rows, a0.q8_up, D, LR, s.xq, n, up, D);
    else if (q8) iq_mmq_grouped(q, kNativeQ8Rows, a0.q8_up, 0, D, LR, 1, s.xq, n, s.bounds, n, s.ids, up, D);
    else xmx_gemm(XmxType::bf16, s.lb, a0.w_up, up, n, D, LR, D, stream);

    // mix: a sub-group 32 columns of a row; mixed's q8_1 block from them as gr_up_q8's 32-column up writes it
    const auto e = q.parallel_for(sycl::nd_range<1>((size_t) n * N, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int b = (int) it.get_group(0) * WARPS + (int) sg.get_group_linear_id();   // the row's block
        const int k = b / (N / WARP), d = (b - k * (N / WARP)) * WARP + lane;
        const float* x = xn + (size_t) k * D;
        const float* u = up + (size_t) k * D;
        float g[HC];
        for (int c = 0; c < HC; ++c) g[c] = x[c * N + d] * sigmoidf_(u[c * N + d]);
        float sum = 0.0f;
        for (int c = 0; c < HC; ++c) sum += g[c];
        const float xi = sum / (float) HC;
        mixed[(size_t) k * N + d] = xi;
        if (q8m == nullptr) return;
        float amax = sycl::fabs(xi), bs = xi;
        for (int o = WARP / 2; o > 0; o >>= 1) {
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
            bs += sycl::permute_group_by_xor(sg, bs, o);
        }
        const float dd = xe::q8_1_finite(amax / 127.0f);
        uint8_t* blk = q8m + (size_t) k * (N / 32) * 36 + (size_t) (d / 32) * 36;
        blk[4 + lane] = (uint8_t) xe::q8_1_quant(xi, dd, amax);
        if (lane == 0) {
            const sycl::half2 ds(sycl::half(dd), sycl::half(xe::q8_1_finite(bs)));
            *reinterpret_cast<sycl::half2*>(blk) = ds;
        }
    });
    if (!stream) core::Runtime::get().wait(e, "fused_gr_read_mmq");
}

}  // namespace strata::kernels
