// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/fused_gr_win.cpp - STRATA_HC_Q8_WIN forms 2 and up: the Q8_0 hyper-connection read rewritten for a
// window of up to 16 tokens, bitwise fused_gr.cpp's gr_read_q8 / gr_up_q8 (hc_read16_parity).
//
// The established kernels and what changes here (each output keeps its summation order):
//   * down: a sub-group's RPW rows share each staged activation load (the established kernel loaded it once a row),
//     the staging spreads (token, 32 float4) pieces over every sub-group (it left 96 of 256 work-items idle), and the
//     xor-tree sums of the RPW rows x TCH tokens run as one reduce-scatter (each stage halves the values a lane keeps,
//     so 32 values cost 31 shuffles, not 160; the adds are warp_sum's, own + partner, so every sum keeps its bits);
//   * lo: the partial dots are added up ONCE, by their own launch (every one of the up kernel's 80 or 160 work-groups
//     added all T x 324 x 16 of them again: 26 MB of reads at 8 tokens);
//   * up: RPW rows a sub-group sharing each lo load, the same reduce-scatter.
// The shapes (RPW, TCH, register file) are grw::read's; hc_read16_parity times each.
#include "fused_gr_win.hpp"

#include "fuse_counter.hpp"
#include "q8_1_finite.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <climits>
#include <cstdint>
#include <utility>

namespace strata::kernels::grw {
namespace {

namespace syclex = sycl::ext::oneapi::experimental;
using float4 = sycl::float4;

constexpr int kStage = Q8_KC / 4 / WARP;   // a chunk's float4 for one token: 5 sub-groups' worth

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float4 load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
inline float4 q8_four(uint32_t q, uint16_t d) {
    const float s = (float) sycl::bit_cast<sycl::half>(d);
    return float4(s * (float) (int8_t) (q & 255u), s * (float) (int8_t) ((q >> 8) & 255u),
                  s * (float) (int8_t) ((q >> 16) & 255u), s * (float) (int8_t) (q >> 24));
}

// warp_sum of each of v[0..n), scattered over the lanes: at a stage whose values split evenly, a lane keeps half of
// them (the upper half where its bit `o` is set) and adds the partner's copy of each - own + partner, the operands
// warp_sum's add has at that lane, so each sum is warp_sum's bits; at another stage it adds them all as warp_sum.  On
// return v[0..rs_left(n)) are the sums of values base..base + rs_left(n), the same in lanes that differ only in rs_dup(n).
template <int n, int o = 16>
inline void reduce_scatter(const sycl::sub_group& sg, float* v, int lane, int& base) {
    if constexpr (o > 0) {
        if constexpr (n % 2 == 0) {
            constexpr int h = n / 2;
            const bool hi = (lane & o) != 0;
#pragma unroll
            for (int i = 0; i < h; ++i) {
                const float keep = hi ? v[i + h] : v[i], send = hi ? v[i] : v[i + h];
                v[i] = keep + sycl::permute_group_by_xor(sg, send, o);
            }
            if (hi) base += h;
            reduce_scatter<h, o / 2>(sg, v, lane, base);
        } else {
#pragma unroll
            for (int i = 0; i < n; ++i) v[i] += sycl::permute_group_by_xor(sg, v[i], o);
            reduce_scatter<n, o / 2>(sg, v, lane, base);
        }
    }
}
constexpr int rs_left(int n, int o = 16) { return o == 0 ? n : n % 2 == 0 ? rs_left(n / 2, o / 2) : rs_left(n, o / 2); }
constexpr int rs_dup(int n, int o = 16) { return o == 0 ? 0 : n % 2 == 0 ? rs_dup(n / 2, o / 2) : (o | rs_dup(n, o / 2)); }

// The sums of v[0..n) for the lanes: the reduce-scatter, or (G) warp_sum of each value with lane i keeping value i, as
// the established kernels do.  G for n < 16: where the reduce-scatter is left with one value and two or more stages to
// go (n 4 and 8), its sums were 1 ulp off warp_sum's in hc_read16_parity (IGC compiles that tail of shuffle-adds
// another way), while every n >= 16 the forms take matched.  On return v[0..kSumsLeft) are the sums of values base,
// base + 1, ..., held where the result is true.
template <int n, bool G> constexpr int kSumsLeft = G ? 1 : rs_left(n);
template <int n> constexpr int kRsDup = rs_dup(n);
template <int n, bool G>
inline bool sums(const sycl::sub_group& sg, float* v, int lane, int& base) {
    if constexpr (G) {
        static_assert(n <= WARP, "a lane a value");
        float mine = 0.0f;
#pragma unroll
        for (int i = 0; i < n; ++i) {
            const float s = warp_sum(sg, v[i]);
            if (lane == i) mine = s;
        }
        v[0] = mine;
        base = lane;
        return lane < n;
    } else {
        base = 0;
        reduce_scatter<n>(sg, v, lane, base);
        return (lane & kRsDup<n>) == 0;
    }
}

template <bool LG>
constexpr auto kernel_props() {
    if constexpr (LG)
        return syclex::properties{syclex::sub_group_size<WARP>, sycl::ext::intel::experimental::grf_size<256>};
    else
        return syclex::properties{syclex::sub_group_size<WARP>};
}

// a kernel's shape: RPW rows a sub-group (sharing each activation load), TCH tokens a pass over the weights (their
// RPW x TCH sums in one reduce-scatter), the large register file
template <int RPW_, int TCH_, bool LG_>
struct Shape {
    static constexpr int RPW = RPW_, TCH = TCH_;
    static constexpr bool LG = LG_;
};
// a form: the down kernel's shape and the up kernel's; FL (homelab): the lo sums in the down launch's tail (lo_one by the
// last work-group of each row group to finish), not a launch of their own
template <class D_, class U_, bool FL_ = false>
struct Form {
    using D = D_;
    using U = U_;
    static constexpr bool FL = FL_;
};

// lo (or the injection) of token k, row r from the partial dots, and rs (r < HC): the Lo launch's arithmetic, also the
// fused tail's
inline void lo_one(const GrWin& m, const float* part, const float* ssg, float* lo, float* rs, int k, int r) {
    float rsc[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        float ss = 0.0f;
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        rsc[c] = sycl::rsqrt(ss / (float) N + m.eps);
    }
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        float pp = 0.0f;
        for (int j = 0; j < Q8_CPS; ++j) pp += part[(size_t) (k * Q8_NKC + c * Q8_CPS + j) * PR + r];
        sum = sycl::fma(rsc[c], pp, sum);
    }
    if (r < LR) {
        const float x = sum / (float) HC;
        lo[k * LR + r] = x / (1.0f + sycl::exp(-x));
    } else {
        m.inject_out(k)[r - LR] = sum;
    }
    if (r < HC) {
        const float v = r == 0 ? rsc[0] : r == 1 ? rsc[1] : r == 2 ? rsc[2] : rsc[3];
        rs[k * HC + r] = v;
        m.rs_out(k)[r] = v;
    }
}

// gr_read_q8's down: (row groups of WARPS x RPW rows + the inject rows) x 16 chunks of 640 columns; each writes its
// rows' partial dots for every token and (row group 0) the chunk's sums of squares.
template <int T, class S, bool FL = false>
struct Down {
    static constexpr int RPW = S::RPW, TCH = S::TCH, NRG = LR / (WARPS * RPW);
    static_assert(LR % (WARPS * RPW) == 0, "whole row groups");
    static_assert(NRG + 1 <= 16, "a fuse counter a row group");
    GrWin m;
    float* part;
    float* ssg;
    sycl::local_accessor<float4, 1> xs;   // [T][Q8_KC / 4]: the chunk's R' * w_norm
    sycl::local_accessor<float, 1> red;   // [T][kStage]: a staging sub-group's sum of squares
    // FL: the fused lo tail's state (unused otherwise)
    float* lo = nullptr;
    float* rs = nullptr;
    float* ssg_x = nullptr;                // row groups 1..NRG's ssg copies, T * Q8_NKC each
    uint32_t* counters = nullptr;          // NRG + 1, one a row group
    float* ssg_of(int rg) const { return rg == 0 ? ssg : ssg_x + (size_t) (rg - 1) * T * Q8_NKC; }
    sycl::local_accessor<int, 1> flag;

    template <int K0>
    void dots(const sycl::sub_group& sg, int lane, int warp, int kc, bool inj, int nrows, int row0) const {
        constexpr int TC = T - K0 < TCH ? T - K0 : TCH;
        const int sub = lane & 7, bl = lane >> 3;
        float acc[RPW * TC];   // [row][token]
#pragma unroll
        for (int i = 0; i < RPW * TC; ++i) acc[i] = 0.0f;
#pragma unroll
        for (int st = 0; st < Q8_SPB; ++st) {
            const int v = 32 * (4 * st + bl) + 4 * sub;
            float4 w[RPW];
            if (inj) {
                const sycl::uint2 b = *reinterpret_cast<const sycl::uint2*>(m.w_inject + (size_t) row0 * D +
                                                                           (size_t) kc * Q8_KC + v);
                w[0] = float4(sycl::bit_cast<float>(b.x() << 16), sycl::bit_cast<float>(b.x() & 0xffff0000u),
                              sycl::bit_cast<float>(b.y() << 16), sycl::bit_cast<float>(b.y() & 0xffff0000u));
#pragma unroll
                for (int r = 1; r < RPW; ++r) w[r] = float4(0.0f);
            } else {
#pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    const uint8_t* qrow = m.q8_down + (size_t) (row0 + r) * Q8_ROW_D;
                    const int col = kc * Q8_KC + v;
                    w[r] = q8_four(*reinterpret_cast<const uint32_t*>(qrow + col),
                                   reinterpret_cast<const uint16_t*>(qrow + D)[col / 32]);
                }
            }
#pragma unroll
            for (int k = 0; k < TC; ++k) {
                const float4 x = xs[(K0 + k) * Q8_KC / 4 + v / 4];
#pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    float& a = acc[r * TC + k];
                    a = sycl::fma(w[r].x(), x.x(), a); a = sycl::fma(w[r].y(), x.y(), a);
                    a = sycl::fma(w[r].z(), x.z(), a); a = sycl::fma(w[r].w(), x.w(), a);
                }
            }
        }
        constexpr bool G = RPW * TC < 16;
        int base = 0;
        if (sums<RPW * TC, G>(sg, acc, lane, base)) {
#pragma unroll
            for (int j = 0; j < kSumsLeft<RPW * TC, G>; ++j) {
                const int idx = base + j, r = idx / TC, k = K0 + idx % TC;
                if (r >= nrows) continue;
                const int prow = inj ? LR + warp : row0 + r;
                part[((size_t) k * Q8_NKC + kc) * PR + prow] = acc[j];
            }
        }
        if constexpr (K0 + TC < T) dots<K0 + TC>(sg, lane, warp, kc, inj, nrows, row0);
    }

    void operator()(sycl::nd_item<2> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int t = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id();
        const int warp = (int) sg.get_group_linear_id();
        const int rg = (int) it.get_group(0), kc = (int) it.get_group(1), c = kc / Q8_CPS;
        const bool inj = rg == NRG;
        const int nrows = inj ? ((m.w_inject != nullptr && warp < HC) ? 1 : 0) : RPW;
        const int row0 = inj ? warp : (rg * WARPS + warp) * RPW;
        // staging: piece j = token j / 5, float4 32 (j % 5) + lane of the chunk - the established kernel's work-item
        // 32 (j % 5) + lane, whose sub-group's sum of squares this sub-group's is
        for (int j = warp; j < T * kStage; j += WARPS) {
            const int k = j / kStage, w5 = j - k * kStage, i = WARP * w5 + lane;
            const int col = kc * Q8_KC + 4 * i;
            const bool ap = m.applies(k);
            const float gw = ap ? 2.0f * sigmoidf_(m.inj_prev(k)[c] / (float) HC) : 0.0f;
            float4 r = load4(m.R[k] + col);
            if (ap) {
                const float4 b = load4(m.bo_prev(k) + (col - c * N));
                r.x() = sycl::fma(b.x(), gw, r.x()); r.y() = sycl::fma(b.y(), gw, r.y());
                r.z() = sycl::fma(b.z(), gw, r.z()); r.w() = sycl::fma(b.w(), gw, r.w());
            }
            const float4 g = load4(m.w_norm + col);
            float ss = 0.0f;
            ss += r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
            xs[k * Q8_KC / 4 + i] = r * g;
            const float v = warp_sum(sg, ss);
            if (lane == 0) red[k * kStage + w5] = v;
        }
        sycl::group_barrier(it.get_group());
        // the established sum over its 8 sub-groups, of which the last 3 held zeros; FL: every row group writes its own
        // copy (the same bits: the same staging), so its tail needs no other row group
        if ((FL || rg == 0) && t < T) {
            float sum = 0.0f;
            for (int w = 0; w < kStage; ++w) sum += red[t * kStage + w];
            (FL ? ssg_of(rg) : ssg)[t * Q8_NKC + kc] = sum;
        }
        if (nrows > 0) dots<0>(sg, lane, warp, kc, inj, nrows, row0);
        if constexpr (FL) {
            // the last of the row group's Q8_NKC work-groups to finish adds up the group's rows (Lo's arithmetic)
            if (!xe::fuse_arrive_last(it, counters + rg, (uint32_t) Q8_NKC, flag)) return;
            const float* ssg_rg = ssg_of(rg);
            const int r0 = inj ? LR : rg * WARPS * RPW;
            const int nr = inj ? (m.w_inject != nullptr ? HC : 0) : WARPS * RPW;
            for (int i = t; i < T * nr; i += THREADS) {
                const int k = i / nr;
                lo_one(m, part, ssg_rg, lo, rs, k, r0 + (i - k * nr));
            }
        }
    }
    auto get(syclex::properties_tag) const { return kernel_props<S::LG>(); }
};

// the partial dots added up once: rs and lo (in the scratch) and the injection, gr_up_q8's arithmetic
struct Lo {
    GrWin m;
    const float* part;
    const float* ssg;
    float* lo;
    float* rs;
    void operator()(sycl::nd_item<1> it) const {
        const int i = (int) it.get_global_id(0);
        if (i >= m.T * PR) return;
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && m.w_inject == nullptr) return;
        lo_one(m, part, ssg, lo, rs, k, r);
    }
};

// gr_up_q8: COLS columns x 4 streams a work-group, RPW rows (of one stream) a sub-group at a time
template <int T, int COLS, class S>
struct Up {
    static constexpr int RPW = S::RPW, TCH = S::TCH;
    static constexpr int UT = THREADS * COLS / 16, UW = UT / WARP;   // 16 columns a 256-thread group
    static_assert(COLS % RPW == 0, "a sub-group's rows in one stream");
    GrWin m;
    const float* lo_g;
    const float* rs_g;
    sycl::local_accessor<float4, 1> lo;   // [T][LR / 4]
    sycl::local_accessor<float, 1> rs_s;  // [T][HC]
    sycl::local_accessor<float, 1> g;     // [T][HC][COLS]
    sycl::local_accessor<float, 1> mx;    // [T][COLS]

    template <int K0>
    void dots(const sycl::sub_group& sg, int lane, int rb, int d0) const {
        constexpr int TC = T - K0 < TCH ? T - K0 : TCH;
        const int c = rb * RPW / COLS, dd0 = rb * RPW - c * COLS, i0 = c * N + d0 + dd0;
        float acc[RPW * TC];
#pragma unroll
        for (int i = 0; i < RPW * TC; ++i) acc[i] = 0.0f;
#pragma unroll
        for (int st = 0; st < 3; ++st) {   // lane: values 4 lane + 128 st (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * st;
            if (v < LR) {
                float4 w[RPW];
#pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    const uint8_t* qrow = m.q8_up + (size_t) (i0 + r) * Q8_ROW_U;
                    w[r] = q8_four(*reinterpret_cast<const uint32_t*>(qrow + v),
                                   reinterpret_cast<const uint16_t*>(qrow + LR)[v / 32]);
                }
#pragma unroll
                for (int k = 0; k < TC; ++k) {
                    const float4 x = lo[(K0 + k) * LR / 4 + v / 4];
#pragma unroll
                    for (int r = 0; r < RPW; ++r) {
                        float& a = acc[r * TC + k];
                        a = sycl::fma(w[r].x(), x.x(), a); a = sycl::fma(w[r].y(), x.y(), a);
                        a = sycl::fma(w[r].z(), x.z(), a); a = sycl::fma(w[r].w(), x.w(), a);
                    }
                }
            }
        }
        constexpr bool G = RPW * TC < 16;
        int base = 0;
        if (sums<RPW * TC, G>(sg, acc, lane, base)) {
#pragma unroll
            for (int j = 0; j < kSumsLeft<RPW * TC, G>; ++j) {
                const int idx = base + j, r = idx / TC, k = K0 + idx % TC;
                const int dd = dd0 + r, i = i0 + r;
                float rv = m.R[k][i];
                const float wn = m.w_norm[i];
                if (m.applies(k)) {
                    const float bo = m.bo_prev(k)[d0 + dd], ip = m.inj_prev(k)[c];
                    rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                    m.R_out[k][i] = rv;
                }
                const float x = rv * wn * rs_s[k * HC + c];
                g[(k * HC + c) * COLS + dd] = x * sigmoidf_(acc[j]);
            }
        }
        if constexpr (K0 + TC < T) dots<K0 + TC>(sg, lane, rb, d0);
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int t = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id();
        const int warp = (int) sg.get_group_linear_id();
        const int d0 = (int) it.get_group(0) * COLS;
        for (int i = t; i < T * LR / 4; i += UT) lo[i] = load4(lo_g + 4 * i);
        if (t < T * HC) rs_s[t] = rs_g[t];
        sycl::group_barrier(it.get_group());
        for (int rb = warp; rb < HC * COLS / RPW; rb += UW) dots<0>(sg, lane, rb, d0);
        sycl::group_barrier(it.get_group());
        for (int i = t; i < T * COLS; i += UT) {
            const int k = i / COLS, col = i - k * COLS;
            float sum = 0.0f;
            for (int c = 0; c < HC; ++c) sum += g[(k * HC + c) * COLS + col];
            m.mixed(k)[d0 + col] = sum / (float) HC;
            if constexpr (COLS == 32) mx[k * COLS + col] = sum / (float) HC;
        }
        if constexpr (COLS == 32) {
            // mixed's q8_1 block (native_quantize_q8_1's arithmetic), a sub-group a token
            if (m.q8m != nullptr) {
                sycl::group_barrier(it.get_group());
                for (int k = warp; k < T; k += UW) {
                    const float xi = mx[k * COLS + lane];
                    float amax = sycl::fabs(xi), sum = xi;
                    for (int o = WARP / 2; o > 0; o >>= 1) {
                        amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
                        sum += sycl::permute_group_by_xor(sg, sum, o);
                    }
                    const float d = xe::q8_1_finite(amax / 127.0f);
                    uint8_t* blk = static_cast<uint8_t*>(m.q8_mixed(k)) + (size_t) (d0 / 32) * 36;
                    blk[4 + lane] = (uint8_t) xe::q8_1_quant(xi, d, amax);
                    if (lane == 0) {
                        const sycl::half2 ds(sycl::half(d), sycl::half(xe::q8_1_finite(sum)));
                        *reinterpret_cast<sycl::half2*>(blk) = ds;
                    }
                }
            }
        }
    }
    auto get(syclex::properties_tag) const { return kernel_props<S::LG>(); }
};

template <int T, class F>
sycl::event read_t(sycl::queue& q, const GrWin& w, float* scratch, int parts, unsigned long long* stamp_buf,
                   int stamp_i0) {
    float* part = scratch;
    float* ssg = part + (size_t) T * Q8_NKC * PR;
    float* lo = ssg + (size_t) T * Q8_NKC;
    float* rs = lo + (size_t) T * LR;
    // FL: row groups 1.. their own ssg copies after rs (row group 0's is ssg itself): T * (5,524 + 16 NRG) of T * 10,240
    float* ssg_more = rs + (size_t) T * HC;
    // FL with both the down and the lo parts: the lo sums in the down launch; with lo alone (timing), the Lo launch
    const bool fl = F::FL && (parts & 3) == 3;
    uint32_t* counters = fl ? xe::fuse_counters(q) + xe::kFuseHcLo : nullptr;   // outside the command group: it may fill
    sycl::event e;
    if (parts & 1)
        e = q.submit([&](sycl::handler& h) {
            const auto go = [&](auto k) {
                k.xs = sycl::local_accessor<float4, 1>(sycl::range<1>((size_t) T * Q8_KC / 4), h);
                k.red = sycl::local_accessor<float, 1>(sycl::range<1>((size_t) T * kStage), h);
                k.flag = sycl::local_accessor<int, 1>(sycl::range<1>(1), h);
                h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (decltype(k)::NRG + 1) * THREADS, Q8_NKC),
                                                 sycl::range<2>(THREADS, 1)), k);
            };
            if constexpr (F::FL) {
                if (fl) {
                    Down<T, typename F::D, true> k{w, part, ssg};
                    k.lo = lo;
                    k.rs = rs;
                    k.ssg_x = ssg_more;
                    k.counters = counters;
                    go(k);
                    return;
                }
            }
            go(Down<T, typename F::D>{w, part, ssg});
        });
    if (stamp_buf) {   // the norm is in `down`: both of the BF16 read's stamps after it
        gpu_stamp(stamp_buf, stamp_i0, &q);
        gpu_stamp(stamp_buf, stamp_i0 + 1, &q);
    }
    if ((parts & 2) && !fl) {
        constexpr int WG = 128;
        const size_t n = ((size_t) T * PR + WG - 1) / WG * WG;
        e = q.parallel_for(sycl::nd_range<1>(n, WG), Lo{w, part, ssg, lo, rs});
    }
    if (parts & 4) {
        const auto up = [&](auto cols) {
            constexpr int COLS = decltype(cols)::value;
            return q.submit([&](sycl::handler& h) {
                using K = Up<T, COLS, typename F::U>;
                K k{w, lo, rs, sycl::local_accessor<float4, 1>(sycl::range<1>((size_t) T * LR / 4), h),
                                 sycl::local_accessor<float, 1>(sycl::range<1>((size_t) T * HC), h),
                                 sycl::local_accessor<float, 1>(sycl::range<1>((size_t) T * HC * COLS), h),
                                 sycl::local_accessor<float, 1>(sycl::range<1>((size_t) T * COLS), h)};
                h.parallel_for(sycl::nd_range<1>((size_t) (N / COLS) * K::UT, K::UT), k);
            });
        };
        // a q8_1 block is 32 columns: the work-group that writes it owns all of them
        e = w.q8m != nullptr ? up(std::integral_constant<int, 32>{}) : up(std::integral_constant<int, 16>{});
    }
    return e;
}

// a form's shapes in the default register file: the OpenCL CPU device (the parity tests) refuses the large one's option
template <class S> using DefaultRf = Shape<S::RPW, S::TCH, false>;
template <class F> using DefaultRfForm = Form<DefaultRf<typename F::D>, DefaultRf<typename F::U>, F::FL>;

template <class F, int... I>
sycl::event read_n(sycl::queue& q, const GrWin& w, float* scratch, int parts, unsigned long long* sb, int si,
                   std::integer_sequence<int, I...>) {
    sycl::event e;
    if constexpr (F::D::LG || F::U::LG) {
        if (!q.get_device().is_gpu()) {
            (void) ((w.T == I + 1 ? (e = read_t<I + 1, DefaultRfForm<F>>(q, w, scratch, parts, sb, si), true) : false) || ...);
            return e;
        }
    }
    (void) ((w.T == I + 1 ? (e = read_t<I + 1, F>(q, w, scratch, parts, sb, si), true) : false) || ...);
    return e;
}

template <class P>
bool strided(const FusedGrArgs* a, int n, P FusedGrArgs::* f, char*& base, int& stride) {
    const auto at = [&](int t) { return (intptr_t) (a[t].*f); };
    const intptr_t d = n > 1 ? at(1) - at(0) : 0;
    if (d > INT_MAX || d < INT_MIN) return false;
    for (int t = 0; t < n; ++t)
        if (at(t) != at(0) + (intptr_t) t * d) return false;
    base = reinterpret_cast<char*>(at(0));
    stride = (int) d;
    return true;
}

}  // namespace

bool pack(const FusedGrArgs* a, int n, GrWin& w) {
    if (n < 1 || n > GrWin::kMax) return false;
    w = GrWin{};
    w.T = n;
    w.eps = a[0].eps;
    w.w_norm = a[0].w_norm; w.w_inject = a[0].w_inject; w.q8_down = a[0].q8_down; w.q8_up = a[0].q8_up;
    for (int t = 0; t < n; ++t) {
        if (a[t].eps != a[0].eps) return false;
        w.R[t] = a[t].R;
        w.R_out[t] = a[t].R_out;
        if (a[t].apply) w.apply |= 1u << t;
    }
    char* bo = nullptr;
    char* inj = nullptr;
    const bool ok = strided(a, n, &FusedGrArgs::bo_prev, bo, w.bo_s) && strided(a, n, &FusedGrArgs::inj_prev, inj, w.inj_s) &&
                    strided(a, n, &FusedGrArgs::rs, w.rs, w.rs_s) && strided(a, n, &FusedGrArgs::inject_out, w.io, w.io_s) &&
                    strided(a, n, &FusedGrArgs::mixed, w.mx, w.mx_s) && strided(a, n, &FusedGrArgs::q8_mixed, w.q8m, w.q8m_s);
    w.bo = bo;
    w.inj = inj;
    return ok;
}

sycl::event read(sycl::queue& q, const GrWin& w, float* scratch, int shape, int parts, unsigned long long* stamp_buf,
                 int stamp_i0) {
    static_assert(sizeof(GrWin) <= 512, "the window's arguments stay far below the kernel-parameter limit");
    const auto seq = std::make_integer_sequence<int, GrWin::kMax>{};
    // hc_read16_parity on the B70 (us a call in a graph, mixed's q8_1 off / on): up to 8 tokens the default register
    // file with 4 rows and 8 tokens in both kernels is the fastest form (1: 13.3 / 14.5, 8: 37.7 / 40.3; the 2x8 path
    // 16.0 / 23.4 and 56.7 / 68.7); past 8 tokens down in 8 rows of 8 tokens and up in 2 rows of 16 tokens, both in
    // the large register file (16: 72.0 / 75.3, the 2x8 path 112.2 / 136.0; a default-file up after a large-file down
    // lost up to 17 us)
    using S48 = Shape<4, 8, false>;
    using Small = Form<S48, S48>;
    using Big = Form<Shape<8, 8, true>, Shape<2, 16, true>>;
    switch (shape) {
    case 0: return w.T <= 8 ? read_n<Small>(q, w, scratch, parts, stamp_buf, stamp_i0, seq)
                            : read_n<Big>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    case 1: return read_n<Form<Shape<8, 8, true>, Shape<2, 16, true>>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    case 2: return read_n<Form<Shape<5, 8, true>, Shape<2, 16, true>>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    case 3: return read_n<Form<Shape<5, 8, true>, Shape<1, 16, true>>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    case 4: return read_n<Form<S48, Shape<2, 16, false>>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    case 5: return read_n<Form<Shape<8, 8, true>, Shape<1, 16, false>>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    case 6: return read_n<Form<Shape<5, 8, true>, Shape<1, 16, false>>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    case 7: return read_n<Form<Shape<10, 4, true>, Shape<1, 16, true>>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    // homelab: the lo sums in the down launch's tail (FL), a launch fewer a read
    case 8: return w.T <= 8 ? read_n<Form<S48, S48, true>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq)
                            : read_n<Form<Shape<8, 8, true>, Shape<2, 16, true>, true>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    // and up in 4 rows a sub-group (each lo load from local memory shared by twice the rows) past 8 tokens
    case 9: return w.T <= 8 ? read_n<Form<S48, S48, true>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq)
                            : read_n<Form<Shape<8, 8, true>, Shape<4, 16, true>, true>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    default: return w.T <= 8 ? read_n<Form<S48, S48, true>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq)
                             : read_n<Form<Shape<8, 8, true>, Shape<4, 8, true>, true>>(q, w, scratch, parts, stamp_buf, stamp_i0, seq);
    }
}

}  // namespace strata::kernels::grw
