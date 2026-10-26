// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/mmvq_slm.hpp - STRATA_MMVQ_SLM: the quantized GEMV of a window's columns with the activations staged
// in local memory, shared by native_mmvq.cpp (its traits) and iq_kernels.cpp (its formats).
//
// The established kernels read each column's q8_1 activations from global memory once a row (or two): a lane takes
// the 32-byte quant run of its own 36-byte block, so one sub-group load of a word touches some 18 cache lines for 128
// useful bytes, and at 16 columns those gathers, not the weights, set the time (the B70's ssm_out at 16 columns
// 117-177 us against a 17 us weight floor).  Here a work-group of SGS sub-groups, R rows each, copies the activations
// of a chunk of the row once into local memory with whole-line loads, and every lane reads its runs from there.
//
// The arithmetic is the established kernels' to the bit.  A row's sum is a tree fixed by their lane partition: VSG
// sub-groups of 32 lanes (4: native_mmvq's work-group layout, the 6144-wide rows; 1: its sub-group layout and
// iq_mmvq's), lane v taking blocks v / T, v / T + 32 VSG / T, ... (T lanes a block) and adding its terms in that
// order from zero; the work-group layout's first sub-group then adds the other three's partial sums in order, and an
// xor butterfly sums the 32 lanes.  One sub-group here plays all VSG of a row in turn: a "round" is the 32 / T blocks
// one virtual sub-group takes in one step, the rounds go virtual sub-group by virtual sub-group, and each one's
// partial sum (p) is folded into the row's running sum (q) as the first sub-group folded it.  The terms are the
// traits' own load and apply on the same values, so every output is the established call's bits.
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::kernels::slm {

constexpr int kWarp = 32;
constexpr int kCols = 16;    // kVerifyMaxT: the window's columns, one pass
constexpr int kQ81Dw = 9;    // a q8_1 block's 36 bytes in 32-bit words

// the established kernels' xor butterfly (native_mmvq's warp_sum, iq_kernels' warp_sum: the same order)
inline float sg_sum(const sycl::sub_group& sg, float v) {
    for (int o = kWarp / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}

struct Args {
    const uint8_t* w;      // the matrix (row_bytes a row, or the traits' own row layout)
    const uint8_t* w2;     // a second matrix of the same type and shape (rows n_out .. 2 n_out - 1), or null
    const uint32_t* x;     // the q8_1 activations, x_dw words a column
    float* y;
    float* y2;
    size_t row_bytes;
    int n_out, rows_all, ncols, bpr, x_dw;
    int rpc;               // rounds a chunk (the local memory holds rpc rounds of every column)
    int rounds;            // rounds a row, over all virtual sub-groups
    int sgs;               // sub-groups a work-group
    bool vec4;             // the activations and their strides 16-byte aligned: the copy in 16-byte words
};

// rounds virtual sub-group l of VSG takes: blocks l W + i STEP + (0 .. W - 1) for i = 0 .. rounds_of - 1
template <int W, int STEP>
inline int rounds_of(int bpr, int l) { return l * W < bpr ? (bpr - l * W + STEP - 1) / STEP : 0; }
template <int W, int STEP, int VSG>
inline int rounds_all(int bpr) {
    int n = 0;
    for (int l = 0; l < VSG; ++l) n += rounds_of<W, STEP>(bpr, l);
    return n;
}

// A: T (lanes a block), KBY (q8_1 blocks a weight block), W (a lane's weight values), load(w, row_bytes, row, kbx,
// bpr, lane) and dot<R>(wv, x, lane, acc) adding each row's term for the block whose first q8_1 block is x.
template <class A, int VSG, int R, bool Large>
struct Kernel {
    Args a;
    sycl::local_accessor<uint32_t, 1> sx;

    static_assert(kWarp % A::T == 0, "a block's lanes must divide the sub-group");
    static constexpr int W = kWarp / A::T;             // blocks a round
    static constexpr int STEP = VSG * W;               // a virtual sub-group's stride in blocks
    static constexpr int RQ = W * A::KBY * kQ81Dw;     // a round's words a column

    // the large register file (256 registers) for the window's accumulators; without it (the OpenCL CPU device of
    // the parity tests, which refuses the option, or STRATA_MMVQ_SLM_GRF=128) IGC's default
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        namespace sx_ = sycl::ext::oneapi::experimental;
        if constexpr (Large) return sx_::properties{sycl::ext::intel::experimental::grf_size<256>, sx_::sub_group_size<kWarp>};
        else return sx_::properties{sx_::sub_group_size<kWarp>};
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int lid = (int) it.get_local_linear_id(), wgs = (int) it.get_local_range(0);
        const int row0 = ((int) it.get_group(0) * a.sgs + (int) sg.get_group_linear_id()) * R;
        const int cs = a.rpc * RQ;                      // a column's words in local memory
        uint32_t* s = sx.template get_multi_ptr<sycl::access::decorated::no>().get();
        // R rows: each row's matrix, row and output (the second matrix's past n_out); a row past the end repeats a
        // row inside, computed and never written, so the sub-group stays whole through the barriers
        const uint8_t* wsrc[R];
        float* ydst[R];
        int rr[R];
        #pragma unroll
        for (int k = 0; k < R; ++k) {
            const int row = row0 + k < a.rows_all ? row0 + k : (row0 < a.rows_all ? row0 : 0);
            const bool second = row >= a.n_out;
            wsrc[k] = second ? a.w2 : a.w;
            ydst[k] = second ? a.y2 : a.y;
            rr[k] = second ? row - a.n_out : row;
        }
        // q: the row's running sum over the virtual sub-groups done; p: the current one's (one virtual sub-group: p
        // alone, the sum itself)
        float q[VSG > 1 ? kCols : 1][R], p[kCols][R];
        #pragma unroll
        for (int j = 0; j < kCols; ++j)
            #pragma unroll
            for (int k = 0; k < R; ++k) {
                p[j][k] = 0.0f;
                if constexpr (VSG > 1) q[j][k] = 0.0f;
            }
        int l = 0, i = 0, n_l = rounds_of<W, STEP>(a.bpr, 0);   // the next round to compute: (l, i) of n_l
        int cl = 0, ci = 0;                                      // the next round to copy
        for (int r0 = 0; r0 < a.rounds; r0 += a.rpc) {
            const int nr = std::min(a.rpc, a.rounds - r0);
            for (int t = 0; t < nr; ++t) {
                const int b0 = cl * W + ci * STEP;
                const int lim = std::min(RQ, (a.bpr - b0) * A::KBY * kQ81Dw);   // the round's words inside the row
                const size_t g0 = (size_t) b0 * A::KBY * kQ81Dw;
                if (a.vec4 && lim % 4 == 0) {
                    const int lim4 = lim / 4, n = a.ncols * lim4;
                    int c = lid / lim4, d = lid - c * lim4;
                    const int dc = wgs / lim4, dd = wgs - dc * lim4;
                    for (int e = lid; e < n; e += wgs) {
                        const auto v = *reinterpret_cast<const sycl::uint4*>(a.x + (size_t) c * a.x_dw + g0 + 4 * d);
                        *reinterpret_cast<sycl::uint4*>(s + c * cs + t * RQ + 4 * d) = v;
                        c += dc;
                        d += dd;
                        if (d >= lim4) { d -= lim4; ++c; }
                    }
                } else {
                    for (int c = 0; c < a.ncols; ++c)
                        for (int d = lid; d < lim; d += wgs) s[c * cs + t * RQ + d] = a.x[(size_t) c * a.x_dw + g0 + d];
                }
                if (++ci == rounds_of<W, STEP>(a.bpr, cl)) { ci = 0; ++cl; }
            }
            sycl::group_barrier(it.get_group());
            for (int t = 0; t < nr; ++t) {
                const int kbx = l * W + i * STEP + lane / A::T;
                if (kbx < a.bpr) {
                    typename A::W wv[R];
                    #pragma unroll
                    for (int k = 0; k < R; ++k) wv[k] = A::load(wsrc[k], a.row_bytes, rr[k], kbx, a.bpr, lane);
                    const uint32_t* xr = s + t * RQ + (lane / A::T) * A::KBY * kQ81Dw;
                    #pragma unroll
                    for (int j = 0; j < kCols; ++j)
                        if (j < a.ncols) A::template dot<R>(wv, xr + j * cs, lane, p[j]);
                }
                if (++i == n_l) {   // virtual sub-group l done: fold its partial sums as the first sub-group did
                    if constexpr (VSG > 1) {
                        #pragma unroll
                        for (int j = 0; j < kCols; ++j)
                            #pragma unroll
                            for (int k = 0; k < R; ++k) {
                                q[j][k] = l == 0 ? p[j][k] : q[j][k] + p[j][k];
                                p[j][k] = 0.0f;
                            }
                    }
                    i = 0;
                    n_l = rounds_of<W, STEP>(a.bpr, ++l);
                }
            }
            sycl::group_barrier(it.get_group());
        }
        if constexpr (VSG > 1)
            for (; l < VSG; ++l)   // virtual sub-groups past the row's end: their zero sums, added as the first one did
                #pragma unroll
                for (int j = 0; j < kCols; ++j)
                    #pragma unroll
                    for (int k = 0; k < R; ++k) q[j][k] = l == 0 ? 0.0f : q[j][k] + 0.0f;
        #pragma unroll
        for (int k = 0; k < R; ++k) {
            if (row0 + k >= a.rows_all) break;
            #pragma unroll
            for (int j = 0; j < kCols; ++j) {
                if (j >= a.ncols) break;
                float v;
                if constexpr (VSG > 1) v = sg_sum(sg, q[j][k]);
                else v = sg_sum(sg, p[j][k]);
                if (lane == 0) ydst[k][(size_t) j * a.n_out + rr[k]] = v;
            }
        }
    }
};

// The work-group layout's sizes and the launch.  sgs sub-groups of R rows a work-group, rpc rounds a chunk from the
// local-memory budget (kb KiB, at least one round, at most the device's).  False, launching nothing, where the
// activations are not 4-byte aligned or one round of every column does not fit.
template <class A, int VSG, int R, bool Large>
bool launch_grf(sycl::queue& q, const void* w, const void* w2, const void* x, float* y, float* y2, size_t row_bytes,
                int n_in, int n_out, int ncols, int sgs, int kb, size_t local_bytes, sycl::event* ev) {
    using K = Kernel<A, VSG, R, Large>;
    if (ncols < 1 || ncols > kCols || n_in % A::DIV != 0 || reinterpret_cast<uintptr_t>(x) % 4 != 0) return false;
    Args a{};
    a.w = static_cast<const uint8_t*>(w);
    a.w2 = static_cast<const uint8_t*>(w2);
    a.x = static_cast<const uint32_t*>(x);
    a.y = y;
    a.y2 = y2;
    a.row_bytes = row_bytes;
    a.n_out = n_out;
    a.rows_all = w2 ? 2 * n_out : n_out;
    a.ncols = ncols;
    a.bpr = n_in / A::DIV;
    a.x_dw = n_in / 32 * kQ81Dw;
    a.rounds = rounds_all<K::W, K::STEP, VSG>(a.bpr);
    a.sgs = sgs;
    a.vec4 = reinterpret_cast<uintptr_t>(x) % 16 == 0 && a.x_dw % 4 == 0 && K::RQ % 4 == 0 && (K::W * A::KBY) % 4 == 0;
    const size_t round_bytes = (size_t) ncols * K::RQ * 4;
    const size_t budget = std::min((size_t) kb * 1024, local_bytes);
    if (round_bytes > budget) return false;
    a.rpc = (int) std::min<size_t>(budget / round_bytes, (size_t) a.rounds);
    const size_t groups = ((size_t) a.rows_all + (size_t) sgs * R - 1) / ((size_t) sgs * R);
    const sycl::event e = q.submit([&](sycl::handler& h) {
        K k{a, sycl::local_accessor<uint32_t, 1>(sycl::range<1>((size_t) ncols * a.rpc * K::RQ), h)};
        h.parallel_for(sycl::nd_range<1>(groups * sgs * kWarp, (size_t) sgs * kWarp), k);
    });
    if (ev) *ev = e;
    return true;
}
// large: the large register file where the device takes it (a GPU)
template <class A, int VSG, int R>
bool launch(sycl::queue& q, const void* w, const void* w2, const void* x, float* y, float* y2, size_t row_bytes,
            int n_in, int n_out, int ncols, int sgs, int kb, size_t local_bytes, bool large, sycl::event* ev = nullptr) {
    static const bool gpu = q.get_device().is_gpu();
    if (large && gpu)
        return launch_grf<A, VSG, R, true>(q, w, w2, x, y, y2, row_bytes, n_in, n_out, ncols, sgs, kb, local_bytes, ev);
    return launch_grf<A, VSG, R, false>(q, w, w2, x, y, y2, row_bytes, n_in, n_out, ncols, sgs, kb, local_bytes, ev);
}

}  // namespace strata::kernels::slm
