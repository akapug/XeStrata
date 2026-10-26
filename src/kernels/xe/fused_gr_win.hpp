// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/fused_gr_win.hpp - STRATA_HC_Q8_WIN: the Q8_0 hyper-connection read of a whole window (up to
// kVerifyMaxT tokens) in one pass, its arguments in a compact form.
//
// fused_gr_read_multi passes up to kFusedGrMaxT FusedGrArgs by value (~136 bytes each); 16 of them would pass the
// device's kernel-parameter limit.  In a window the tokens' buffers are a base plus the token times a stride (verify's
// gr_read_group), so GrWin keeps each token's R and R_out and one base and byte stride for the rest: 376 bytes for 16.
#pragma once

#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace strata::kernels::grw {

// fused_gr.cpp's geometry and its Q8_0 read's layout (it asserts they agree)
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, PR = LR + HC;
constexpr int THREADS = 256, WARP = 32, WARPS = THREADS / WARP;
constexpr int Q8_RPW = 4, Q8_RG = LR / (WARPS * Q8_RPW), Q8_KC = 640, Q8_NKC = D / Q8_KC, Q8_CPS = N / Q8_KC;
constexpr int Q8_SPB = Q8_KC / 128;
constexpr size_t Q8_ROW_D = (size_t) D / 32 * 34, Q8_ROW_U = (size_t) LR / 32 * 34;

struct GrWin {
    // A kernel tile, not allocation capacity: retain the established 376-byte ABI
    // and <=16 specializations. Larger windows are tiled by fused_gr_read_multi.
    static constexpr int kMax = 16;
    const float* R[kMax];
    float* R_out[kMax];
    // token k's buffer = base + k * stride (bytes)
    const char* bo;
    const char* inj;
    char* rs;
    char* io;
    char* mx;
    char* q8m;
    int bo_s, inj_s, rs_s, io_s, mx_s, q8m_s;
    const float* w_norm;
    const uint16_t* w_inject;
    const uint8_t* q8_down;
    const uint8_t* q8_up;
    float eps;
    uint32_t apply;   ///< bit k: token k folds its previous half's write
    int T;

    bool applies(int k) const { return ((apply >> k) & 1u) != 0; }
    const float* bo_prev(int k) const { return reinterpret_cast<const float*>(bo + (ptrdiff_t) k * bo_s); }
    const float* inj_prev(int k) const { return reinterpret_cast<const float*>(inj + (ptrdiff_t) k * inj_s); }
    float* rs_out(int k) const { return reinterpret_cast<float*>(rs + (ptrdiff_t) k * rs_s); }
    float* inject_out(int k) const { return reinterpret_cast<float*>(io + (ptrdiff_t) k * io_s); }
    float* mixed(int k) const { return reinterpret_cast<float*>(mx + (ptrdiff_t) k * mx_s); }
    void* q8_mixed(int k) const { return q8m == nullptr ? nullptr : q8m + (ptrdiff_t) k * q8m_s; }
    /// token k's arguments as the established kernels read them (no lo: the Q8_0 read keeps it in local memory)
    FusedGrArgs get(int k) const {
        FusedGrArgs a;
        a.R = R[k]; a.R_out = R_out[k]; a.apply = applies(k); a.bo_prev = bo_prev(k); a.inj_prev = inj_prev(k);
        a.w_norm = w_norm; a.w_inject = w_inject; a.eps = eps; a.rs = rs_out(k); a.inject_out = inject_out(k);
        a.mixed = mixed(k); a.q8_down = q8_down; a.q8_up = q8_up; a.q8_mixed = q8_mixed(k);
        return a;
    }
};

/// `a[0..n)` (already validated as fused_gr_read_multi's) as a GrWin, or false where they are not a window's layout
/// (a buffer that is not base + token * stride, different eps): the caller keeps the established path.
bool pack(const FusedGrArgs* a, int n, GrWin& w);

/// STRATA_HC_Q8_WIN forms 2 and up: the rewritten kernels (fused_gr_win.cpp) for w.T tokens in form `shape` (form - 2),
/// a down shape and an up shape each (rows a sub-group, tokens a pass, register file: fused_gr_win.cpp's grw::read);
/// shape 0 (form 2) picks the fastest measured for w.T, the others are fixed (for hc_read16_parity); shapes 8..10
/// (forms 10..12, homelab) add up the lo sums in the down launch's tail (a launch fewer), 9 and 10 with 4-row up shapes;
/// `scratch` is fused_gr_read_multi's (w.T * D floats).  `parts` (tests): bit 1 the down launch, 2 the lo reduction, 4
/// the up launch.
constexpr int kShapes = 11;
sycl::event read(sycl::queue& q, const GrWin& w, float* scratch, int shape, int parts, unsigned long long* stamp_buf,
                 int stamp_i0);

}  // namespace strata::kernels::grw
