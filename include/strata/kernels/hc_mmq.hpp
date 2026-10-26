// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/hc_mmq.hpp - a batch window's hyper-connection read with its two projections on the matrix
// engines: fused_gr_read_multi's arithmetic for n rows at once, split at the products.
//
//   norm   : R' = R + bo_prev * 2 sigmoid(inj_prev / hc), written back in place (when `apply`); rs, xn = R' w_norm rs
//            (fused_gr_read_multi's split norm, bit for bit), and xn as the down product's input (BF16, or iq_mmq's
//            int8 activations)
//   down   : lo = xn . w_down^T (xmx_gemm, or iq_mmq_grouped / opt-in decode on the native Q8_0 rows)
//   mid    : inject = w_inject . xn (the multi read's sub-group sums, bit for bit), lo = silu(lo / hc), and lo as the
//            up product's input
//   up     : u = lo . w_up^T
//   mix    : mixed[d] = mean_c xn[c, d] sigmoid(u[c, d]), and mixed's q8_1 (as the read's 32-column up writes it)
//
// rs, xn, R' and inject are the fused read's; lo, mixed and the q8_1 differ by the products' rounding (BF16 or int8
// inputs to the products instead of FP32 against the weights).
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/dmmq_dispatch.hpp"

namespace strata::kernels {

/// Device scratch for up to `rows` rows (each buffer 64-byte aligned).
struct HcMmqScratch {
    uint16_t* xb = nullptr;           ///< rows * hc * n_embd: xn in BF16 (the BF16 down product's input)
    uint16_t* lb = nullptr;           ///< rows * hc_lr: lo in BF16 (the BF16 up product's input)
    float* up = nullptr;              ///< rows * hc * n_embd: the up product
    uint8_t* xq = nullptr;            ///< hc_mmq_xq_bytes(rows): xn, then lo, as iq_mmq's int8 activations
    const int32_t* bounds = nullptr;  ///< {0, n} for this window's n, on the device
    const int32_t* ids = nullptr;     ///< 0 .. rows - 1, on the device
};

/// Bytes of HcMmqScratch::xq for `rows` rows at the read's geometry.
size_t hc_mmq_xq_bytes(int64_t rows);
/// Whether the device runs the read's products on matrix engines: the Q8_0 form (`q8`: iq_mmq) or the BF16 one
/// (xmx_gemm, not its DP4a fallback).  Probes the device: call before any graph is recorded.
bool hc_mmq_engine(bool q8);
/// Whether fused_gr_read_mmq takes the read of `w`'s weights (the Q8_0 rows when q8_down and q8_up are set, else the
/// BF16 ones) into `lo` with scratch `s`: the geometry, and the products' shape and alignment rules.  Host only.
bool hc_mmq_weights_ok(const FusedGrArgs& w, const float* lo, const HcMmqScratch& s) noexcept;
/// Default-off row opt-ins shared by production and the roll-up. The established MMQ setting is unchanged;
/// STRATA_BATCH_HC_XMX selects only native Q8 decode, floored by STRATA_DMMQ_MIN_ROWS (default seven).
int hc_mmq_min();
int hc_xmx_min();
/// The production path selector, including both projection contracts, engine capability and scratch. Probe before
/// graph recording. An unsupported decode falls back to MMQ only if that separate opt-in permits it, else fused.
dmmq::HcPath hc_mmq_path(const FusedGrArgs& w, const float* lo, const HcMmqScratch& s, int rows);
/// fused_gr_read_multi(a, n, xn_scratch) with the products on the matrix engines.  The rows' buffers must be
/// contiguous (a[t]'s R, bo_prev, inj_prev, lo, rs, inject_out, mixed, q8_mixed at a[0]'s plus t of their size),
/// R_out == R, and apply, eps and the weights the same for every row.  On `stream`; no host sync.
void fused_gr_read_mmq(const FusedGrArgs* a, int n, float* xn_scratch, const HcMmqScratch& s, void* stream,
                       unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);

}  // namespace strata::kernels
