// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/hc_xmx.hpp - homelab "hcx": a batch window's Q8_0 hyper-connection read with both projections
// as int8 products on Xe2's XMX, in three launches, the norm folded into the down product.
//
// The read (fused_gr.hpp) for rows t of a window, stream c, column d, k over the hc * n_embd inputs:
//   R'      = R + bo_prev * 2 sigmoid(inj_prev[c] / hc)                (when apply; written back in place)
//   rs[c]   = rsqrt(mean_d R'[c, d]^2 + eps),  y = R' * w_norm,  xn = y * rs[c]
//   lo      = silu((sum_c rs[c] sum_{k in c} w_down[r, k] y[k]) / hc)
//   inject  = sum_c rs[c] sum_{k in c} w_inject[j, k] y[k]             (FP32, as the established read)
//   u       = w_up . lo,  mixed[d] = mean_c xn[c, d] sigmoid(u[c, d]),  mixed's q8_1
//
// What is new: rs is a scale per stream, so the down product is taken on y (before the norm) and rs is applied to the
// per-stream partial sums afterwards; the norm, the activations' quantization and the down product are then one pass
// over R.  The activations of both products are int8 blocks of 32 (scale amax / 127) in one or two slices: slice two
// holds the first slice's rounding residual at 1/128 of its scale, so a block's product is the exact int32
// 128 * (q1 . w) + (q2 . w), scaled once.  The int32 sums are exact whatever engine adds them (XMX, or the portable
// body's dp4a); everything after them is the same float code in the same order, so the XMX and portable bodies give
// the same bits.  The error against the FP32 read is the activations' rounding, at most amax / 254 an element with one
// slice and amax / 32512 with two (hcx_parity derives the per-output bound and checks every output against it).
//
//   down : a work-group a chunk of kHcxKch inputs (one stream's), all hc_lr outputs + the inject rows: R' (written
//          back), the chunk's sums of squares and inject partials (FP32), y quantized into local memory, then a
//          sub-group 16 outputs: the chunk's int8 products -> partial sums (scratch)
//   lo   : a work-group a row: rs, lo = silu(...), the inject; lo quantized (scratch)
//   up   : a work-group kHcxUpCols columns of every stream (a q8_1 block): the int8 products, then mixed and its q8_1
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/window_capacity.hpp"

namespace strata::kernels {

// the read's geometry (fused_gr.cpp's; the probe checks the weights against it)
constexpr int kHcxN = 2560, kHcxHC = 4, kHcxD = kHcxN * kHcxHC, kHcxLR = 320;
constexpr int kHcxBlock = 32;                        ///< an activation block (and a Q8_0 weight block)
constexpr int kHcxKch = 320;                         ///< inputs a down work-group (32 work-groups)
constexpr int kHcxChunks = kHcxD / kHcxKch;          ///< partial sums per output
constexpr int kHcxPR = kHcxLR + kHcxHC;              ///< partial rows: the down outputs, then the inject rows
constexpr int kHcxUpCols = 32;                       ///< columns an up work-group (one q8_1 block of mixed)
constexpr int kHcxMaxRows = kVerifyMaxT;             ///< rows a call
constexpr int kHcxMaxSlices = 2;
// the settings' defaults: STRATA_BATCH_HCX (rows from which a batch window's reads take hcx; 0 = off) and
// STRATA_BATCH_HCX_SLICES (1 or 2 activation slices)
constexpr int kHcxMinDefault = 0;
constexpr int kHcxSlicesDefault = 2;

/// Which body computes the int32 block sums: the XMX one where the device has Xe2's 8 x 16 x 32 int8 tiles
/// (`automatic` takes it there, else the portable one), or one of them by name (tests).
enum class HcxBody { automatic, xmx, portable };

/// hcx's scratch for up to `rows` rows (hcx_scratch_bytes), carved by hcx_carve: the down partial sums, the chunks'
/// sums of squares, lo's int8 slices and scales; `u` (tests only, null in production): the up product, rows * D.
struct HcxScratch {
    float* part = nullptr;   ///< [row][chunk][kHcxPR]
    float* ssg = nullptr;    ///< [row][chunk]
    int8_t* lq = nullptr;    ///< [slice][row][kHcxLR]
    float* lsc = nullptr;    ///< [row][kHcxLR / 32]: the scale of a block's int32 sum (slice two: already / 128)
    float* u = nullptr;      ///< tests: [row][kHcxD]
};
size_t hcx_scratch_bytes(int rows);
HcxScratch hcx_carve(void* base, int rows);

/// STRATA_BATCH_HCX (default kHcxMinDefault, off) and STRATA_BATCH_HCX_SLICES (default kHcxSlicesDefault), read once.
int hcx_min();
int hcx_slices();
/// Whether the compute device runs the XMX body (Xe2's int8 tiles and their layout check, iq_mmq_decode_ok's probe).
/// Probes the device: call before any graph is recorded.
bool hcx_engine();
/// Whether `a[0..n)` is a window hcx takes: n in 1..kHcxMaxRows, the Q8_0 projections (aligned), R_out == R, every
/// row's buffers at a[0]'s plus the row times their size, apply, eps and the weights the same.  Host only.
bool hcx_window_ok(const FusedGrArgs* a, int n) noexcept;
/// The production choice: STRATA_BATCH_HCX on, n at least its rows, the engine, and the window.
bool hcx_take(const FusedGrArgs* a, int n);
/// fused_gr_read_multi(a, n, ...)'s outputs (R', rs, lo, inject_out, mixed, q8_mixed) the hcx way, `slices` 1 or 2.
/// On `stream`; no host sync.  Throws on a window hcx_window_ok refuses or a body the device lacks.
void hcx_read(const FusedGrArgs* a, int n, const HcxScratch& s, void* stream, int slices = kHcxSlicesDefault,
              HcxBody body = HcxBody::automatic, unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);

}  // namespace strata::kernels
