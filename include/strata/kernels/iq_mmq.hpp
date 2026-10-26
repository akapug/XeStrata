// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/iq_mmq.hpp - the prompt path's expert products on the int8 matrix engines, the weights read
// in their GGUF blocks (llama.cpp's MMQ, written anew on joint_matrix): the activations as int8 with a scale for each
// 32 values, the weights decoded in local memory to int8 with the scale of their 32 (or 16) values, and each such
// block's integer sum scaled by both into FP32.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Whether `q`'s device runs these products on its matrix engines: it reports a tile shape they are written for (int8
/// inputs, int32 accumulators: NVIDIA's 16 x 16 x 16 on 32 lanes, Intel's XMX 8 x 16 x 32 on 16 or 8 x 8 x 32 on 8) at
/// its sub-group size, or it is an RDNA3 or RDNA4 GPU (their WMMA); and its accumulators hold their elements where the
/// products read them (checked on the device once).  STRATA_NO_XMX=1 says no.
bool iq_mmq_usable(sycl::queue& q);
/// Whether the same products run on `q`'s device without matrix engines (dp4a; local memory and work-group size
/// permitting): iq_mmq_grouped takes that form where iq_mmq_usable says no.
bool iq_mmq_dp4a_usable(sycl::queue& q);
/// What runs these products on `q`'s device, for the start's report: "WMMA" (RDNA3, RDNA4), "joint_matrix" (a shape
/// the device reports: NVIDIA's tensor cores), "XMX" (Intel's shapes), "DP4a" (no matrix engines), or null (neither).
const char* iq_mmq_path(sycl::queue& q);
/// The ggml types whose weights these products decode (the i-quants but IQ1_S, Q2_0, Q8_0, Q5_0, Q2_K to Q6_K).
bool iq_mmq_type_ok(int ggml_type);
/// Bytes of a weight row of `cols` values in ggml_type (iq_mmq_type_ok).
size_t iq_mmq_row_bytes(int ggml_type, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values (a multiple of 32): the int8 values, then a scale per 32, then the
/// sum per 32 (the scale times the int8 values' sum).
size_t iq_mmq_act_bytes(int64_t rows, int64_t cols);
/// Whether products of ggml_type read the activations' sums (the K-quants with a min term).
bool iq_mmq_type_sums(int ggml_type);
/// Whether products of ggml_type read rows of twice the weights' columns, every activation block twice (Q2_K: its
/// per-16 mins as a second block of ones): iq_mmq_quantize's `dup`, and iq_mmq_grouped's w_cols doubled.
bool iq_mmq_type_dup(int ggml_type);
/// The activations: row i of xq from row ids[i] of x (row i when ids is null), `ld` floats a row of x; `sums`: their
/// block sums too (iq_mmq_type_sums), else that part is left unwritten; `dup`: each block twice (rows of 2 cols).
sycl::event iq_mmq_quantize(sycl::queue& q, const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld,
                            int64_t rows, bool sums, bool dup = false);
/// iq_mmq_quantize of the SwiGLU silu(gate) * up, `rows` rows of `cols` values each (homelab K1: the shared expert's
/// down product on iq_mmq_decode)
sycl::event iq_mmq_swiglu_quantize(sycl::queue& q, const float* gate, const float* up, void* xq, int64_t cols,
                                   int64_t rows, bool sums);
/// The rows iq_mmq_quantize(x, src, ...) writes for `tokens` x `k_used` rows, written token by token: token t's row of
/// x is quantized once and its blocks stored at rows slot[t * k_used + k] (`slot` the inverse of `src`): the same bytes,
/// a k_used-th of the quantizations (upstream c675f5b7, llama.cpp's quantize_scatter_mmq_q8_1).
sycl::event iq_mmq_quantize_scatter(sycl::queue& q, const float* x, const int32_t* slot, void* xq, int64_t cols,
                                    int64_t ld, int64_t tokens, int k_used, bool sums, bool dup = false);

/// n experts, one launch: expert e's weights ([w_rows, w_cols] in ggml_type's blocks) at w + e * expert_bytes, its
/// activation rows bounds[e] .. bounds[e + 1] of xq (bounds on the device; xq holds total_rows rows), into dst rows
/// ids[row] (ld_dst floats apart).  max_rows: the most rows an expert has.  w_rows a multiple of 128, w_cols of 32
/// (and of the type's block).
/// wtab (on the device, n entries): expert e's weights at wtab[e] + woff instead, wherever each lies.
/// rows: the n experts' rows together (0: total_rows), which with max_rows picks the tile's shape.
sycl::event iq_mmq_grouped(sycl::queue& q, int ggml_type, const void* w, size_t expert_bytes, int64_t w_rows,
                           int64_t w_cols, int n, const void* xq, int64_t total_rows, const int32_t* bounds,
                           int64_t max_rows, const int32_t* ids, float* dst, int64_t ld_dst,
                           const uint8_t* const* wtab = nullptr, size_t woff = 0, int64_t rows = 0);

/// Default-off STRATA_DMMQ_NATIVE_Q8_1: dense/shared decode may reuse native quantized operands. Prompt MMQ and HC
/// never consult it. Quantization and reductions remain distinct quality questions; this is not an output parity claim.
bool iq_mmq_decode_native_q8_1();
/// Adapt caller-owned native_quantize_q8_1/native_swiglu_q8_1 blocks to the existing planar decode layout. No allocation;
/// `native` and `xq` must not overlap, rows in 1..16, cols a positive multiple of 32. The FP16-stored scale is expanded
/// to FP32; sums (Q4_K/Q5_K) are that scale times the integer codes' sum, NOT native's raw-activation half sum.
sycl::event iq_mmq_decode_from_q8_1(sycl::queue& q, const void* native, void* xq, int64_t cols, int64_t rows, bool sums);

/// homelab K1 (STRATA_BATCH_DENSE_XMX): a decode window's dense product dst[r, n] = xq[r] . W[n] for 1..16 rows, on
/// Intel's Xe2 XMX (8 x 16 x 32 int8 on 16 lanes), each weight block decoded once for all the rows.  xq as
/// iq_mmq_quantize writes it for `rows` rows of w_cols; W [w_rows, w_cols] in the type's blocks (w_rows a multiple of
/// 16, w_cols a multiple of the type's complete weight block; row-split native layouts need even block counts).
/// The same quantized operands as iq_mmq_grouped, its sums in another order. Whether it runs for these:
bool iq_mmq_decode_ok(sycl::queue& q, int ggml_type, int64_t w_rows, int64_t w_cols, int64_t rows);
/// The K split (1..16; 0: the aim's) and the sub-groups aimed for, in place of STRATA_DMMQ_KS and STRATA_DMMQ_SGS (tests).
/// A negative value reloads that setting from the environment on the next launch.
void iq_mmq_decode_set_split(int ks, int sgs);
sycl::event iq_mmq_decode(sycl::queue& q, int ggml_type, const void* w, int64_t w_rows, int64_t w_cols, const void* xq,
                          int64_t rows, float* dst, int64_t ld_dst);

/// homelab densefuse, STRATA_DMMQ_LEAN's bits (dmmq_contract.hpp; 0 = off, the default), read once; the setter replaces
/// them (tests; a negative value reads the environment again on the next call).
int iq_mmq_decode_lean();
void iq_mmq_decode_set_lean(int bits);
/// iq_mmq_decode's products in LeanKernel's form, n = 1 or 2 matrices of one type and shape on one xq (w[i] into
/// dst[i], ld_dst apart), each with its own launch's K split: iq_mmq_decode's bits.  xmx: Intel's Xe2 tiles (where
/// iq_mmq_decode_ok); false: the same steps without matrix engines, on any device (the OpenCL CPU device's parity).
sycl::event iq_mmq_decode_lean_run(sycl::queue& q, int ggml_type, const void* const* w, int n, int64_t w_rows,
                                   int64_t w_cols, const void* xq, int64_t rows, float* const* dst, int64_t ld_dst,
                                   bool xmx);
/// STRATA_DMMQ_LEAN's pair bit: two products of one type and shape on one xq (the shared expert's gate and up) in one
/// launch, each iq_mmq_decode's bits; false (nothing launched) where the bit is off or iq_mmq_decode does not run.
bool iq_mmq_decode_pair(sycl::queue& q, int ggml_type, const void* w0, const void* w1, int64_t w_rows, int64_t w_cols,
                        const void* xq, int64_t rows, float* dst0, float* dst1, int64_t ld_dst);
/// The K split iq_mmq_decode takes for these (its slices' sum order).
int iq_mmq_decode_split(int64_t w_rows, int64_t w_cols);
/// iq_mmq_decode's arithmetic on the host (tests): the same decoded blocks, int32 block sums, scale order and slices.
void iq_mmq_decode_reference(int ggml_type, const void* w, int64_t w_rows, int64_t w_cols, const void* xq, int64_t rows,
                             float* dst, int64_t ld_dst);

}  // namespace strata::kernels

