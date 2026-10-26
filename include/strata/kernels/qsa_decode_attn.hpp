// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/qsa_decode_attn.hpp - plan v0.3 P3/P7: split-K decode attention over the selected cells.
//
// The first attention kernel runs one block per query head (24 blocks on a 48-SM part), reads each key row with one
// thread (uncoalesced) and walks every selected cell serially for the values; at a 4K context (2,051 selected
// cells) that costs ~0.3-0.5 ms per QSA layer.  This one reads the KV POOLS directly through the page table and the
// selection ids (no gather copy), in chunks of CHUNK cells per block, and serves all `n_head / n_head_kv` query
// heads that share a KV head from one read of the chunk:
//
//     grid (ceil(cap / CHUNK), n_head_kv):  s = q.k * scale for 12 heads x CHUNK cells, chunk max m and sum l,
//                                           acc = p . V  ->  partials
//     grid (n_head):                        merge the chunks with the usual log-sum-exp rescale
//
// FP16 pools or INT8 pools (codes + fp16 scale per 64 values, `kv_q8.hpp`).  The grid is sized by the capacity;
// the real count comes from `step[kStepWidth]` as for every other capturable QSA kernel.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

struct QsaAttnPools {
    const uint16_t* k_pool = nullptr;   ///< fp16 [page][kv_head][page_size][head_dim], or null when int8
    const uint16_t* v_pool = nullptr;
    const int8_t* k_q = nullptr;        ///< int8 codes, same layout
    const int8_t* v_q = nullptr;
    const uint16_t* k_scale = nullptr;  ///< fp16 [page][kv_head][page_size][head_dim / 64]
    const uint16_t* v_scale = nullptr;
    const uint8_t* k_q4 = nullptr;      ///< q4_0 block_q4_0 [page][kv_head][page_size][head_dim / 32 * 18]
    const uint8_t* v_q4 = nullptr;
    const int32_t* page_table = nullptr;
};

/// Scratch floats for `cap` selected cells: partial accumulators, maxima and sums.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream);

/// Plan v0.3 P5: `n_q` queries at once, each with its own selection: q [n_q, n_head, 256], ids [n_q, cap], steps
/// [n_q, kStepCount], attn [n_q, n_head, 256]; scratch is `n_q` times the single-query size.
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

/// qsa_decode_attn_batch for a batch window's rows, each over its own slot's K/V (STRATA_BATCH_QSA_ROWS): row r's
/// pools from the table, everything else as qsa_decode_attn_batch (q, ids, steps, scratch and attn at row strides).
/// One format for all rows (`mode`, set by qsa_attn_slots_set).  Row r's arithmetic is the one-query call's.
struct QsaAttnSlots {
    const void* k[kQsaSlotsMax] = {};          ///< fp16 pool, int8 codes or q4_0 blocks, per `mode`
    const void* v[kQsaSlotsMax] = {};
    const uint16_t* k_scale[kQsaSlotsMax] = {};
    const uint16_t* v_scale[kQsaSlotsMax] = {};
    const int32_t* page_table[kQsaSlotsMax] = {};
    int mode = -1;                             ///< 0 fp16, 1 int8, 2 q4_0, 3 K8V4; -1 unset
};
/// Row r's entry from its pools; false if their format is not the rows' before it (the caller keeps the per-row calls).
bool qsa_attn_slots_set(QsaAttnSlots& tab, int r, const QsaAttnPools& pools);
void qsa_decode_attn_slots(const float* q, const QsaAttnSlots& tab, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

/// STRATA_QSA_ATTN_LANES (default off, homelab): the chunk pass in 16-lane sub-groups, a cell a lane in the score pass
/// (no local-memory reads of the query or the weights in the loops), for head_dim 256 with FP16 or INT8 K/V; every
/// output is the default kernel's bit for bit.  For tests, before graph capture: 1 on, 0 off, -1 the setting again.
void qsa_decode_attn_set_lanes(int on);

/// STRATA_QSA_ATTN_FAST (default off, homelab; ahead of STRATA_QSA_ATTN_LANES): the lanes form with fma score chains and
/// a work-group walking several chunks with an online softmax, so fewer partials go through memory; the same answer to
/// float rounding, not the same bits.  STRATA_QSA_ATTN_FAST_GROUPS sets the work-groups a call it aims for (512).
/// For tests, before graph capture: on 1/0/-1 as qsa_decode_attn_set_lanes; span > 0 forces the chunks a work-group.
void qsa_decode_attn_set_fast(int on, int span);
/// Both forms run only for calls of STRATA_QSA_ATTN_LANES_MIN_ROWS rows or more (4); fewer take the default kernel.

/// The same kernels over EVERY cell of the cache, cells [0, step[kStepNKv]), with no selection: the full attention of
/// qwen35moe (Ornith-1.5: 16/2 heads of 256) and qwen3moe (llm-jp-4.1: 40/4 heads of 128).  `max_cells` is the
/// cache's capacity and sizes the grid (`qsa_decode_attn_scratch_floats(max_cells, s)` scratch floats per query);
/// a grid that followed the context would be baked into a captured graph.  The shapes the kernels are built for are
/// (head_dim 256, 12 or 8 query heads per KV head) and (128, 10); any other throws, naming the shape.
void qsa_dense_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* step, int64_t max_cells,
                         const QsaShapes& s, float* scratch, float* attn, void* stream);
void qsa_dense_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* steps, int64_t max_cells,
                          const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

}  // namespace strata::kernels
