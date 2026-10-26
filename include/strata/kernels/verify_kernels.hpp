// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/verify_kernels.hpp - plan v0.3 P6: kernels for the speculative VERIFY window, where T
// tokens (the last accepted token and T-1 drafts) go through a layer at once.
//
// The GDN recurrence is the one piece of state that cannot simply be overwritten by the next window, so its
// kernels come in two halves:
//
//   * the verify half reads the conv history and the recurrent state, runs all T tokens in order (bitwise the
//     single-token kernels' arithmetic per token) and writes ONLY the per-token outputs - the state is left
//     untouched;
//   * the commit half, launched once acceptance is known, replays the first `n_keep` tokens from the inputs the
//     verify half stored and writes the state.  Rejected tokens therefore never touch the state and there is no
//     snapshot of the 113 MB recurrent state to keep.
//
// Everything that varies per window (token ids, `n_keep`) is read from DEVICE memory so the kernels can be captured.
#pragma once

#include <cstdint>
#include "strata/kernels/window_capacity.hpp"

namespace strata::kernels {

/// For token t of T: conv over [history(3) | qkv_0 .. qkv_t] -> SiLU -> L2 norm of the q/k heads -> h[t].
/// `history` is NOT written, except with `commit` (one token, t_begin 0: a one-token window keeps its token), which
/// leaves it as gdn_conv_commit with n_keep 1 would.  Bitwise `fused_gdn_conv_l2` per token.
void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin = 0, bool commit = false);
/// history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (0 leaves it as it was).
void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream);
/// alpha/beta for T columns of x (T, n_embd): gate (T, h_v), beta (T, h_v).  Bitwise `fused_gdn_ab` per column.
void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream);
/// Apply GDN's established softplus/sigmoid to raw alpha/beta projections in place.
void gdn_ab_activate(float* gate, float* beta, const float* dt, const float* ssm_a,
                     int h_v, int n_tok, void* stream);
/// gdn_ab_multi's arguments, for the conv launch that runs it too (STRATA_GDN_AB_FORM 4..6)
struct GdnAbArgs {
    const float* x = nullptr;
    const uint16_t* w_alpha = nullptr;
    const uint16_t* w_beta = nullptr;
    const float* dt = nullptr;
    const float* ssm_a = nullptr;
    float* gate = nullptr;
    float* beta = nullptr;
    int n_embd = 0, h_v = 0, n_tok = 0;
};
/// STRATA_GDN_AB_FORM (homelab, default 0): gdn_ab_multi over a sub-group a (row, TC tokens), the same bits; 1, 2, 3 its
/// own launch (TC 1, 2, 4); 4, 5, 6 the same, and the batch window's GDN layer runs it inside gdn_conv_l2_slots's
/// launch (gdn_ab_fused_in_conv).  Set before graph capture; for tests.
void gdn_ab_set_form(int form);
int gdn_ab_form();
bool gdn_ab_fused_in_conv();
/// The recurrence + output norm for T tokens (h = (T, conv_channels) as q|k|v, gate/beta (T, h_v), z/y
/// (T, value_dim)).  With `n_keep == nullptr` the state is read and NOT written (verify); otherwise the first
/// *n_keep tokens are run and the state is written (commit; `y` may be scratch).  Bitwise `fused_gdn_step_norm`.
void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin = 0, void* y_q8_1 = nullptr);
/// BATCH WINDOWS, ONE LAUNCH FOR EVERY SLOT (STRATA_BATCH_GDN_SLOTS).  A batch window's rows belong to different
/// slots, each with its own conv history and recurrent state; the per-slot loops in record_window launched these two
/// kernels once per slot and GDN layer (2 x 36 launches a row, each one work-group a head: 32 work-groups on a
/// 32-core card).  The `slots` forms take a table instead: slot group j covers window rows [first[j], first[j] + ntok[j])
/// and reads/writes `state[j]` / `hist[j]`.  Every row's arithmetic is the per-slot call's (the plain one-group-a-head
/// form; the three forms give the same bits), so the tokens are bitwise those of the loop.
struct GdnSlotTable {
    float* state[kVerifyMaxT] = {};        ///< slot group j's recurrent state (the layer's)
    float* hist[kVerifyMaxT] = {};         ///< slot group j's conv history (the layer's; written only with `commit`)
    int first[kVerifyMaxT] = {};           ///< its first window row
    int ntok[kVerifyMaxT] = {};            ///< its rows (1 without --batch-mtp)
    int n = 0;                             ///< slot groups
    /// With `commit`: device int32 per window row; group j commits its history and state only where keep[first[j]] is
    /// nonzero (a pad row for an idle slot, keep 0, leaves both as they were).  nullptr: every group commits.
    const int32_t* keep = nullptr;
};
/// gdn_conv_l2_multi for every slot group at once (rows [0, rows) of qkv/h, the window's).  `commit`: every group has
/// one row and keeps it (STRATA_BATCH_SELF_COMMIT): each history is left as gdn_conv_commit with n_keep 1 leaves it.
/// `ab` (STRATA_GDN_AB_FORM 4..6): gdn_ab_multi(*ab) in the same launch, its bits (its own work-groups, after the conv's).
void gdn_conv_l2_slots(const GdnSlotTable& tab, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, void* stream, bool commit = false, const GdnAbArgs* ab = nullptr);
/// STRATA_BATCH_GDN_FORM for tests: gdn_step_norm_slots's form (0 plain, 1 split, 2 large register file, 3 block
/// loads and stores, 4 both, 5 block in 16-lane sub-groups; every form gives the same bits); before graph capture.
void gdn_slots_set_form(int form);
/// The forms gdn_slots_set_form takes: [0, gdn_slots_forms()).
int gdn_slots_forms();
/// gdn_step_norm_multi for every slot group at once.  `commit`: each group runs all its rows and writes its state (the
/// one-row self-commit); else the states are read only (verify).  `y_q8_1`: the window's q8_1 image of y from row
/// `row0` on (the caller's xq_ for the group [row0, ...)), as the per-slot calls' QFUSE pointer.
void gdn_step_norm_slots(const GdnSlotTable& tab, const float* h, int conv_channels, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         bool commit, void* stream, int row0 = 0, void* y_q8_1 = nullptr);
/// The commit of `n_layers` GDN layers in two launches (upstream 6f3e3810): layer i's conv history is
/// `state + i * state_stride + conv_off` and its qkv rows `qkv + i * qkv_stride`, as gdn_conv_commit; its recurrence
/// state `state + i * state_stride` over the first *n_keep rows of `h + i * h_stride` / `gate|beta + i * gb_stride`, as
/// gdn_step_norm_multi's commit (n_keep set, no outputs).  Every layer's arithmetic is the per-layer call's, so the
/// result is bitwise the loop of per-layer calls.  Returns false (nothing launched) where the per-layer commit would not
/// take the state-only kernel (STRATA_GDN_COMMIT_SPLIT=0, STRATA_DBG_GDN=1): the caller then keeps the loop.
bool gdn_commit_layers(float* state, int64_t state_stride, int64_t conv_off, const float* qkv, int64_t qkv_stride,
                       const float* h, int64_t h_stride, const float* gate, const float* beta, int64_t gb_stride,
                       int conv_channels, int h_k, int h_v, int n_max, int n_layers, const int32_t* n_keep, void* stream);
/// (`y_q8_1`: the outputs from t_out_begin on also as native_quantize_q8_1 writes them, value_dim per token - the same
/// bytes, a launch fewer; upstream 22abb92d's QFUSE.)
/// Times gdn_step_norm_multi's forms (the same bits) on the stream's device for every window size and keeps the
/// fastest; call it before any graph is recorded (once a device; later calls return at once).  Untuned, the
/// one-work-group-a-head form runs.  STRATA_GDN_STEP=0|1|2 fixes the form.
void gdn_step_tune(int conv_channels, int h_k, int h_v, void* stream);
/// For tests, before graph capture: STRATA_GDN_STEP's form for gdn_step_norm_multi (0 one group a head, 1 split,
/// 2 prefetch; -1 the tuned one), and STRATA_GDN_BLOCK_GRF (block loads/stores and the large register file in the
/// per-slot recurrence and the state commit; the same bits).
void gdn_step_set_form(int form);
void gdn_step_set_block_grf(bool on);
/// Whether side branches of a graph pay on this device: 16 rounds of eight small kernels in a row and three more,
/// recorded in a row on `stream`, against the three as one on `stream` beside one on each of `side0` / `side1`
/// (forked from `stream` and joined back, as the window's GDN layer branches), each graph replayed in turns; true when
/// the branched graph takes at most 95% of the other's time.  Call it before any graph is recorded on these queues.
bool graph_branch_pays(void* stream, void* side0, void* side1);
/// Spin until *flag >= value (a mapped host flag).  The value is fixed at capture, so several rings can be
/// outstanding at once (the split verify window keeps two).
void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream);
/// Whether a running kernel sees the host's write to host memory, as wait_flag_ge needs: a kernel rings, then spins
/// (bounded) on a flag the host sets once it has seen the ring.  The UHD 770 sees neither direction while the
/// kernel runs (the verify window then runs in host-launched segments); the B70 sees both.  Synchronizes `stream`.
bool doorbell_visible(void* stream);
/// The device clock into buf[i] (a one-thread kernel: the verify window's stage profiler).  CUDA wrote %globaltimer
/// in ns; SYCL's clocks count ticks of an unspecified rate (the B70's sub-group clock: GPU core cycles), so the
/// profiler measures each window's rate (Verifier::accumulate_profile).  The clock's scope: STRATA_VERIFY_PROFILE_CLOCK
/// (device, sub_group or work_group; default device), else the first of them the device has.
void gpu_stamp(unsigned long long* buf, int i, void* stream);
/// Whether the device has a clock gpu_stamp reads (the B70: the sub-group clock alone).
bool gpu_stamp_available();
/// The clock scope gpu_stamp reads: "device", "sub_group", "work_group", or "none".
const char* gpu_stamp_clock();

// ---- perf-review E-6: a layer whose routed experts are all in VRAM needs nothing from the host
/// One group's plan, built on the device when every routed expert of its n*k entries is resident: the host pool's
/// layout (counts | start | dst | tok | pad | ptr | ptr2 | start2, `capx` entries) and order (distinct experts in
/// routing order, their entries ascending), no PCIe groups.  *skip = ring when it did, else 0.
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream);
/// wait_flag_ge that also returns when *skip == value (device memory).
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream);
/// copy_i32_from_mapped unless *skip == value.
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream);
/// copy_from_mapped, or zeros when *skip == value (n a multiple of 4, 16-byte aligned).
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream);

/// Rows of the S2/S4/S8 embedding for T token ids read from DEVICE memory; out (T, n).  Bitwise `embedding_gather`.
void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream);
/// R[t][c][:] = x[t][:] for the hc streams.
void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// dst[:n] = src[(*index) * stride + :n]  (index read from device memory; a negative index copies nothing).
void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream);

/// Plan v0.3 P6: copy *n (device memory) blobs of `blob_bytes` from mapped host memory (src[k], device aliases)
/// into dst + k * blob_bytes with coalesced 16-byte loads - the PCIe share of a layer's missed experts, staged
/// into VRAM before the grouped expert kernel reads them.  Launched for a capacity of `cap` blobs.
void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream);
/// Measures once per device how many work-groups fetch_blobs needs to fill the link (the fewest within 2% of the
/// fastest), so the VRAM hits beside it keep the rest of the GPU.  Before it fetch_blobs runs two a compute unit.
void fetch_blobs_tune(void* stream);
/// ptr[k] = base + k * blob_bytes for k < *n (the staged copies `fetch_blobs` made).
void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream);

// ---- the MTP draft layer (src/core/mtp.cpp)
/// R[t][c][:] = h[t][c][:] + e[t][:]  (the embedding branch added to every stream).
void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream);
/// Every routed expert is resident (slot = expert id): slot[i] = ids[i], dst[i] = i, *count = n.
void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream);
/// The draft chain's next input: R_dst[:] = R_src[row], tok_dst[0] = ids[row], out[j] = ids[row], with
/// row = *row_dev (device memory).  `out` may be mapped host memory.
void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs = nullptr,
                float* out_p = nullptr);
/// --pipeline-windows, the drafter's chain teacher forced: `*tok = force[j]` when force[j] >= 0 (`force` is mapped
/// host memory, read when the kernel runs), else `*tok` is left as it is.
void force_token(int32_t* tok, const int32_t* force, int j, void* stream);
/// Row *row_dev of a, b and c (a_n, b_n, c_n floats a row) copied to their row 0 (the draft layer's rest runs on
/// row 0).  Graph-capturable: the row is read on the device.
void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
                       void* stream);
/// dst row i = src row ids[i] (row_bytes each, multiple of 16), for n rows.
void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream);
/// ids[t] = table[ids[t]] for n entries (a subset index back to a token id).
void map_ids(int32_t* ids, const int32_t* table, int n, void* stream);
/// probs[t] = softmax(logits[t])[ids[t]] for n_rows rows of n_vocab (the probability of each row's argmax).
void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream);
/// #879 guard (upstream f332a855): flags[t * kNonfiniteBlocks + b] = 1 when slice b of row t (n floats a row) holds a
/// NaN or an infinity, else 0, for n_rows rows.  Every word is written on every launch (nothing to clear between
/// windows), so `flags` can be mapped host memory read after the window's own sync.  Graph-capturable.
inline constexpr int kNonfiniteBlocks = 16;
void logits_nonfinite_rows(const float* logits, int n_rows, int n, uint32_t* flags, void* stream);
/// Dense-attention step records for `n` cells: [cell, cell+1, (cell+1)/4, cell+1] from cells[i] (device memory).
void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream);
/// A sliding attention window: for `n` step records (kStepCount ints each, n_kv at [1]) the selection becomes the
/// last `window` cells: ids[q * ids_stride + j] = max(0, n_kv - window) + j and the record's width = the count.
void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream);

}  // namespace strata::kernels
