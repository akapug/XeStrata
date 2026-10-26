// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#include "strata/core/homelab_profile.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/kernels/dmmq_contract.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/dmmq_dispatch.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/hc_mmq.hpp"
#include "strata/kernels/hc_xmx.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/session_slots.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/l2_prefetch.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
// STRATA_ROUTE_RESIDENT=<margin logits> (+ STRATA_ROUTE_RESIDENT_RANKS=lo-hi, default 6-9; upstream 5df35dcb, 8bf203f5):
// residency-biased routing (opt-in, changes the output).
struct RouteResidentCfg;
RouteResidentCfg& route_resident_cfg();
struct RouteResidentCfg {
    float margin = 0.0f;
    int lo = 6, hi = 9;
    // #1578: one counter block per GPU.  A layer split inits a verifier per stage; the Verifier asks for its device's
    // block in init (outside graph capture) and keeps the pointer, so a launch never picks the block by the host
    // thread's current device.  nullptr (the allocation failed): the kernel runs without counters.
    std::mutex mu;
    std::map<int, unsigned long long*> blocks;
    unsigned long long* stats_on(int dev) {
        if (margin <= 0.0f || dev < 0) return nullptr;
        std::lock_guard<std::mutex> lk(mu);
        const auto it = blocks.find(dev);
        if (it != blocks.end()) return it->second;
        unsigned long long* p = nullptr;
        {
            const OnDevice on(dev);
            if (!strata::gpu::alloc_device(&p, 8 * sizeof(unsigned long long)) ||
                !strata::gpu::memset(p, 0, 8 * sizeof(unsigned long long))) {
                if (p != nullptr) strata::gpu::free(p);
                p = nullptr;
                std::fprintf(stderr, "route-resident: the counters could not be allocated on GPU %d; it runs without them\n", dev);
            }
        }
        if (blocks.empty()) {
            std::atexit([] {
                RouteResidentCfg& c = route_resident_cfg();
                for (const auto& kv : c.blocks) {
                    if (kv.second == nullptr) continue;
                    unsigned long long h[8] = {};
                    const OnDevice on(kv.first);
                    if (!strata::gpu::device_sync() || !strata::gpu::copy(h, kv.second, sizeof(h))) continue;
                    char who[32] = "";
                    if (c.blocks.size() > 1) std::snprintf(who, sizeof who, " (GPU %d)", kv.first);
                    std::fprintf(stderr, "route-resident%s: margin=%g ranks=%d-%d windows T>8 (prompt reads): tail_nonres=%llu "
                                         "swaps=%llu (%.1f%%) nonres_entries before=%llu after=%llu\n",
                                 who, c.margin, c.lo, c.hi, h[0], h[1], h[0] ? 100.0 * (double) h[1] / (double) h[0] : 0.0, h[2], h[3]);
                    std::fprintf(stderr, "route-resident%s: margin=%g ranks=%d-%d windows T<=8 (decode): tail_nonres=%llu "
                                         "swaps=%llu (%.1f%%) nonres_entries before=%llu after=%llu\n",
                                 who, c.margin, c.lo, c.hi, h[4], h[5], h[4] ? 100.0 * (double) h[5] / (double) h[4] : 0.0, h[6], h[7]);
                }
            });
        }
        blocks[dev] = p;
        return p;
    }
};
RouteResidentCfg& route_resident_cfg() {
    static RouteResidentCfg c;
    static const bool parsed = [] {
        if (const char* v = std::getenv("STRATA_ROUTE_RESIDENT")) c.margin = (float) std::strtod(v, nullptr);
        if (const char* v = std::getenv("STRATA_ROUTE_RESIDENT_RANKS")) {   // lo-hi
            char* end = nullptr;
            const long lo = std::strtol(v, &end, 10);
            if (end != v && *end == '-') {
                const char* h = end + 1;
                const long hi = std::strtol(h, &end, 10);
                if (end != h) { c.lo = (int) lo; c.hi = (int) hi; }
            }
        }
        return true;
    }();
    (void) parsed;
    return c;
}
// A one-token window always keeps its token, so its graph advances the GDN conv history and recurrence state itself
// (its indexer append and PLE history are final already) and Verifier::commit launches no commit graph after it
// (upstream 352cad8f): the draft round then runs alone instead of beside the commit's kernels, which read and write
// each GDN layer's state.  STRATA_ONE_TOKEN_COMMIT=0: the commit graph after every window.
bool one_token_self_commit() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ONE_TOKEN_COMMIT");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
// the head's hyper-connection read for the window's tokens in one fused_gr_read_multi (upstream edc592b1: its sums
// are gr_read's); STRATA_HEAD_MIX_MULTI=0: gr_read per token
bool head_mix_multi_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
// STRATA_NO_BATCH_KV_STEP=1: the window's and the commit's K/V and indexer appends one call per token again
bool no_batch_kv() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_NO_BATCH_KV_STEP");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}
// ---- batch windows (sa2 port plan; every switch opt-in until tools/batch_test.py passes on the B70)
bool env_flag(const char* name, bool dflt) {
    const char* v = std::getenv(name);
    return v == nullptr ? dflt : std::strtol(v, nullptr, 10) != 0;
}
// STRATA_BATCH_SELF_COMMIT=1: a batch window whose rows are each another slot's one token keeps them itself, as a
// one-token solo window does (upstream CUDA verify.cpp's g_batch_self_commit): the GDN conv histories and states
// advance inside the window graph and no commit graph (nor its host sync) runs after it.  The rest of the commit graph
// only repeated what the window wrote (the indexer tails put back and appended again from the same rows, the PLE
// histories put back to the window's own snapshot), so the state after the window is the committed state, bitwise.
bool batch_self_commit_on() { static const bool on = env_flag("STRATA_BATCH_SELF_COMMIT", false); return on; }
// STRATA_BATCH_GDN_SLOTS=1: every slot's GDN conv and recurrence of a layer in one launch each (gdn_*_slots).
bool batch_gdn_slots_on() { static const bool on = env_flag("STRATA_BATCH_GDN_SLOTS", false); return on; }
// STRATA_BATCH_TOPK_ONCE=1: a batch window's QSA top-k for all its rows in one launch (rows are independent in
// qsa_block_topk; the solo window already calls it with n rows).
bool batch_topk_once_on() { static const bool on = env_flag("STRATA_BATCH_TOPK_ONCE", false); return on; }
// STRATA_BATCH_QSA_ROWS=1: a batch window's QSA rows in one launch per kernel, each row over its own slot's state (the
// `_slots` kernels: a table of the rows' pointers, baked into the captured graph as the per-row launches' were): the
// indexer tails' snapshots and appends, the K/V appends (where no row's state keeps a host copy), the block scores, one
// top-k (as STRATA_BATCH_TOPK_ONCE) and the attention.  Every row's arithmetic is its one-row call's: the same tokens.
// A streamed state's resolve stays one launch a row; with two rows of a streamed slot the rows keep the per-row order.
constexpr bool kBatchQsaRowsDefault = false;
bool batch_qsa_rows_on() { static const bool on = env_flag("STRATA_BATCH_QSA_ROWS", kBatchQsaRowsDefault); return on; }
// STRATA_BATCH_QSA_STREAMED=1 (with STRATA_BATCH_QSA_ROWS): streamed rows (--kv-resident) join the one-launch K/V
// append (each row's host copy in its table entry) and resolve in two launches for the window (kv_stream_resolve_slots)
// instead of three launches a row.  Every row's stores are its one-row calls'.
constexpr bool kBatchQsaStreamedDefault = false;
bool batch_qsa_streamed_on() {
    static const bool on = env_flag("STRATA_BATCH_QSA_STREAMED", kBatchQsaStreamedDefault);
    return on;
}
// STRATA_BATCH_QSA_APPEND_ROWS=1 / STRATA_BATCH_QSA_RESOLVE_ROWS=1 (homelab, with STRATA_BATCH_QSA_ROWS): each half of
// STRATA_BATCH_QSA_STREAMED alone.  The B70 ubench (16 rows, 6 layers): the K/V append a launch a row with its host copy
// 1.317 ms a stage window (~14 us a launch, 1449x its bytes' floor), the resolve three launches a row 0.625 ms.
constexpr bool kBatchQsaAppendRowsDefault = false, kBatchQsaResolveRowsDefault = false;
bool batch_qsa_append_rows_on() {
    static const bool on = batch_qsa_streamed_on() || env_flag("STRATA_BATCH_QSA_APPEND_ROWS", kBatchQsaAppendRowsDefault);
    return on;
}
bool batch_qsa_resolve_rows_on() {
    static const bool on = batch_qsa_streamed_on() || env_flag("STRATA_BATCH_QSA_RESOLVE_ROWS", kBatchQsaResolveRowsDefault);
    return on;
}
static_assert(strata::kernels::kQsaSlotsMax == strata::kernels::kVerifyMaxT, "a _slots table holds a batch window's rows");
// STRATA_BATCH_HANDOFF_ONE=1 (homelab K2): a layer split's hand-off (the residual streams, the block output and the
// injection of every row) leaves the earlier stage and enters the later one in one launch each, instead of three
// launches a row (48 + 48 at 16 rows).  The same bytes (copy_rows_mapped_multi copies as copy_from_mapped does).
constexpr bool kHandoffOneDefault = false;
bool handoff_one_on() { static const bool on = env_flag("STRATA_BATCH_HANDOFF_ONE", kHandoffOneDefault); return on; }
// STRATA_BATCH_PLE_LATE=1 (homelab K2): a batch window's PLE rows (the n-gram table's 16 rows a token, SSD reads in the
// default --ple-io direct) are read by the host right after the window's launch, while the GPU runs layer 0, and copied
// in at layer 1 after a wait on a flag of their own (word 8 of the flag-B line), as the solo window's late PLE does with
// layer 0's ring.  The first stage's launch no longer waits for the reads.  The same rows, the same bits.
constexpr bool kBatchPleLateDefault = false;
bool batch_ple_late_on() { static const bool on = env_flag("STRATA_BATCH_PLE_LATE", kBatchPleLateDefault); return on; }
// STRATA_BATCH_PLE_SLOTS=1 (homelab): a batch window's PLE post-ops for all its rows in 7 launches
// (native_ple_postops_slots: each row over its own slot's history, a pad row's left as it was), not ~11 launches a row
// (the block, the history advance and the snapshot copy).  Needs the batched projections and --native-ple-postops.
// Every row's values, its history advance and its snapshot are the per-row loop's: the same bits (ple_slots_parity).
constexpr bool kBatchPleSlotsDefault = false;
bool batch_ple_slots_on() {
    static const bool on = env_flag("STRATA_BATCH_PLE_SLOTS", kBatchPleSlotsDefault);
    return on;
}
constexpr int kPleFlagWord = 8;   // h_flagB_ is a 64-byte line; word 0 is flag B
static_assert(strata::kernels::PLE_N_HEADS == 16, "Verifier::ple_late_rows_ holds 16 rows a token");
// STRATA_BATCH_RESIDENT_GRAPH=1: a batch window whose layers have every expert in VRAM runs the zero-doorbell graph
// (resident_plan on the device, no host ring per layer), as the solo window's res_graph_ does.
bool batch_res_graph_on() { static const bool on = env_flag("STRATA_BATCH_RESIDENT_GRAPH", false); return on; }
// STRATA_SKIP_NO_PCIE_SHARE=1 (homelab): a verifier told it can never have a PCIe share (set_no_pcie_share) leaves out
// the share's flag wait, fetch and grouped launch, ~0.15 ms a stage-1 window on the B70 split (V4p profile: "PCIe grp"
// 0.11 + 0.04 ms with no group).  The same bits: that launch wrote nothing.
constexpr bool kSkipNoPcieShareDefault = false;
bool skip_no_pcie_share_on() {
    static const bool on = env_flag("STRATA_SKIP_NO_PCIE_SHARE", kSkipNoPcieShareDefault);
    return on;
}
// STRATA_BATCH_SKIP (TIMING ONLY: the tokens are garbage): leave parts out of the batch window to see what they cost.
// 1 = the routed experts' grouped call, 2 = the rows' QSA select + attention, 4 = the dense projections (qkv, z,
// ssm_out, q, o), 8 = the hc reads
// STRATA_BATCH_DENSE_MMQ_MIN: from this many rows a batch window's dense projections (qkv, z, ssm_out, q, o) run on
// the int8 matrix engines (iq_mmq) instead of the q8_1 GEMV; 0 = never.  Not bit-equal to the solo window's GEMV.
constexpr int kBatchDenseMmqMinDefault = 0;
// STRATA_BATCH_DENSE_GEMV_NARROW_TYPES (homelab; default empty: no change): GGUF type numbers whose NARROWING products
// (n_out < n_in: the o and ssm_out projections, 6144 -> 2560 on Flash-Next) stay on the GEMV at every row count.  The
// matrix engine tiles n_out by 128, so a 2,560-wide output is 20 tiles over a 6,144-long K: the B70 ubench at 16 rows
// (odie-b70-ubench-run1, median us, mmq vs GEMV): Q6_K (14) 233 vs 117, IQ3_S (21) 177 vs 127, IQ4_XS (23)
// 222 vs 122, but Q4_K (12) 126 vs 177 and Q5_K (13) 149 vs 173; measured value "14,21,23".  Q6_K's row-packed native
// type counts as 14.
bool batch_dense_gemv_narrow(int native_type, int64_t n_in, int64_t n_out) {
    static const std::vector<int> types = [] {
        std::vector<int> v;
        const char* e = std::getenv("STRATA_BATCH_DENSE_GEMV_NARROW_TYPES");
        for (const char* c = e; c != nullptr && *c != '\0';) {
            char* end = nullptr;
            const long t = std::strtol(c, &end, 10);
            if (end == c) break;
            v.push_back((int) t);
            c = *end == ',' ? end + 1 : end;
        }
        return v;
    }();
    const int t = native_type == strata::kernels::kNativeQ6KRows ? 14 : native_type;
    return n_out < n_in && std::find(types.begin(), types.end(), t) != types.end();
}
// STRATA_BATCH_DENSE_XMX (homelab K1; 0 = off, the default): from this many rows, also floored by
// STRATA_DMMQ_MIN_ROWS (default 7; at most 16), a batch window's dense
// projections (qkv, z, ssm_out, q, o) run on iq_mmq_decode, the decode-shaped XMX product (a sub-group 16 outputs over a
// slice of K, every weight block decoded once for all the rows), whatever their width; past it, and where it does not
// take the type, the rules above.  The same int8 operands as the matrix engines' prompt tiles, summed in another order.
constexpr int kBatchDenseXmxDefault = 0;
int batch_dense_xmx_min() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_BATCH_DENSE_XMX");
        return strata::kernels::dmmq::dispatch_min(e != nullptr && e[0] != '\0' ? std::atoi(e) : kBatchDenseXmxDefault,
                                                   std::getenv("STRATA_DMMQ_MIN_ROWS"));
    }();
    return v;
}
// New coverage stays independent of the existing dense/shared/HC controls.
// EXTRA adds native QSA K/V and the output head; BF16_XMX is a minimum row
// count (zero/off by default). BF16_ROLES: 1 alpha/beta, 2 indexer, 4 router,
// 8 PLE; it is only consulted when BF16_XMX is enabled.
bool batch_dense_extra_on() { static const bool on = env_flag("STRATA_BATCH_DENSE_EXTRA", false); return on; }
int batch_bf16_xmx_min() {
    static const int rows = [] {
        const char* v = std::getenv("STRATA_BATCH_BF16_XMX");
        return v == nullptr || std::strcmp(v, "0") == 0 ? 0 :
            std::max(2, strata::kernels::dmmq::positive_int(v, strata::kernels::kVerifyMaxT));
    }();
    return rows;
}
int batch_bf16_roles() {
    constexpr int kRolesDefault = 15;
    static const int roles = [] {
        const char* v = std::getenv("STRATA_BATCH_BF16_ROLES");
        if (v == nullptr) return kRolesDefault;
        if (std::strcmp(v, "0") == 0) return 0;
        return strata::kernels::dmmq::positive_int(v, kRolesDefault);
    }();
    return roles;
}
int batch_dense_mmq_min() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_BATCH_DENSE_MMQ_MIN");
        return e != nullptr && e[0] != '\0' ? std::atoi(e) : kBatchDenseMmqMinDefault;
    }();
    return v;
}
int batch_skip() {
    static const int v = [] { const char* e = std::getenv("STRATA_BATCH_SKIP"); return e ? std::atoi(e) : 0; }();
    return v;
}
bool one_row_per_slot(const int* rows, int S) {
    for (int i = 0; i < S; ++i)
        for (int j = i + 1; j < S; ++j)
            if (rows[i] == rows[j]) return false;
    return S >= 1;
}
bool batch_self_commits(const int* rows, int S) { return batch_self_commit_on() && one_row_per_slot(rows, S); }
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes, "verify")) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
}  // namespace

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // outside a verify stage these are the last window's numbers (it finished), not the stalled work's (upstream
    // 9631c96)
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", current ? "" : " (last window, not the current stage)",
                 last_t_, (long long) last_pos0_, cur_layer_ + 1, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
}

Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    if (cs_) strata::gpu::stream_sync(cs_);
    for (auto& e : exec_)
        if (e) strata::gpu::graph_destroy(e);
    for (auto& e : exec_res_)
        if (e) strata::gpu::graph_destroy(e);
    for (auto& v : segs_)
        for (auto* e : v) strata::gpu::graph_destroy(e);
    if (commit_exec_) strata::gpu::graph_destroy(commit_exec_);
    for (auto& kv : exec_bm_)
        if (kv.second) strata::gpu::graph_destroy(kv.second);
    for (auto& kv : commit_bm_)
        if (kv.second) strata::gpu::graph_destroy(kv.second);
    if (arena_b_) strata::gpu::free(arena_b_);
    if (h_commitb_) strata::gpu::free(h_commitb_);
    if (logit_bias_device_) strata::gpu::free(logit_bias_device_);
    if (cs_ && cs_ != ext_stream_) strata::gpu::stream_destroy(cs_);   // set_stream: the stage's stream, shared
    if (ev_done_) strata::gpu::event_destroy(ev_done_);
    if (ev_commit_) strata::gpu::event_destroy(ev_commit_);
    if (prof_pin_) strata::gpu::free(prof_pin_);
    for (strata::gpu::Event* e : tl_ev_)
        if (e) strata::gpu::event_destroy(e);
    if (tl_q_) strata::gpu::stream_destroy(tl_q_);
    for (strata::gpu::Event* e : prof_ev_)
        if (e) strata::gpu::event_destroy(e);
    for (strata::gpu::Stream s : df_side_)
        if (s) strata::gpu::stream_destroy(s);
    if (fetch_side_) strata::gpu::stream_destroy(fetch_side_);
    if (copy_) { strata::gpu::stream_sync(copy_); strata::gpu::stream_destroy(copy_); }
    if (commit_done_) strata::gpu::event_destroy(commit_done_);
    if (arena_) strata::gpu::free(arena_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_rowx_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_, h_nf_, h_route_};
    for (void* h : hosts)
        if (h) strata::gpu::free(h);
}

void Verifier::tune_device(const ModelGeometry& g, void* stream) {
    // the GDN step's fastest form on this device for each window size
    if (g.ssm_v_heads > 0 && g.ssm_k_heads > 0)
        strata::kernels::gdn_step_tune((int) g.ssm_conv_channels, (int) g.ssm_k_heads, (int) g.ssm_v_heads, stream);
    // and the PCIe share's fetch: the work-groups that fill this device's link
    strata::kernels::fetch_blobs_tune(stream);
    // and the multi-row BF16 GEMV's form for the window's shapes: the router, the indexer's key and query, the PLE
    // block's key and value
    for (const int64_t n_out : {(int64_t) g.n_expert, (int64_t) g.idx_key_dim, (int64_t) g.idx_q_heads * g.idx_key_dim,
                                (int64_t) strata::kernels::NG_HC_DIM, (int64_t) g.n_embd})
        strata::kernels::bf16_gemv_fp32_mmvf_multi_tune(g.n_embd, n_out, stream);
}

bool Verifier::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    if (next_ && !next_->set_logit_bias(bias, err)) return false;
    if (le_ < g_->n_layers) return true;
    const OnDevice on_device(device_);
    if (!bias.empty() && bias.size() != (size_t) n_vocab_) {
        err = "logit_bias vocabulary size mismatch"; return false;
    }
    if (bias != logit_bias_host_ && !bias.empty()) {
        const size_t bytes = bias.size() * sizeof(float);
        if ((!logit_bias_device_ && !strata::gpu::alloc_device(&logit_bias_device_, bytes)) ||
            !strata::gpu::copy(logit_bias_device_, bias.data(), bytes)) {
            err = "cannot upload logit_bias"; return false;
        }
    }
    logit_bias_host_ = bias;
    sampling_.logit_bias = bias.empty() ? nullptr : logit_bias_device_;
    return true;
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    device_ = current_device();   // a layer split's stage on another GPU: init runs on it (OnDevice)
    rr_stats_ = route_resident_cfg().stats_on(device_);   // STRATA_ROUTE_RESIDENT: this GPU's counters, outside graph capture
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > strata::kernels::verify_max_t() || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::verify_max_t()) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above "
              "says how much is short and what makes room (a smaller --max-context, --kv q4_0, setup --draft-vocab en, "
              "images on the CPU)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(3 * T * 4 + 16, (void**) &h_rowx_, (void**) &m_rowx_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped((size_t) strata::kernels::kVerifyMaxT * strata::kernels::kNonfiniteBlocks * 4, (void**) &h_nf_,
                     (void**) &m_nf_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped((size_t) g.n_layers * T * K * 4, (void**) &h_route_, (void**) &m_route_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T); rowx_ = b.take<int32_t>(3 * T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        if (batch_dense_mmq_min() > 0 || batch_dense_xmx_min() > 0) {
            dq_ = b.take<uint8_t>(2 * strata::kernels::iq_mmq_act_bytes(T, max_in));   // twice: a type that repeats blocks
            dbounds_ = b.take<int32_t>(2 * (T + 1)); dids_ = b.take<int32_t>(T);
        }
        if (batch_bf16_xmx_min() > 0) bx_ = b.take<uint16_t>(2 * T * max_in);
        if (strata::kernels::hc_mmq_min() > 0 || strata::kernels::hc_xmx_min() > 0) {
            hxb_ = b.take<uint16_t>(T * HC * N); hlb_ = b.take<uint16_t>(T * (uint64_t) g.hc_lr);
            hup_ = b.take<float>(T * HC * N); hxq_ = b.take<uint8_t>(strata::kernels::hc_mmq_xq_bytes((int64_t) T));
            hbounds_ = b.take<int32_t>(2 * (T + 1)); hids_ = b.take<int32_t>(T);
        }
        if (strata::kernels::hcx_min() > 0) hcx_ = b.take<uint8_t>(strata::kernels::hcx_scratch_bytes((int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        one_ = b.take<int32_t>(4);
        hist_snap_ = b.take<float>(T * HS);
        ple_key_ = b.take<float>(T * (uint64_t) strata::kernels::NG_HC_DIM); ple_val_ = b.take<float>(T * N);
    };
    Bump count;
    carve(count);
    if (!strata::gpu::alloc_device(&arena_, count.used)) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    strata::gpu::memset(arena_, 0, count.used);
    prof_on_ = std::getenv("STRATA_VERIFY_PROFILE") != nullptr;
    // The stamps are recorded in the window graph, so they need a clock a kernel reads: without one the profile is off
    // rather than the capture failing.  Level Zero native commands and profiling tags cannot be recorded in a graph.
    if (prof_on_ && !strata::kernels::gpu_stamp_available()) {
        std::fprintf(stderr, "strata verify: STRATA_VERIFY_PROFILE needs a clock a kernel reads (sycl_ext_oneapi_clock, "
                             "any scope), which this device does not have; the stage profile is off\n");
        prof_on_ = false;
    }
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 4;
        if (!strata::gpu::alloc_device((void**) &prof_, np * 8) || !strata::gpu::event_create(&prof_ev_[0], true) ||
            !strata::gpu::event_create(&prof_ev_[1], true)) { prof_on_ = false; }
        else { strata::gpu::memset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
        static bool said = false;
        if (prof_on_ && !said)
            std::fprintf(stderr, "strata verify: the stage profile reads the %s-scope clock (STRATA_VERIFY_PROFILE_CLOCK)\n",
                         strata::kernels::gpu_stamp_clock());
        said = true;
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    {
        const int32_t one = 1;
        if (!strata::gpu::copy(one_, &one, sizeof one)) { err = "verify: the arena could not be set"; return false; }
    }
    if (dq_ != nullptr) {   // the dense products' tables, and the matrix engines' probe before any graph is recorded
        std::vector<int32_t> bd((size_t) (2 * (T + 1))), id((size_t) T);
        for (int64_t i = 0; i <= T; ++i) { bd[(size_t) (2 * i)] = 0; bd[(size_t) (2 * i + 1)] = (int32_t) i; }
        for (int64_t i = 0; i < T; ++i) id[(size_t) i] = (int32_t) i;
        if (!strata::gpu::copy(dbounds_, bd.data(), bd.size() * 4) || !strata::gpu::copy(dids_, id.data(), id.size() * 4)) {
            err = "verify: the dense products' tables could not be set";
            return false;
        }
        if (!strata::prefill::mmq::chosen()) dq_ = nullptr;
        if (batch_dense_xmx_min() > 0)
            std::fprintf(stderr, "strata verify: batch dense products from %d rows on the decode XMX product "
                                 "(STRATA_BATCH_DENSE_XMX)\n", batch_dense_xmx_min());
        if (strata::kernels::iq_mmq_decode_lean() != 0)   // homelab densefuse
            std::fprintf(stderr, "strata verify: decode XMX lean form bits %d (STRATA_DMMQ_LEAN: 1 kernel, 2 shared "
                                 "gate|up pair, 4 z reuses qkv's int8 rows)\n", strata::kernels::iq_mmq_decode_lean());
        std::fprintf(stderr, "strata verify: batch dense products from %d rows on %s\n", batch_dense_mmq_min(),
                     dq_ != nullptr ? strata::prefill::mmq::path() : "the GEMV (no int8 matrix path)");
    }
    if (bx_ != nullptr) {
        const bool available = strata::kernels::xmx_available(strata::kernels::XmxType::bf16);
        std::fprintf(stderr, "strata verify: BF16x2 dense XMX from %d rows roles=%d available=%d; extra dense=%d\n",
                     batch_bf16_xmx_min(), batch_bf16_roles(), (int) available, (int) batch_dense_extra_on());
        if (!available) bx_ = nullptr;
    }
    // The shared-only opt-in needs the same allocating/layout probe even when dense scratch was not requested.
    if (batch_dense_xmx_min() > 0 || strata::kernels::shared_expert_xmx_min() > 0)
        (void) strata::kernels::iq_mmq_decode_ok(strata::core::Runtime::get().compute(), 8, 16, 32, 1);
    if (hup_ != nullptr) {   // the hc reads' tables, and every probe (engines, shapes, alignment) before any recording
        std::vector<int32_t> bd((size_t) (2 * (T + 1))), id((size_t) T);
        for (int64_t i = 0; i <= T; ++i) { bd[(size_t) (2 * i)] = 0; bd[(size_t) (2 * i + 1)] = (int32_t) i; }
        for (int64_t i = 0; i < T; ++i) id[(size_t) i] = (int32_t) i;
        if (!strata::gpu::copy(hbounds_, bd.data(), bd.size() * 4) || !strata::gpu::copy(hids_, id.data(), id.size() * 4)) {
            err = "verify: the hc products' tables could not be set";
            return false;
        }
        strata::kernels::HcMmqScratch hs;
        hs.xb = hxb_; hs.lb = hlb_; hs.up = hup_; hs.xq = hxq_; hs.bounds = hbounds_; hs.ids = hids_;
        hc_mmq_layer_.assign((size_t) g.n_layers, 0);
        int on[2] = {0, 0};   // reads taking the BF16 and the Q8_0 form
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const LayerView v(wt, l);
            const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
            for (int h = 0; h < 2; ++h) {
                const WeightRef* wn = v.get((std::string(pfx[h]) + "norm.weight").c_str());
                const WeightRef* wd = v.get((std::string(pfx[h]) + "down.weight").c_str());
                const WeightRef* wu = v.get((std::string(pfx[h]) + "up.weight").c_str());
                if (!wn || !wd || !wu) continue;
                strata::kernels::FusedGrArgs w;   // the weights as gr_read_group passes them
                w.w_norm = (const float*) wn->data;
                w.w_down = (const uint16_t*) wd->data; w.w_up = (const uint16_t*) wu->data;
                const bool q8 = wd->hc_q8 != nullptr && wu->hc_q8 != nullptr;
                if (q8) { w.q8_down = (const uint8_t*) wd->hc_q8; w.q8_up = (const uint8_t*) wu->hc_q8; }
                if (strata::kernels::hc_mmq_path(w, lo_, hs, (int) T) != strata::kernels::dmmq::HcPath::fused) {
                    hc_mmq_layer_[(size_t) l] |= (uint8_t) (1u << h);
                    ++on[q8];
                }
            }
        }
        std::fprintf(stderr, "strata verify: batch hc MMQ from %d rows, Q8 decode from %d rows: "
                     "%d BF16, %d Q8_0 eligible reads of %d\n", strata::kernels::hc_mmq_min(),
                     strata::kernels::hc_xmx_min(), on[0], on[1], (int) (2 * g.n_layers));
    }
    if (hcx_ != nullptr) {   // STRATA_BATCH_HCX: the device probe (iq_mmq's int8 XMX check) before any recording
        const bool geom = g.n_embd == strata::kernels::kHcxN && g.hc == strata::kernels::kHcxHC &&
                          g.hc_lr == strata::kernels::kHcxLR;
        hcx_on_ = geom && strata::kernels::hcx_engine();
        hcx_s_ = strata::kernels::hcx_carve(hcx_, (int) T);
        std::fprintf(stderr, "strata verify: batch hc XMX read (STRATA_BATCH_HCX) from %d rows, %d activation "
                     "slices: %s\n", strata::kernels::hcx_min(), strata::kernels::hcx_slices(),
                     hcx_on_ ? "on" : geom ? "off (no int8 XMX tiles on this device)" : "off (another geometry)");
    }
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    // STRATA_COPY_PRIO=1: the copy stream at high priority (upstream 728097fb6, opt-in)
    static const bool copy_prio = [] {
        const char* v = std::getenv("STRATA_COPY_PRIO");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    copy_ = strata::gpu::stream_create(copy_prio);
    if (copy_ == nullptr) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (ext_stream_ != nullptr) cs_ = ext_stream_;   // set_stream (pipelined windows): the stage's shared stream
    else cs_ = strata::gpu::stream_create();
    if (!cs_) {
        err = "verify: stream create failed";
        return false;
    }
    try {   // before any graph is recorded
        tune_device(g, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify: ") + e.what();
        return false;
    }
    {   // the mixer's side branches (upstream 2e4ddf6e) where they pay on this device: measured once (the RTX 4070
        // +0.7%; the B70 lost 4%, its graphs' forks and joins cost more than the overlap gave).  STRATA_DF_BRANCH=0|1
        // fixes the choice.
        const char* v = std::getenv("STRATA_DF_BRANCH");
        for (int i = 0; i < 2; ++i)
            if (!df_side_[i]) df_side_[i] = strata::gpu::stream_create();
        if (!df_side_[0] || !df_side_[1]) df_branch_ = false;
        else if (v != nullptr) df_branch_ = std::strtol(v, nullptr, 10) != 0;
        else {
            try {
                df_branch_ = strata::kernels::graph_branch_pays(cs_, df_side_[0], df_side_[1]);
            } catch (const std::exception&) {
                df_branch_ = false;   // a device that cannot record such a graph keeps one queue
            }
        }
        // the PCIe share's fetch on a side branch of the window graph, beside the VRAM hits (post()): the fetch waits
        // on the link, the hits on the GPU, so side by side the hits' time hides under the fetch.  Where the graph's
        // branches pay (df_branch_) unless STRATA_FETCH_BRANCH=0|1 fixes it.
        if (!fetch_side_) fetch_side_ = strata::gpu::stream_create();
        const char* fv = std::getenv("STRATA_FETCH_BRANCH");
        fetch_branch_ = fetch_side_ != nullptr && (fv != nullptr ? std::strtol(fv, nullptr, 10) != 0 : df_branch_);
    }
    if (!strata::gpu::event_create(&commit_done_)) {
        err = "verify: event create failed";
        return false;
    }
    {
        const char* v = std::getenv("STRATA_VERIFY_SEGMENTED");
        segmented_ = v != nullptr ? std::strtol(v, nullptr, 10) != 0 : !strata::kernels::doorbell_visible(cs_);
        if (segmented_)
            std::fprintf(stderr, "strata verify: the GPU's running kernels do not see the host's writes: each window "
                                 "runs as segments the host launches (slower)\n");
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs: off by default)
    {
        const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN");
        // (not with set_always_publish: the device table may lag the host's while windows are in flight)
        device_plan_ = !always_publish_ && v != nullptr && (int) std::strtol(v, nullptr, 10) != 0;
    }
    {
        const char* v = std::getenv("STRATA_VERIFY_RESIDENT_GRAPH");
        res_graph_ = !always_publish_ && hits.h_res != nullptr && hits.d_res != nullptr &&
                     (v == nullptr || std::strtol(v, nullptr, 10) != 0);
    }
    if (device_plan_ || res_graph_) {
        bool ok2 = strata::gpu::alloc_device((void**) &skip_, 64) && strata::gpu::memset(skip_, 0, 64);
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = strata::gpu::alloc_device((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) &&
                  strata::gpu::copy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long));
        }
        if (!ok2) { device_plan_ = false; res_graph_ = false; }
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers\n", max_t,
                 (double) count.used / 1048576.0);
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, strata::gpu::Stream cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2 && !batch_rec_) ? 2 : 1;   // a batch window is one group
    // STRATA_DF_BRANCH: mixer work that reads only the layer's input on a side branch of the graph.  fork(i): side i
    // after everything on cs so far; join(i): cs after everything on side i - a branch never races with what came
    // before its fork or after its join.  Not in a split or batch window, nor under the profiler (its stamps are on cs).
    const bool br = df_branch_ && G == 1 && !batch_rec_ && !prof_on_;
    auto fork = [&](int i) { return strata::gpu::stream_fork(cs, df_side_[i]); };
    auto join = [&](int i) { return strata::gpu::stream_join(cs, df_side_[i]); };
    // see Verifier::commit; not in a pipelined window (always_publish_), which a misprediction rolls back
    const bool self_commit = T == 1 && !batch_rec_ && !always_publish_ && one_token_self_commit();
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    auto stamp = [&](int64_t l, int i, int grp) { if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs); };
    // STRATA_BATCH_DENSE_MMQ_MIN: a batch window's dense product y[rows, n_out] = x[rows, n_in] W^T on the int8 matrix
    // engines, as one expert of `rows` rows (plain_prompt.cpp's dense form); false: the caller takes the q8_1 GEMV
    // STRATA_DMMQ_LEAN's reuse bit: what the last quantize left in dq_ (x, its width and rows, whether with sums), so
    // that a product of the same input (the GDN z after qkv: conv and a/b write neither x nor dq_) reads those bytes
    struct { const float* x = nullptr; int64_t n = 0, rows = 0; bool sums = false; } dq_last;
    auto dense = [&](const WeightRef* w, const float* x, float* y, int64_t n_in, int64_t n_out, int64_t rows,
                     bool reuse = false) {
        if (dq_ != nullptr && batch_rec_ && batch_dense_xmx_min() > 0 && rows >= batch_dense_xmx_min()) {
            sycl::queue& q = strata::core::Runtime::get().stream(cs);
            if (strata::kernels::iq_mmq_decode_ok(q, w->native_type, n_out, n_in, rows)) {
                const bool sums = strata::kernels::iq_mmq_type_sums(w->native_type);
                if (strata::kernels::iq_mmq_decode_native_q8_1()) {
                    // Every dense caller already prepared xq_ for its GEMV fallback (including fused producers).
                    strata::kernels::iq_mmq_decode_from_q8_1(q, xq_, dq_, n_in, rows, sums);
                    dq_last.x = nullptr;
                } else if (!(reuse && (strata::kernels::iq_mmq_decode_lean() & strata::kernels::dmmq::kLeanReuse) &&
                             dq_last.x == x && dq_last.n == n_in && dq_last.rows == rows && (dq_last.sums || !sums) &&
                             !strata::kernels::iq_mmq_type_dup(w->native_type))) {
                    strata::prefill::mmq::quantize(x, nullptr, dq_, w->native_type, n_in, n_in, rows, cs);
                    dq_last.x = strata::kernels::iq_mmq_type_dup(w->native_type) ? nullptr : x;
                    dq_last.n = n_in;
                    dq_last.rows = rows;
                    dq_last.sums = sums;
                }
                strata::kernels::iq_mmq_decode(q, w->native_type, w->native_data, n_out, n_in, dq_, rows, y, n_out);
                return true;
            }
        }
        if (dq_ == nullptr || !batch_rec_ || rows < batch_dense_mmq_min() || batch_dense_mmq_min() <= 0 ||
            !strata::kernels::iq_mmq_type_ok(w->native_type) || n_out % 128 != 0 || n_in % 32 != 0 ||
            batch_dense_gemv_narrow(w->native_type, n_in, n_out))
            return false;
        static strata::prefill::mmq::Context ctx;
        strata::prefill::mmq::quantize(x, nullptr, dq_, w->native_type, n_in, n_in, rows, cs);
        dq_last.x = nullptr;
        strata::prefill::mmq::Product p;
        p.w = w->native_data; p.type = w->native_type; p.w_rows = n_out; p.w_cols = n_in; p.n = 1; p.xq = dq_;
        p.bounds = dbounds_ + 2 * rows; p.ids = dids_; p.total_rows = rows; p.max_rows = rows;
        p.dst = y; p.ld_dst = n_out;
        ctx.run(p, cs);
        return true;
    };
    auto bdense = [&](const uint16_t* w, const float* x, float* y, int64_t n_in, int64_t n_out, int rows, int role) {
        const bool taken = batch_rec_ && bx_ != nullptr && rows >= batch_bf16_xmx_min() && (batch_bf16_roles() & role) != 0 &&
            strata::kernels::bf16_gemm_fp32_xmx(x, n_in, w, y, n_out, n_in, n_out, rows, bx_, cs);
        static const bool trace = env_flag("STRATA_BATCH_DENSE_TRACE", false);
        if (trace && batch_rec_)
            std::fprintf(stderr, "strata dense capture: bf16x2=%d role=%d rows=%d K=%lld N=%lld\n",
                         (int) taken, role, rows, (long long) n_in, (long long) n_out);
        return taken;
    };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    bool gate_later_[2] = {false, false};   // the group's shared rows still to be scaled by the combine (pre -> post)
    if (!batch_rec_) groups_[T] = G;
    const int64_t nQall = g.n_qsa_layers();
    // a batch window: row t is slot t, whose state lives in its own session (upstream PR #559)
    auto slot_ss = [&](int t) -> SessionState& { return batch_rec_ ? *slots_[(size_t) brow_[t]] : ss; };
    // STRATA_BATCH_SELF_COMMIT: the window keeps its rows (each row another slot's one token)
    const bool keep_rows = batch_rec_ && batch_self_commits(brow_, T);
    const bool gdn_slots = batch_rec_ && batch_gdn_slots_on();

    // ---- the window's inputs, from mapped staging
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    // a batch window's per-row writes (stage_batch: a pad row appends its K/V past its slot's history and keeps
    // nothing else); the append kernels read a step's position at kStepPos, so a position array of stride 1 stands in
    static_assert(kStepPos == 0, "the batch window's row positions stand in for the steps");
    if (batch_rec_) copy_i32_from_mapped(rowx_, m_rowx_, (int64_t) 3 * MT, cs);
    const int32_t* row_apos = rowx_;
    const int32_t* row_ipos = rowx_ + MT;
    const int32_t* row_keep = rowx_ + 2 * MT;
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);
    // The window's PLE rows: read from the table by the host after the launch, while the GPU runs layer 0, and
    // copied in at layer 1 after the GPU's wait for layer 0's flag (upstream a36be1db reads them at layer 0's ring):
    // the reads no longer hold the window's launch back (0.35 ms a window on the RTX 4070).  Not where the host
    // raises no flag for layer 0 before layer 1: the resident graph, a batch window, a stage that starts after layer 0.
    const bool ple_late = ple_on && !recording_res_ && !batch_rec_ && lb_ == 0;
    const bool ple_late_b = ple_on && batch_rec_ && lb_ == 0 && batch_ple_late_on();   // STRATA_BATCH_PLE_LATE
    if (ple_on && !ple_late && !ple_late_b) copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
    const int hrow0 = batch_rec_ ? row_base_ : 0;   // a slot group's own hand-off rows (batch_launch)
    if (lb_ > 0 && handoff_one_on()) {   // STRATA_BATCH_HANDOFF_ONE: the T rows' three pieces in one launch
        const float* hin = hand_in_ + (size_t) hrow0 * HB;
        const RowsCopy pc[3] = {{R_, hin, HC * N, HC * N, HB}, {bo_, hin + HC * N, N, N, HB},
                                {inj2_, hin + HC * N + N, HC, HC, HB}};
        copy_rows_mapped_multi(pc, 3, T, cs);
    } else if (lb_ > 0) {
        for (int t = 0; t < T; ++t) {
            const float* hin = hand_in_ + (size_t) (hrow0 + t) * HB;
            copy_from_mapped(Rt(t), hin, HC * N, cs);
            copy_from_mapped(bo_ + (size_t) t * N, hin + HC * N, N, cs);
            copy_from_mapped(inj2_ + (size_t) t * HC, hin + HC * N + N, HC, cs);
        }
    } else if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            if (ple_late && grp == 0) {
                wait_flag_ge(m_flag_, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            if (ple_late_b && grp == 0) {   // the host gathered the rows after the launch (ple_late_publish)
                wait_flag_ge(m_flagB_ + kPleFlagWord, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            // the window's residual writes in one launch (each row its own R, block output and injection)
            if (dec_batch) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
            // The key and value projections depend only on each row's n-gram embedding: the group's rows go through
            // the multi-row forms of the same kernels, the weights read once, each row's arithmetic the single-row
            // call's (upstream d92c9feb).  The history-dependent rest stays row by row.  STRATA_PLE_BATCH=0: row by row.
            static const bool ple_batch = [] {
                const char* v = std::getenv("STRATA_PLE_BATCH");
                return v == nullptr || std::strtol(v, nullptr, 10) != 0;
            }();
            PleWeights pw = ss.ple.w;
            const bool nkey = pw.key_native_data != nullptr && pw.key_bf16 == nullptr;
            const bool batched = ple_batch && n > 1 && (pw.key_bf16 != nullptr || nkey) && ple_native_bf16();
            if (batched) {
                try {
                    if (pw.key_bf16 != nullptr) {
                        if (!bdense(pw.key_bf16, ple_ + tb * N, ple_key_ + (size_t) tb * NG_HC_DIM, N, NG_HC_DIM, n, 8))
                            bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.key_bf16, ple_key_ + (size_t) tb * NG_HC_DIM,
                                                      NG_HC_DIM, N, NG_HC_DIM, n, cs);
                    } else {
                        native_quantize_q8_1(ple_ + tb * N, xq_, (int) N, n, cs);
                        native_mmvq(pw.key_native_type, pw.key_native_data, xq_, ple_key_ + (size_t) tb * NG_HC_DIM,
                                    (int) N, NG_HC_DIM, n, cs);
                    }
                    if (!bdense(pw.value_bf16, ple_ + tb * N, ple_val_ + tb * N, N, N, n, 8))
                        bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.value_bf16, ple_val_ + tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE batch: ") + e.what();
                    return false;
                }
            }
            // with the batched projections and the native post-ops, the rows' post-ops in one pass that also writes
            // the commit's history snapshots: native_ple_postops_batch's arithmetic is the rows' calls with the history
            // advanced after each (upstream 2e4ddf6e's STRATA_DF_PLE).  Scratch: the group's expert rows (parts_,
            // hit_out_, combined by now) and lo_ (the next read rewrites it).  STRATA_DF_PLE=0: row by row.
            static const bool df_ple = [] {
                const char* e = std::getenv("STRATA_DF_PLE");
                return e == nullptr || std::strtol(e, nullptr, 10) != 0;
            }();
            const bool ple_pass = df_ple && batched && dec_batch && !batch_rec_ && ple_native_postops_enabled();
            // a batch window's rows each over its own slot's history (STRATA_BATCH_PLE_SLOTS): one run of rows a slot,
            // or the per-row loop where a slot's rows are not contiguous
            PleSlots pslots;
            bool ple_slots = !ple_pass && batched && dec_batch && batch_rec_ && batch_ple_slots_on() &&
                             ple_native_postops_enabled();
            for (int t = tb; ple_slots && t < te;) {
                const int first = t;
                while (t < te && brow_[t] == brow_[first]) ++t;
                for (int j = 0; j < pslots.n && ple_slots; ++j) ple_slots = pslots.hist[j] != slot_ss(first).ple_hist;
                if (!ple_slots) break;
                pslots.hist[pslots.n] = slot_ss(first).ple_hist;
                pslots.first[pslots.n] = first - tb;
                pslots.ntok[pslots.n] = t - first;
                ++pslots.n;
            }
            if (ple_slots) {
                try {
                    native_ple_postops_slots(ple_key_ + (size_t) tb * NG_HC_DIM, Rt(tb), ple_val_ + tb * N, pslots,
                                             row_keep + tb, ss.ple.w, parts_ + (size_t) tb * K * N,
                                             hit_out_ + (size_t) tb * K * N, lo_ + (size_t) tb * g.hc_lr, n,
                                             hist_snap_ + (size_t) tb * HS, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE (slots): ") + e.what();
                    return false;
                }
            }
            if (ple_pass) {
                try {
                    native_ple_postops_batch_snap(ple_key_ + (size_t) tb * NG_HC_DIM, Rt(tb), ple_val_ + tb * N,
                                                  ss.ple.hist, ss.ple.w, parts_ + (size_t) tb * K * N,
                                                  hit_out_ + (size_t) tb * K * N, lo_ + (size_t) tb * g.hc_lr, n,
                                                  hist_snap_ + (size_t) tb * HS, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE (window): ") + e.what();
                    return false;
                }
            }
            for (int t = tb; t < te && !ple_pass && !ple_slots; ++t) {
                if (batched) {
                    pw.pre_key = ple_key_ + (size_t) t * NG_HC_DIM;
                    pw.pre_value = ple_val_ + t * N;
                }
                if (!dec_batch) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                float* hist = batch_rec_ ? slot_ss(t).ple_hist : ss.ple.hist;   // the row's own history
                try {
                    ple_block(ple_ + t * N, Rt(t), hist, pw, po, ss.ple.scratch, cs);
                    ple_history_advance(hist, normalized, cs, batch_rec_ ? row_keep + t : nullptr);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kVerifyMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                if (wd[half]->hc_q8 != nullptr && wu[half]->hc_q8 != nullptr) {   // STRATA_HC_Q8
                    a.q8_down = (const uint8_t*) wd[half]->hc_q8; a.q8_up = (const uint8_t*) wu[half]->hc_q8;
                    // the first half's mixed as q8_1 too, for the GDN / QSA projections (no quantize launch)
                    if (half == 0 && fused_gr_q8_fuse()) a.q8_mixed = xq_ + (size_t) (t - tb) * (N / 32) * 36;
                }
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
            }
            if (batch_rec_ && (batch_skip() & 8)) return;
            unsigned long long* sb = (prof_on_ && grp == 0) ? prof_ : nullptr;
            const int si = (int) (l * kProfPer + (half == 0 ? 27 : 30));
            // STRATA_BATCH_HCX (hc_xmx.hpp): both products as int8 on XMX, the norm in the down launch; before the
            // MMQ opt-ins below
            if (batch_rec_ && hcx_on_ && strata::kernels::hcx_take(fa, n)) {
                strata::kernels::hcx_read(fa, n, hcx_s_, cs, strata::kernels::hcx_slices(),
                                          strata::kernels::HcxBody::automatic, sb, si);
                return;
            }
            // Shared with the roll-up: explicit HC opt-ins, both projection contracts and the actual row count.
            if (batch_rec_ && tb == 0 && !hc_mmq_layer_.empty() &&
                ((hc_mmq_layer_[(size_t) l] >> half) & 1u) != 0) {
                HcMmqScratch hs;
                hs.xb = hxb_; hs.lb = hlb_; hs.up = hup_; hs.xq = hxq_; hs.bounds = hbounds_ + 2 * n; hs.ids = hids_;
                if (hc_mmq_path(fa[0], fa[0].lo, hs, n) != dmmq::HcPath::fused) {
                    fused_gr_read_mmq(fa, n, xn_, hs, cs, sb, si);
                    return;
                }
            }
            fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs, sb, si);
        };
        // STRATA_HC_PREFETCH=1: a half's Q8_0 projections asked into L2 ahead of its read (l2_prefetch.hpp)
        auto prefetch_hc = [&](int half) {
            if (hc_prefetch() != 1 || wd[half]->hc_q8 == nullptr || wu[half]->hc_q8 == nullptr) return;
            l2_prefetch(wd[half]->hc_q8, (size_t) wd[half]->elements / 32 * 34, wu[half]->hc_q8,
                        (size_t) wu[half]->elements / 32 * 34, cs);
        };
        gr_read_group(0, pending, inj2_, inj_);
        // the read wrote mixed's q8_1 into xq_ itself (STRATA_HC_Q8's 32-column up)
        const bool mixed_q8 = wd[0]->hc_q8 != nullptr && wu[0]->hc_q8 != nullptr && fused_gr_q8_fuse();
        stamp(l, 1, grp);
        float* xm = mixed_ + tb * N;
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                if (!mixed_q8) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                if (br) {   // a/b and z beside q/k/v and the conv: they read only this layer's input (xm, xq_)
                    if (!fork(0)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                    gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                 (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                 (int) HV, n, df_side_[0]);
                    if (!fork(1)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                    native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,
                                df_side_[1]);
                }
                if (!(batch_rec_ && (batch_skip() & 4)) && !dense(wqkv, xm, qkv + (size_t) tb * C, N, C, n)) {
                    MmvqRoleScope role(MmvqRole::qkv);   // the STRATA_MMVQ_SLM rules key on the product's role
                    native_mmvq(wqkv->native_type, wqkv->native_data, xq_, qkv + (size_t) tb * C, (int) N, (int) C, n, cs);
                }
                stamp(l, 2, grp);
                // the slot groups of this window's rows (a batch window: brow_), for the one-launch forms
                GdnSlotTable gtab;
                gtab.keep = row_keep;   // a pad row's slot keeps its history and state
                if (gdn_slots)
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        float* stx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats;
                        gtab.state[gtab.n] = stx;
                        gtab.hist[gtab.n] = stx + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        gtab.first[gtab.n] = first;
                        gtab.ntok[gtab.n] = t - first;
                        ++gtab.n;
                    }
                // STRATA_GDN_AB_FORM 4..6: alpha/beta inside the conv's launch (gdn_ab_multi's bits), not a launch of its own
                const bool ab_gemm = !br &&
                    bdense((const uint16_t*) wa->data, xm, gate + (size_t) tb * HV, N, HV, n, 1) &&
                    bdense((const uint16_t*) wb->data, xm, beta + (size_t) tb * HV, N, HV, n, 1);
                if (ab_gemm) gdn_ab_activate(gate + (size_t) tb * HV, beta + (size_t) tb * HV,
                                             (const float*) wdt->data, (const float*) wsa->data, (int) HV, n, cs);
                const bool ab_in_conv = gdn_slots && !br && !ab_gemm && gdn_ab_fused_in_conv();
                if (gdn_slots) {   // every slot's conv in one launch (rows [tb, te) are the whole window: G is 1)
                    GdnAbArgs ab;
                    if (ab_in_conv) {
                        ab.x = xm; ab.w_alpha = (const uint16_t*) wa->data; ab.w_beta = (const uint16_t*) wb->data;
                        ab.dt = (const float*) wdt->data; ab.ssm_a = (const float*) wsa->data;
                        ab.gate = gate + (size_t) tb * HV; ab.beta = beta + (size_t) tb * HV;
                        ab.n_embd = (int) N; ab.h_v = (int) HV; ab.n_tok = n;
                    }
                    gdn_conv_l2_slots(gtab, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, cs, keep_rows,
                                      ab_in_conv ? &ab : nullptr);
                } else
                if (batch_rec_) {   // each slot's rows from its own conv history (--batch-mtp: a slot's rows are contiguous)
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        const float* cx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats +
                                          (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        gdn_conv_l2_multi(cx, qkv + (size_t) first * C, (const float*) wc->data, hb + (size_t) first * C,
                                          (int) C, (int) (2 * HK), EPS, t - first, cs, 0, keep_rows);
                    }
                } else
                // a one-token window keeps its token: it advances the conv history and the state itself (commit)
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb,
                                  self_commit);
                stamp(l, 3, grp);
                if (br) {   // gate, beta and z, before the recurrence reads them
                    if (!join(0) || !join(1)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                } else {
                    if (!ab_in_conv && !ab_gemm)
                        gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                     (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                     (int) HV, n, cs);
                    stamp(l, 4, grp);
                    if (!(batch_rec_ && (batch_skip() & 4)) && !dense(wg, xm, z_ + (size_t) tb * ZV, N, ZV, n, true))
                        native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, cs);
                    stamp(l, 5, grp);
                }
                // the output norm also writes ssm_out's q8_1 input (the same bytes, a launch fewer; upstream
                // 22abb92d, 0a22c467 for the batch's rows).  STRATA_QFUSE=0: a quantize launch of its own.
                static const bool qfuse = [] {
                    const char* e = std::getenv("STRATA_QFUSE");
                    return e == nullptr || std::strtol(e, nullptr, 10) != 0;
                }();
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                if (gdn_slots) {   // every slot's recurrence in one launch (self-commit: each writes its state)
                    gdn_step_norm_slots(gtab, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                        (int) HV, keep_rows, cs, tb, qfuse ? (void*) xq_ : nullptr);
                } else
                if (batch_rec_) {   // each slot's recurrence over its own rows, from its own state
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        float* stx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats;
                        gdn_step_norm_multi(stx, hb + (size_t) first * C, (int) C, gate + (size_t) first * HV,
                                            beta + (size_t) first * HV, z_ + (size_t) first * ZV,
                                            (const float*) wnm->data, EPS, y_ + (size_t) first * ZV, (int) HK, (int) HV,
                                            t - first, keep_rows ? one_ : nullptr, cs, 0,
                                            qfuse ? (void*) (xq_ + (size_t) (first - tb) * (ZV / 32) * 36) : nullptr);
                    }
                } else
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, self_commit ? one_ : nullptr, cs, tb, qfuse ? (void*) xq_ : nullptr);
                stamp(l, 6, grp);
                prefetch_hc(1);   // overlapped with ssm_out, read right after it
                if (!qfuse) native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs);
                if (!(batch_rec_ && (batch_skip() & 4)) && !dense(wout, y_ + (size_t) tb * ZV, bo_ + tb * N, ZV, N, n)) {
                    MmvqRoleScope role(MmvqRole::ssm_out);
                    native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
                }
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope_on = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos,
                                        strata::gpu::Stream on) {
                    if (native_qsa_enabled() && native_rope_enabled() && native_norm_rope_usable(cols, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos, on);
                        return;
                    }
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, on);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, on);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, on);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, on);
                };
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    norm_rope_on(data, norm, rows, cols, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                // a Q4_0 K/V takes the batched rows too: they rotate the query as the per-token path does (upstream
                // 2e4ddf6e's STRATA_DF_QB_Q4; =0: Q4_0 token by token)
                static const bool qb_q4 = [] {
                    const char* e = std::getenv("STRATA_DF_QB_Q4");
                    return e == nullptr || std::strtol(e, nullptr, 10) != 0;
                }();
                const bool qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && (!st.kv_q4 || qb_q4);
                if (!mixed_q8) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                // the query side on branches (upstream 2e4ddf6e): q with its split, norm, RoPE and rotation on side 0, the
                // indexer query on side 1, beside the K/V side; side 1 joins before the block scores, side 0 before
                // attention (xq_ and mixed_ are not rewritten before then)
                const bool qbr = br && qb;
                // q out of the q|gate rows, normed and rotated: one launch where the fused kernel applies, else the
                // split copy and norm_rope (the batched rows only: native QSA and native RoPE are on)
                auto q_split_norm_rope = [&](strata::gpu::Stream on) -> bool {
                    if (native_norm_rope_usable((int) HD, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(qfull_ + tb * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data,
                                                 qcur_ + tb * NH * HD, (int) (n * NH), (int) HD, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos_ + tb * NH, on);
                        return true;
                    }
                    if (!strata::gpu::copy2d_async(qcur_ + tb * NH * HD, (size_t) HD * 4, qfull_ + tb * NH * 2 * HD,
                                                   (size_t) HD * 2 * 4, (size_t) HD * 4, (size_t) (n * NH), on)) {
                        err = std::string("verify: the q/gate split failed: ") + strata::gpu::last_error();
                        return false;
                    }
                    norm_rope_on(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH, on);
                    return true;
                };
                if (qbr) {
                    if (!fork(0)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                    native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N,
                                (int) (NH * 2 * HD), n, df_side_[0]);
                    if (!q_split_norm_rope(df_side_[0])) return false;
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, df_side_[0]);
                    if (!fork(1)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                              N, IQ * ID, n, df_side_[1]);
                    norm_rope_on(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, df_side_[1]);
                }
                if (bdense((const uint16_t*) wik->data, xm, idx_raw + tb * ID, N, ID, n, 2)) {
                    // XMX produced the same raw projection layout; established post-ops follow below.
                } else if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
                else for (int t = tb; t < te; ++t)
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                stamp(l, 7, grp);
                if (!batch_dense_extra_on() || !dense(wk, xm, kcur_ + tb * NKV * HD, N, NKV * HD, n))
                    native_mmvq(wk->native_type, wk->native_data, xq_, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                if (!batch_dense_extra_on() || !dense(wv, xm, vcur_ + tb * NKV * HD, N, NKV * HD, n))
                    native_mmvq(wv->native_type, wv->native_data, xq_, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
                    fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
                // STRATA_BATCH_QSA_ROWS: every row's K/V append in one launch (no row's state keeps a host copy, one format),
                // and every slot's tail snapshot and indexer appends in one more (below, after the K/V)
                const bool qrows = batch_rec_ && batch_qsa_rows_on();
                KvAppendSlots kvtab;
                bool kv_rows = qrows;
                const QsaState& s0 = slot_ss(tb).qsa_states[qi];   // the rows' format (every slot is carved alike)
                for (int t = tb; kv_rows && t < te; ++t) {
                    const QsaState& sx = slot_ss(t).qsa_states[qi];
                    kv_rows = (!sx.host.present() || batch_qsa_append_rows_on()) &&
                              sx.host.present() == s0.host.present() && sx.kv_hybrid == s0.kv_hybrid &&
                              sx.kv_q4 == s0.kv_q4 && sx.kv_int8 == s0.kv_int8 && sx.page_table != nullptr;
                    const int r = t - tb;
                    const KvHostPools& h = sx.host;
                    kvtab.page_table[r] = sx.page_table;
                    if (sx.kv_hybrid) {
                        kvtab.k[r] = sx.k_q; kvtab.k_scale[r] = sx.k_scale; kvtab.v[r] = sx.v_q4;
                        kvtab.host_k[r] = h.k_q; kvtab.host_k_scale[r] = h.k_scale; kvtab.host_v[r] = h.v_q4;
                    } else if (sx.kv_q4) {
                        kvtab.k[r] = sx.k_q4; kvtab.v[r] = sx.v_q4;
                        kvtab.host_k[r] = h.k_q4; kvtab.host_v[r] = h.v_q4;
                    } else if (sx.kv_int8) {
                        kvtab.k[r] = sx.k_q; kvtab.v[r] = sx.v_q; kvtab.k_scale[r] = sx.k_scale; kvtab.v_scale[r] = sx.v_scale;
                        kvtab.host_k[r] = h.k_q; kvtab.host_v[r] = h.v_q;
                        kvtab.host_k_scale[r] = h.k_scale; kvtab.host_v_scale[r] = h.v_scale;
                    } else {
                        kvtab.k[r] = sx.k_pool; kvtab.v[r] = sx.v_pool;
                        kvtab.host_k[r] = h.k_pool; kvtab.host_v[r] = h.v_pool;
                    }
                }
                if (qrows) {
                    // the snapshot rides in the indexer launch below
                } else
                if (batch_rec_) {   // every slot's indexer tail before its first row, restored by its commit
                    for (int t = tb; t < te; ++t)
                        if (t == tb || brow_[t] != brow_[t - 1])
                            copy_from_mapped(tail_snap_b_ + ((size_t) brow_[t] * nQall + qi) * TS,
                                             slot_ss(t).qsa_states[qi].idx_tail, TS, cs);
                } else
                if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                // the window's cells in one launch per pool (a batch's rows each have their own K/V: one per row)
                const bool kv_steps = dec_batch && !no_batch_kv() && !batch_rec_;
                if (kv_steps) {
                    const int32_t* step_b = step_ + (ptrdiff_t) tb * kStepCount;
                    const float* kc_b = kcur_ + tb * NKV * HD;
                    const float* vc_b = vcur_ + tb * NKV * HD;
                    if (st.kv_hybrid) {   // K8V4: the used pool for both halves (layer.cpp)
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();   // streamed: the host copy too
                        kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_b,
                                           (int) kStepCount, kc_b, kc_b, (int) (NKV * HD), n, s, cs,
                                           mirror ? &hk : nullptr);
                        kv_append_q4_steps(st.v_q4, st.v_q4, st.page_table, step_b, (int) kStepCount, n, vc_b, vc_b,
                                           s, cs, mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, step_b, (int) kStepCount, n, kc_b, vc_b, s,
                                           cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_b,
                                           (int) kStepCount, kc_b, vc_b, (int) (NKV * HD), n, s, cs, &st.host);
                    else
                        for (int t = tb; t < te; ++t)
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_ + (ptrdiff_t) t * kStepCount,
                                           kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                } else if (kv_rows) {   // each row's own cell (row_apos: a pad row's lies past its slot's history)
                    const int32_t* step_b = row_apos + tb;
                    const float* kc_b = kcur_ + tb * NKV * HD;
                    const float* vc_b = vcur_ + tb * NKV * HD;
                    if (s0.kv_hybrid) {   // K8V4: K as int8 (the K plane only), V as q4_0 (its blocks in the k entries)
                        kv_append_q8_slots(kvtab, step_b, 1, kc_b, kc_b, (int) (NKV * HD), n, 1, s, cs);
                        KvAppendSlots vtab;
                        for (int r = 0; r < n; ++r) {
                            vtab.k[r] = kvtab.v[r];
                            vtab.page_table[r] = kvtab.page_table[r];
                            vtab.host_k[r] = kvtab.host_v[r];
                        }
                        kv_append_q4_slots(vtab, step_b, 1, n, vc_b, vc_b, 1, s, cs);
                    } else if (s0.kv_q4)
                        kv_append_q4_slots(kvtab, step_b, 1, n, kc_b, vc_b, 2, s, cs);
                    else if (s0.kv_int8)
                        kv_append_q8_slots(kvtab, step_b, 1, kc_b, vc_b, (int) (NKV * HD), n, 2, s, cs);
                    else
                        kv_append_slots(kvtab, step_b, 1, kc_b, vc_b, (int) (NKV * HD), n, s, cs);
                } else
                for (int t = tb; t < te; ++t) {
                    const QsaState& st = slot_ss(t).qsa_states[qi];   // the row's own K/V (ss's outside a batch)
                    const int32_t* step_t = batch_rec_ ? row_apos + t : step_ + t * kStepCount;
                    if (st.kv_hybrid) {   // K8V4: the used pool for both halves (layer.cpp)
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();   // streamed: the host copy too
                        kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, mirror ? &hk : nullptr);
                        kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                }
                if (kv_steps) {
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    native_qsa_indexer_append_steps(idx_raw + tb * ID, step_ + (ptrdiff_t) tb * kStepCount + kStepPos,
                                                    (int) kStepCount, n, 0, (const float*) wikn->data, EPS, ib, s,
                                                    st.max_cells, rope_scaling(), cs);
                } else if (qrows) {   // one work-group a slot: its tail snapshot (restored by its commit), then its rows
                    QsaIndexerSlots itab;
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        const QsaState& sx = slot_ss(first).qsa_states[qi];
                        const int j = itab.n++;
                        itab.snap[j] = tail_snap_b_ + ((size_t) brow_[first] * nQall + qi) * TS;
                        itab.tail[j] = sx.idx_tail;
                        itab.dead[j] = sx.idx_dead;
                        itab.pooled[j] = sx.idx_pooled;
                        itab.block_pos[j] = sx.idx_block_pos;
                        itab.first[j] = first - tb;
                        itab.ntok[j] = t - first;
                        itab.max_cells[j] = (int) std::min<int64_t>(sx.max_cells, INT32_MAX);
                    }
                    native_qsa_indexer_append_slots(idx_raw + tb * ID, row_ipos + tb, 1, 0, (const float*) wikn->data,
                                                    EPS, itab, s, rope_scaling(), cs);   // (-1: a pad row, none)
                } else
                for (int t = tb; t < te; ++t) {
                    const QsaState& sx = slot_ss(t).qsa_states[qi];
                    const QsaIndexerBuffers ib{sx.idx_tail, sx.idx_dead, sx.idx_pooled, sx.idx_block_pos};
                    native_qsa_indexer_append(idx_raw + t * ID, batch_rec_ ? row_ipos + t : step_ + t * kStepCount + kStepPos, 0,
                                              (const float*) wikn->data, EPS, ib, s, sx.max_cells,
                                              rope_scaling(), cs);
                }
                stamp(l, 9, grp);
                if (!qbr && !(batch_rec_ && (batch_skip() & 4)) && !dense(wq, xm, qfull_ + tb * NH * 2 * HD, N, NH * 2 * HD, n))
                native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD),
                            n, cs);
                if (qbr) {
                    // the query side runs on the branches
                } else if (qb) {
                    if (!q_split_norm_rope(cs)) return false;
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, cs);   // <Hq, Hk> = <q, k>
                    if (!bdense((const uint16_t*) wiq->data, xm, qidx_ + tb * IQ * ID, N, IQ * ID, n, 2))
                        bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                                  N, IQ * ID, n, cs);
                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);
                } else {
                for (int t = tb; t < te; ++t) {
                    float* qc = qcur_ + t * NH * HD;
                    if (!strata::gpu::copy2d_async(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4, (size_t) NH, cs)) {
                        err = std::string("verify: the q/gate split failed: ") + strata::gpu::last_error();
                        return false;
                    }
                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
                    if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
                }
                for (int t = tb; t < te; ++t) {
                    float* qx = qidx_ + t * IQ * ID;
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
                }
                }
                stamp(l, 10, grp);
                if (batch_rec_ && (batch_skip() & 2)) {
                } else if (batch_rec_ && (batch_topk_once_on() || qrows)) {
                    // STRATA_BATCH_TOPK_ONCE: each row's block scores from its own slot's indexer keys, then ONE top-k
                    // launch for the rows (row-independent: the solo window's call with n rows), then each row's
                    // resolve and attention over its own slot's K/V.  A row's resolve may evict another row's blocks
                    // only within one slot (one map a slot): --batch-mtp's two rows of a slot keep the per-row order
                    // below, since its resolve-then-attend pairs must not interleave.
                    // STRATA_BATCH_QSA_ROWS: the scores and the attention one launch each, from the rows' tables
                    QsaScoreSlots stab;
                    QsaAttnSlots atab;
                    bool streamed = false, attn_rows = qrows;
                    for (int t = tb; qrows && t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        stab.pooled[t - tb] = sx.idx_pooled;
                        stab.dead[t - tb] = sx.idx_dead;
                        streamed = streamed || sx.kv_mode == 1;
                        attn_rows = attn_rows && qsa_attn_slots_set(atab, t - tb, qsa_attn_pools(sx));
                    }
                    // a streamed slot's resolve may evict its own earlier row's blocks: two rows of one keep the
                    // resolve-then-attend pairs below
                    attn_rows = attn_rows && (!streamed || one_row_per_slot(brow_ + tb, n));
                    if (qrows)
                        qsa_block_scores_slots(stab, qidx_ + (size_t) tb * IQ * ID, step_ + (size_t) tb * kStepCount, n,
                                               max_blocks_, s, scores_ + (size_t) tb * max_blocks_, cs);
                    else
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        qsa_block_scores(sx.idx_pooled, sx.idx_dead, qidx_ + (size_t) t * IQ * ID,
                                         step_ + (size_t) t * kStepCount, 1, max_blocks_, s,
                                         scores_ + (size_t) t * max_blocks_, cs);
                    }
                    qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + (size_t) tb * kStepCount, n, max_blocks_,
                                   cap_, s, sel_ + (size_t) tb * cap_, cs);
                    if (attn_rows) {   // each streamed slot's resolve (one row each), then every row's attention
                        KvResolveSlots rtab;
                        bool rrows = streamed && batch_qsa_resolve_rows_on();
                        for (int t = tb; rrows && t < te; ++t) {
                            const QsaState& sx = slot_ss(t).qsa_states[qi];
                            rrows = sx.kv_mode == 1 && qsa_kv_format(sx) == qsa_kv_format(s0);
                            rtab.map[t - tb] = sx.map;
                            rtab.slots[t - tb] = qsa_attn_pools(sx);
                            rtab.host[t - tb] = sx.host;
                        }
                        if (rrows)
                            kv_stream_resolve_slots(rtab, qsa_kv_format(s0), sel_ + (size_t) tb * cap_,
                                                    step_ + (size_t) tb * kStepCount, n, cap_, s, cs);
                        for (int t = tb; streamed && !rrows && t < te; ++t)
                            qsa_kv_resolve(slot_ss(t).qsa_states[qi], *g_, sel_ + (size_t) t * cap_,
                                           step_ + (size_t) t * kStepCount, 1, cap_, cs);
                        qsa_decode_attn_slots(qcur_ + (size_t) tb * NH * HD, atab, sel_ + (size_t) tb * cap_,
                                              step_ + (size_t) tb * kStepCount, cap_, s,
                                              attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n,
                                              cs);
                    } else
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        const int32_t* step_t = step_ + (size_t) t * kStepCount;
                        qsa_kv_resolve(sx, *g_, sel_ + (size_t) t * cap_, step_t, 1, cap_, cs);
                        const QsaAttnPools px = qsa_attn_pools(sx);
                        qsa_decode_attn_batch(qcur_ + (size_t) t * NH * HD, px, sel_ + (size_t) t * cap_, step_t,
                                              cap_, s, attn_scratch_ + (size_t) t * attn_scratch_floats_,
                                              attn_ + t * NH * HD, 1, cs);
                    }
                } else
                if (batch_rec_) {   // each row selects and attends over its own slot's K/V
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        const int32_t* step_t = step_ + (size_t) t * kStepCount;
                        qsa_block_scores(sx.idx_pooled, sx.idx_dead, qidx_ + (size_t) t * IQ * ID, step_t, 1,
                                         max_blocks_, s, scores_ + (size_t) t * max_blocks_, cs);
                        qsa_block_topk(scores_ + (size_t) t * max_blocks_, step_t, 1, max_blocks_, cap_, s,
                                       sel_ + (size_t) t * cap_, cs);
                        qsa_kv_resolve(sx, *g_, sel_ + (size_t) t * cap_, step_t, 1, cap_, cs);
                        const QsaAttnPools px = qsa_attn_pools(sx);
                        qsa_decode_attn_batch(qcur_ + (size_t) t * NH * HD, px, sel_ + (size_t) t * cap_, step_t,
                                              cap_, s, attn_scratch_ + (size_t) t * attn_scratch_floats_,
                                              attn_ + t * NH * HD, 1, cs);
                    }
                } else {
                if (qbr && !join(1)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                // the window's rows read each block's key once (upstream 5b6c9398's STRATA_QSA_ROWS; the same bits).
                // STRATA_QSA_ROWS=0: a sub-group per row and block, as before
                static const bool qsa_rows = [] {
                    const char* v = std::getenv("STRATA_QSA_ROWS");
                    return v == nullptr || std::strtol(v, nullptr, 10) != 0;
                }();
                const float* q_win = qidx_ + (ptrdiff_t) tb * IQ * ID;
                const int32_t* st_win = step_ + (ptrdiff_t) tb * kStepCount;
                float* sc_win = scores_ + (size_t) tb * max_blocks_;
                if (qsa_rows && n > 1)
                    qsa_block_scores_rows(st.idx_pooled, st.idx_dead, q_win, st_win, n, max_blocks_, s, sc_win, cs);
                else
                    qsa_block_scores(st.idx_pooled, st.idx_dead, q_win, st_win, n, max_blocks_, s, sc_win, cs);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, cs);
                stamp(l, 11, grp);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                const QsaAttnPools pools = qsa_attn_pools(st);
                if (qbr && !join(0)) { err = std::string("verify branch: ") + strata::gpu::last_error(); return false; }
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                }
                stamp(l, 13, grp);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = tb; t < te; ++t) {
                    if (native_qsa_enabled())
                        native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                              (int) NH, (int) HD, cs);
                    else
                        qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                }
                stamp(l, 14, grp);
                prefetch_hc(1);   // overlapped with o, read right after it
                native_quantize_q8_1(attn32_ + tb * NH * HD, xq_, (int) (NH * HD), n, cs);
                if (!(batch_rec_ && (batch_skip() & 4)) && !dense(wo, attn32_ + tb * NH * HD, bo_ + tb * N, NH * HD, N, n))
                    native_mmvq(wo->native_type, wo->native_data, xq_, bo_ + tb * N, (int) (NH * HD), (int) N, n, cs);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16, grp);
        gr_read_group(1, true, inj_, inj2_);
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        // STRATA_LFUSE_GATE=0: the shared expert's scalar gate in a launch of its own, and its sigmoid scale too
        static const bool lfuse_gate = [] {
            const char* e = std::getenv("STRATA_LFUSE_GATE");
            return e == nullptr || std::strtol(e, nullptr, 10) != 0;
        }();
        bool g_ready = false;   // the shared expert's raw gates came with the router's launch
        bool qdedup = false;    // the experts' q8_1 image of the input was made before the shared expert
        // the native router is 512 x 10 only; another expert count (the GSQ-RCO Coder keeps 256 of 512) routes the
        // window the same two-launch way through the generic top-k, which is what moe_route runs per token for it
        const bool native_rt = native_router_enabled() && NE == 512 && K == 10;
        if (dec_batch && n > 1 && w_router != nullptr && (native_rt || NE != 512)) {
            try {
                // the shared expert's scalar gate as one more work-group of the router's launch (upstream b08cf3e1)
                const WeightRef* w_sg = v.get("ffn_gate_inp_shexp.weight");
                const bool sg_aux = lfuse_gate && w_sg != nullptr && w_sg->kind == WeightKind::Bf16InF32 &&
                                    shared_expert_native_bf16();
                // STRATA_ROUTER_FUSE: the GEMV, the gate and the top-10 in one launch (the same bits)
                const bool projected = bdense((const uint16_t*) w_router->data, mixed_ + tb * N,
                                                logits_ + tb * NE, N, NE, n, 4);
                bool routed = false;
                if (!projected && native_rt && bf16_router_fuse()) {
                    routed = bf16_router_fused(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, N,
                                               n, sg_aux ? (const uint16_t*) w_sg->data : nullptr,
                                               sg_aux ? sh_g_ + tb : nullptr, 1, ids_ + tb * K, w_ + tb * K, cs);
                    g_ready = routed && sg_aux;
                }
                if (!projected && !routed && sg_aux)
                    g_ready = bf16_gemv_fp32_mmvf_multi_aux(mixed_ + tb * N, N, (const uint16_t*) w_router->data,
                                                            logits_ + tb * NE, NE, N, NE, n, (const uint16_t*) w_sg->data,
                                                            sh_g_ + tb, 1, cs);
                if (!projected && !routed && !g_ready)
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE,
                                              N, NE, n, cs);
                if (!routed && native_rt) native_router_top10_multi(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, cs);
                else if (!routed) router_top10(logits_ + tb * NE, n, (int) NE, (int) K, ids_ + tb * K, w_ + tb * K, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        // STRATA_ROUTE_PRIOR=lambda (upstream 2de9ae06, opt-in, changes the output): the top-10 picked again with a
        // logit bonus for the resident experts; the weights stay the router's (router.cpp)
        static const float route_prior = [] { const char* v = std::getenv("STRATA_ROUTE_PRIOR"); return v ? (float) std::strtod(v, nullptr) : 0.0f; }();
        if (route_prior > 0.0f && NE == 512 && K == 10 && hits_.d_res != nullptr) {
            try {
                route_prior_top10(logits_ + tb * NE, hits_.d_res + l * g.n_expert, route_prior, ids_ + tb * K, w_ + tb * K, n, cs);
            } catch (const std::exception& e) { err = "verify route-prior: " + std::string(e.what()); return false; }
        }
        if (route_resident_cfg().margin > 0.0f && NE == 512 && K == 10 && hits_.d_res != nullptr) {
            // STRATA_ROUTE_RESIDENT: after the router, before the plan/doorbell read ids_/w_ (opt-in, changes the output)
            try {
                native_route_resident(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, hits_.d_res + l * g.n_expert, n,
                                      route_resident_cfg().margin, route_resident_cfg().lo, route_resident_cfg().hi,
                                      rr_stats_ != nullptr ? rr_stats_ + (n <= 8 ? 4 : 0) : nullptr, cs);
            } catch (const std::exception& e) { err = "verify route-resident: " + std::string(e.what()); return false; }
        }
        if (device_plan_ || recording_res_)   // E-6: every routed expert resident: this group's plan without the host
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
                          (uint32_t) ((l - lb_) * G + grp + 1), cs);
        if (!recording_res_)   // the resident graph asks the host nothing
            doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                             m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
        if (route_rec_ && !batch_rec_)   // add_routing's ids: every layer's, also where the host is not asked
            copy_i32_from_mapped(m_route_ + ((size_t) l * (size_t) MT + (size_t) tb) * (size_t) K, ids_ + tb * K,
                                 (int64_t) n * K, cs);
        stamp(l, 17, grp);
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = xq_;
            // The experts' canonical q8_1 image first, reused by the established shared GEMV path (upstream be7889ed).
            // Its scale may span 256 values: it is not native32 provenance for the opt-in shared decode adapter.
            // The BF16 copy is only for a gate that reads it. STRATA_VERIFY_QDEDUP=0: as before.
            static const bool qdedup_on = [] {
                const char* e = std::getenv("STRATA_VERIFY_QDEDUP");
                return e == nullptr || std::strtol(e, nullptr, 10) != 0;
            }();
            qdedup = qdedup_on && strata::kernels::cpu::expert_layout().native;
            if (qdedup) quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
            if (!(qdedup && shared_expert_native_bf16())) {
                if (dec_batch) f32_to_bf16_bulk(mixed_ + tb * N, sh_bf16_ + tb * N, (int64_t) n * N, cs);   // contiguous rows
                else for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, cs);
            }
            try {
                // the window's combine scales the shared rows itself (upstream b08cf3e1; the same bits, a launch fewer)
                gate_later_[grp] = shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data,
                                                       sh_gate_ + (size_t) tb * g.n_ff, sh_up_ + (size_t) tb * g.n_ff,
                                                       sh_g_ + tb, shared_ + tb * N, N, g.n_ff, cs,
                                                       lfuse_gate && dec_batch && n > 1 && native_moe_combine_enabled(),
                                                       g_ready, qdedup ? (const void*) (nat_xq_ + (size_t) tb * (N / 32) * 36)
                                                                       : nullptr, false);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
        }
        if (strata::kernels::cpu::expert_layout().native) {
            if (!qdedup) quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
        } else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 18, grp);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        if (recording_res_) {
            // the plan resident_plan built in pre(l)
        } else if (segmented_) {     // the host launched this segment after the plan was published
            if (device_plan_)
                copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring,
                                            cs);
            else
                copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
        } else if (device_plan_) {   // E-6: skipped when the device planned this group (all its experts resident)
            wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
            copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
        } else {
            wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's GPU plan
            copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
        }
        stamp(l, 19, grp);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // plan v0.3 P6: the VRAM groups now; the PCIe groups once the copy engine has landed them in staging
        // grid_groups: the VRAM call a row per possible group, the PCIe call kPcieGroupRows (it rarely has any group,
        // and its rows stride over the ones it has)
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t grid_groups) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type);
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, hit_out, cs, grid_groups);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, hit_out, cs);
            }
        };
        // the PCIe share: its flag, and in the kernel mode the copy into staging.  On a side branch beside the VRAM
        // hits (fetch_branch_: the fetch waits on the link, the hits on the GPU; the fetch kernel leaves the hits their
        // work-groups), else on cs after them.
        auto pcie_share = [&](strata::gpu::Stream on) {
            if (segmented_) {}                                   // landed before the host launched this segment
            else if (device_plan_) wait_flag_ge_or(m_flagB_, ring, skip_ + grp, on);
            else wait_flag_ge(m_flagB_, ring, on);                 // the PCIe share is in staging (DMA) or mapped
            if (sink_.pcie_mode == 2) {                            // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, on);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), on);
            }
        };
        const bool pcie_path = !(no_pcie_share_ && skip_no_pcie_share_on());
        const bool fb = fetch_branch_ && !recording_res_ && pcie_path;
        if (fb) {
            if (!strata::gpu::stream_fork(cs, fetch_side_)) { err = std::string("verify fetch branch: ") + strata::gpu::last_error(); return false; }
            pcie_share(fetch_side_);
        }
        if (!(batch_rec_ && (batch_skip() & 1))) grouped(p_ptr, p_start, p_counts, cap);
        stamp(l, 20, grp);
        if (recording_res_) {   // no PCIe share and no CPU share: the CPU's rows are zeros
            copy_or_zero_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N,
                                     skip_ + grp, ring, cs);
        } else {
            if (fb) {
                if (!strata::gpu::stream_join(cs, fetch_side_)) { err = std::string("verify fetch branch: ") + strata::gpu::last_error(); return false; }
            } else if (pcie_path) {
                pcie_share(cs);
            }
            stamp(l, 21, grp);
            if (pcie_path) grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows);
            stamp(l, 22, grp);
            if (device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
                if (!segmented_) wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs);
                copy_or_zero_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N,
                                         skip_ + grp, ring, cs);
            } else {
                if (!segmented_) wait_flag_ge(m_flag_, ring, cs);   // the CPU's share is in the mapped rows
                stamp(l, 23, grp);
                if (dec_batch)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                    copy_rows_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                          p_dst, p_counts + 1, cs);
                else
                    copy_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
            }
        }
        const bool multi_combine = dec_batch && n > 1 && native_moe_combine_enabled();
        const bool fold_hits = multi_combine && native_moe_fuse_hits();   // the hit rows added inside the combine
        if (!fold_hits) moe_hit_add(parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, cap, N, cs);
        if (multi_combine) {   // one launch for the window's rows
            try {
                if (fold_hits)
                    native_moe_combine_multi_hits(parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, w_ + tb * K,
                                                  shared_ + tb * N, bo_ + tb * N, N, K, n, cs,
                                                  gate_later_[grp] ? sh_g_ + tb : nullptr);
                else
                    native_moe_combine_multi(parts_ + (size_t) tb * K * N, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, N,
                                             K, n, cs, gate_later_[grp] ? sh_g_ + tb : nullptr);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        stamp(l, 24, grp);
        if (l == g.n_layers - 1) {
            if (dec_batch) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
            else for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
            if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        return true;
    };

    // segmented: a graph ends where the host must answer a ring (before each group's experts)
    auto cut = [&]() -> bool {
        if (!segmented_ || recording_res_) return true;
        strata::gpu::Graph* e = nullptr;
        if (!strata::gpu::end_capture(cs, &e)) {
            err = std::string("verify: segment capture: ") + strata::gpu::last_error();
            return false;
        }
        seg_out_->push_back(e);
        if (!strata::gpu::begin_capture(cs)) { err = "verify: begin capture failed"; return false; }
        return true;
    };
    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!cut() || !post(l, grp)) return false;
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }
    // The non-final stage has no head: reuse its otherwise unused head stamps (+0/+1).  Only the existing
    // timeline + stage profiler opt in; no new buffer, clock or graph split.  Both hand-off forms get the same span.
    const bool handoff_tl = batch_rec_ && (batch_timeline() > 0 || batch_raw_windows());
    // The same packing kernels, now to group-owned device rows when opted in.  The copy engine reads these, NOT
    // R_/bo_/inj2_: those scratch buffers belong to the next group as soon as this graph and its commit finish.
    float* const hand_write = batch_rec_ && outbound_.device_rows() ? outbound_.device_rows() : hand_out_;
    if (le_ < g.n_layers && handoff_tl) stamp(g.n_layers, 0, 0);
    if (le_ < g.n_layers && handoff_one_on()) {   // STRATA_BATCH_HANDOFF_ONE: one launch for the T rows' pieces
        float* hout = hand_write + (size_t) hrow0 * HB;
        const RowsCopy pc[3] = {{hout, R_, HC * N, HB, HC * N}, {hout + HC * N, bo_, N, HB, N},
                                {hout + HC * N + N, inj2_, HC, HB, HC}};
        copy_rows_mapped_multi(pc, 3, T, cs);
        if (handoff_tl) stamp(g.n_layers, 1, 0);
        return true;
    }
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        for (int t = 0; t < T; ++t) {
            float* hout = hand_write + (size_t) (hrow0 + t) * HB;
            copy_from_mapped(hout, Rt(t), HC * N, cs);
            copy_from_mapped(hout + HC * N, bo_ + (size_t) t * N, N, cs);
            copy_from_mapped(hout + HC * N + N, inj2_ + (size_t) t * HC, HC, cs);
        }
        if (handoff_tl) stamp(g.n_layers, 1, 0);
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        // STRATA_HC_Q8=1: the final mixer's Q8_0 projections are read in this form only
        const bool mix_q8 = hd->hc_q8 != nullptr && hu->hc_q8 != nullptr;
        const bool mix_multi = (head_mix_multi_enabled() || mix_q8) && head_ != nullptr && head_->loaded() &&
                               fused_gr_supported(g.n_embd, g.hc, g.hc_lr);
        if (mix_multi) {
            // the final mixer, the window's tokens in one read (no pending write: the last layer's was done above)
            if (hn->kind != WeightKind::F32 || hd->kind != WeightKind::Bf16InF32 || hu->kind != WeightKind::Bf16InF32) {
                err = "verify: the output_hc_* weights have the wrong engine forms";
                return false;
            }
            FusedGrArgs fa[kVerifyMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = Rt(t); fa[t].R_out = Rt(t); fa[t].apply = false;
                fa[t].w_norm = (const float*) hn->data; fa[t].w_down = (const uint16_t*) hd->data;
                fa[t].w_up = (const uint16_t*) hu->data; fa[t].eps = EPS;
                if (mix_q8) { fa[t].q8_down = (const uint8_t*) hd->hc_q8; fa[t].q8_up = (const uint8_t*) hu->hc_q8; }
                fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = head_mixed_ + t * N;
            }
            try {
                fused_gr_read_multi(fa, T, xn_, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head mixer: ") + e.what();
                return false;
            }
        }
        for (int t = 0; t < T && !mix_multi; ++t) {
            BlockBuffers bb = ss.block;
            bb.R = Rt(t);
            bb.mixed = head_mixed_ + t * N;
            if (head_ != nullptr && head_->loaded()) {
                if (!lm_head_mix(wt, g, bb, cs, err)) return false;
            } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                return false;
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                WeightRef hw;
                hw.native_type = head_->type(); hw.native_data = head_->weights();
                if (!batch_dense_extra_on() || !dense(&hw, head_mixed_, head_logits_, N, n_vocab_, T)) {
                    native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
                }
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
        // #879: whether each row's logits are all finite, beside the pick (read after the window's own sync)
        if (nan_guard_on()) strata::kernels::logits_nonfinite_rows(head_logits_, T, (int) n_vocab_, m_nf_, cs);
    }
    stamp(g.n_layers, 1, 0);
    return true;
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""};
    // ms where the windows' clock rate was measured (prof_mark), else millions of ticks
    const bool ms = prof_ticks_ > 0 && prof_ns_ > 0;
    const double unit = ms ? prof_ns_ / prof_ticks_ / 1e6 : 1e-6;
    std::string out;
    char b[120];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] * unit / (double) prof_windows_);
            out += b;
        }
    }
    if (ms)
        std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows (%s clock, %.3f ticks/ns)",
                      total * unit / (double) prof_windows_, (long long) prof_windows_,
                      strata::kernels::gpu_stamp_clock(), prof_ticks_ / prof_ns_);
    else
        std::snprintf(b, sizeof b, " | total %.2f M ticks/window over %lld windows (%s clock)",
                      total * unit / (double) prof_windows_, (long long) prof_windows_, strata::kernels::gpu_stamp_clock());
    out += b;
    prof_ns_ = prof_ticks_ = 0;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

bool Verifier::capture(int T, std::string& err) {
    if (exec_[T] != nullptr || !segs_[T].empty()) return true;
    {   // said before the capture: a process that exits inside it leaves this line as the trace (upstream 0f931943)
        size_t free_b = 0, total_b = 0;
        if (strata::gpu::mem_info(&free_b, &total_b))
            std::fprintf(stderr, "strata verify: capturing the %d-token window (%zu MiB of VRAM free)\n", T, free_b >> 20);
    }
    if (!strata::gpu::begin_capture(cs_)) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    bool ok = false;
    seg_out_ = &segs_[T];
    try {
        ok = record_window(T, cs_, rerr);
    } catch (const std::exception& e) {
        rerr = std::string("verify: window capture: ") + e.what();
    }
    seg_out_ = nullptr;
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        for (auto* e : segs_[T]) strata::gpu::graph_destroy(e);
        segs_[T].clear();
        err = rerr;
        return false;
    }
    strata::gpu::Graph* last = nullptr;
    if (!strata::gpu::end_capture(cs_, &last)) {
        err = std::string("verify: end capture: ") + strata::gpu::last_error();
        return false;
    }
    if (segmented_) segs_[T].push_back(last);
    else exec_[T] = last;
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr)   // what the window graph holds (SYCL names no kernels)
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes\n", T,
                     strata::gpu::graph_nodes(segmented_ ? segs_[T].back() : exec_[T]));
    const bool synced = strata::gpu::stream_sync(cs_);
    std::fprintf(stderr, "strata verify: captured the %d-token window (sync %s)\n", T,
                 synced ? "ok" : strata::gpu::last_error());
    return true;
}

bool Verifier::all_resident() const {
    const int64_t ne = g_->n_expert;
    for (int64_t i = lb_ * ne; i < le_ * ne; ++i)
        if (hits_.h_res[i] < 0) return false;
    return true;
}

bool Verifier::capture_res(int T, std::string& err) {
    if (exec_res_[T] != nullptr) return true;
    if (!strata::gpu::begin_capture(cs_)) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    bool ok = false;
    recording_res_ = true;
    try {
        ok = record_window(T, cs_, rerr);
    } catch (const std::exception& e) {
        rerr = std::string("verify: window capture: ") + e.what();
    }
    recording_res_ = false;
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        err = rerr;
        return false;
    }
    if (!strata::gpu::end_capture(cs_, &exec_res_[T])) {
        err = std::string("verify: end capture: ") + strata::gpu::last_error();
        return false;
    }
    const bool synced = strata::gpu::stream_sync(cs_);
    std::fprintf(stderr, "strata verify: captured the %d-token window with every expert resident (sync %s)\n", T,
                 synced ? "ok" : strata::gpu::last_error());
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (!strata::gpu::begin_capture(cs_)) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // STRATA_COMMIT_BATCH (default on; =0 keeps the per-layer loop, upstream 6f3e3810): every GDN layer's conv
        // history and state in two launches, and every QSA layer's tail restore and indexer append in one, instead of
        // two or three per layer.  The same arithmetic per layer, bitwise.  The commit runs between a window and the
        // draft round.
        static const bool commit_batch = [] {
            const char* e = std::getenv("STRATA_COMMIT_BATCH");
            return e == nullptr || std::strtol(e, nullptr, 10) != 0;
        }();
        bool gdn_batched = false, qsa_batched = false;
        if (commit_batch) {
            int64_t n_gdn = 0;
            for (int64_t l = lb_; l < le_ && ok; ++l)   // the per-layer loop's check of each layer's ssm_norm.weight
                if (!is_qsa_layer(g, l)) {
                    ++n_gdn;
                    ok = need(LayerView(*wt_, l), "ssm_norm.weight", err) != nullptr;
                }
            if (ok && n_gdn > 0)
                gdn_batched = gdn_commit_layers(
                    ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats, (int64_t) gdn_floats,
                    (int64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size, qkv_L_ + (size_t) gdn_index * MT * C,
                    MT * C, h_L_ + (size_t) gdn_index * MT * C, MT * C, gate_L_ + (size_t) gdn_index * MT * HV,
                    beta_L_ + (size_t) gdn_index * MT * HV, MT * HV, (int) C, (int) g.ssm_k_heads, (int) HV, (int) MT,
                    (int) n_gdn, commit_, cs_);
        }
        if (commit_batch && ok && !no_batch_kv()) {
            QsaIndexerLayers il;
            int nq = 0;
            int64_t mc = -1;
            bool fits = true;
            for (int64_t l = lb_, qi = qsa_index; l < le_ && fits; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const WeightRef* wikn = need(LayerView(*wt_, l), "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                const QsaState& st = ss.qsa_states[qi];
                if (nq >= QsaIndexerLayers::kMax || (mc >= 0 && st.max_cells != mc)) { fits = false; break; }
                mc = st.max_cells;
                il.gamma[nq] = (const float*) wikn->data;
                il.tail_src[nq] = tail_snap_ + (size_t) qi * TS;
                il.tail[nq] = st.idx_tail;
                il.dead[nq] = st.idx_dead;
                il.pooled[nq] = st.idx_pooled;
                il.block_pos[nq] = st.idx_block_pos;
                ++nq;
                ++qi;
            }
            if (ok && fits && nq > 0) {
                native_qsa_indexer_append_steps_layers(idx_raw_L_ + (size_t) qsa_index * MT * ID, MT * ID, commit_ + 2, 1,
                                                       (int) MT, 0, EPS, il, nq, s, mc, rope_scaling(), cs_);
                qsa_batched = true;
            }
        }
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (is_qsa_layer(g, l) ? qsa_batched : gdn_batched) {
                ++(is_qsa_layer(g, l) ? qsa_index : gdn_index);
            } else if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_,
                                    (int) MT);   // the state only: no token's output is read (upstream 22abb92d)
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                if (!no_batch_kv())
                    native_qsa_indexer_append_steps(idx_raw_L_ + (size_t) qsa_index * MT * ID, commit_ + 2, 1, (int) MT,
                                                    0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                    rope_scaling(), cs_);
                else
                for (int64_t t = 0; t < MT; ++t)
                    native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                              rope_scaling(), cs_);
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        return false;
    }
    if (!strata::gpu::end_capture(cs_, &commit_exec_)) {
        err = std::string("verify: commit capture: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    homelab_profile::Window hp_window(T, 1);
    homelab_profile::Span hp_run("verify.run");
    const OnDevice on_device(device_);
    last_batch_ = false;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) {
        err = "verify: the window runs past the context";
        return false;
    }
    const bool res = res_graph_ && all_resident();
    homelab_profile::Span hp_capture("verify.capture");
    if (!(res ? capture_res(T, err) : capture(T, err)) || !capture_commit(err)) return false;
    hp_capture.end();
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) (pos0 + t);
    }
    const bool do_ple = ss.ple.ready() && ple_stage();
    const bool ple_late = do_ple && !res && lb_ == 0;   // read after the launch below (record_window's ple_late)
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];
    if (do_ple) {
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, ple_rows + (size_t) t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(ple_rows + (size_t) t * PLE_N_HEADS);   // the rows the drafter did not read ahead
        }
        homelab_profile::Span hp_ple("verify.ple_gather_early");
        if (!ple_late && !ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
    homelab_profile::Span hp_submit("verify.graph_submit");
    prof_mark(0);
    const bool le = strata::gpu::graph_launch(res ? exec_res_[T] : segmented_ ? segs_[T][0] : exec_[T], cs_);
    if (!le) { err = std::string("verify: launch: ") + strata::gpu::last_error(); return false; }
    hp_submit.end();
    homelab_profile::Span hp_idle("verify.stream_idle");
    (void) strata::gpu::stream_idle(cs_);
    hp_idle.end();
    VDBG("launched\n");
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    if (ple_late) {   // while the GPU runs layer 0 up to its experts; layer 1 waits for layer 0's flag
        homelab_profile::Span hp_ple("verify.ple_gather_late");
        const Clock::time_point tp = Clock::now();
        if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) {
            // the window is running: an empty plan and every flag raised let it finish before the error returns
            sink_.counts[0] = sink_.counts[1] = sink_.counts[2] = 0;
            sink_.start[0] = sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = UINT32_MAX;
            raise_flag(h_flagB_, UINT32_MAX);
            *flag = UINT32_MAX;
            strata::gpu::stream_sync(cs_);
            strata::gpu::stream_sync(copy_);
            return false;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ms_host += ms_since(tp);
    }
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    for (int64_t k = 0; !res && k < (le_ - lb_) * G; ++k) {   // the resident graph rings nothing
        const int64_t l = lb_ + k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        homelab_profile::Span hp_wait("verify.marker_wait", (int) l);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        if (segmented_ && !strata::gpu::stream_sync(cs_)) {   // the segment that rings `want` has run
            err = std::string("verify: segment ") + std::to_string(k) + ": " + strata::gpu::last_error();
            return false;
        }
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const strata::gpu::Status st = strata::gpu::stream_status(cs_);
                if (st == strata::gpu::Status::Failed) {
                    err = "verify: layer " + std::to_string(l) + ": " + strata::gpu::last_error();
                    return false;
                }
                if (st == strata::gpu::Status::Done && *seq < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (graph finished)";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) { err = "verify: timed out at layer " + std::to_string(l); return false; }
        }
        const Clock::time_point b = Clock::now();
        hp_wait.end();
        homelab_profile::Span hp_pool("verify.pool_and_publish", (int) l);
        VDBG("layer %lld rang\n", (long long) l);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window: the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        VDBG("layer %lld served\n", (long long) l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        if (segmented_) {   // the PCIe share's copies too, then the next segment
            if (!strata::gpu::stream_sync(copy_) || *(volatile uint32_t*) h_flagB_ < want) {
                err = "verify: the PCIe share of layer " + std::to_string(l) + " did not land";
                return false;
            }
            if (!strata::gpu::graph_launch(segs_[T][(size_t) k + 1], cs_)) {
                err = std::string("verify: launch segment: ") + strata::gpu::last_error();
                return false;
            }
        }
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
        hp_pool.end();
    }
    prof_mark(1);
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    homelab_profile::Span hp_finish("verify.final_compute_sync");
    const bool se = strata::gpu::stream_sync(cs_);
    hp_finish.end();
    if (!se) { err = std::string("verify: ") + strata::gpu::last_error(); return false; }
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    homelab_profile::Span hp_copy("verify.final_copy_sync");
    strata::gpu::stream_sync(copy_);
    hp_copy.end();   // no host function of this window may raise flag B in the next one
    if (prof_on_ && G == 1) {       // the window's GPU stage stamps
        strata::gpu::copy(prof_h_.data(), prof_, prof_h_.size() * 8);
        accumulate_profile(prof_h_.data());
    }
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        if (next_ == nullptr) return true;
        next_->rows(hand_out_, 0, T);
        return next_->run(T, tokens, pos0, pool, out, err);
    }
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (!strata::gpu::stream_sync(cs_)) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            strata::gpu::copy(h.data(), head_logits_, h.size() * 4);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "strata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

void Verifier::add_routing(float* usage, int64_t n_expert, int rows) const {
    if (h_route_ == nullptr || !route_rec_) return;
    const int64_t K = ss_->k;
    rows = std::min(rows, last_t_);
    for (int64_t l = lb_; l < le_; ++l)
        for (int t = 0; t < rows; ++t) {
            const volatile int32_t* ids = h_route_ + ((size_t) l * (size_t) max_t_ + (size_t) t) * (size_t) K;
            for (int64_t j = 0; j < K; ++j) {
                const int32_t e = ids[j];
                if (e >= 0 && e < n_expert) usage[(size_t) l * (size_t) n_expert + (size_t) e] += 1.0f;
            }
        }
}

void Verifier::accumulate_profile(const unsigned long long* stamps) {
    const ModelGeometry& g = *g_;
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return stamps[(size_t) (l * kProfPer + i)]; };
    // The derived columns read stamps the hc-read kernels write themselves or the next layer's first.  A slot no
    // kernel stamped is 0 and its unsigned difference wrapped to ~1e19 ns; a missing or out-of-order stamp now
    // contributes nothing (upstream e5b47dd).
    const auto gap = [](unsigned long long to, unsigned long long from) {
        return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
    };
    // only this stage's layers [lb_, le_) are ever stamped (a layer split); the head only on the last stage
    for (int64_t l = lb_; l < le_; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        if (l + 1 < le_) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
        const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
        if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
            prof_sum_[kind][27] += dn;      // hc-read0: norm
            prof_sum_[kind][28] += dd;      //           down
            prof_sum_[kind][29] += du;      //           up (through both halves, as before)
            prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
        }
    }
    if (le_ == L) prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
    // the window's rate: its start and end stamps against its profiling tags (both taken before the stamps are read)
    float span = 0;
    const double ticks = gap(at(L, 3), at(L, 2));
    if (ticks > 0 && strata::gpu::event_elapsed_ms(&span, prof_ev_[0], prof_ev_[1]) && span > 0) {
        prof_ticks_ += ticks;
        prof_ns_ += (double) span * 1e6;
    }
    ++prof_windows_;
}

void Verifier::prof_mark(int end) {
    if (!prof_on_) return;
    const int i = (int) (g_->n_layers * kProfPer + 2 + end);
    if (end) strata::kernels::gpu_stamp(prof_, i, cs_);
    prof_tag_ok_[end] = strata::gpu::event_record(prof_ev_[end], cs_);
    if (!end) strata::kernels::gpu_stamp(prof_, i, cs_);
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int G = last_batch_ ? 1 : (groups_[last_t_] > 0 ? groups_[last_t_] : 1);
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    for (int i = 0; i < n; ++i) strata::gpu::copy_async(stage + (size_t) i * bytes, src[i], bytes, v->copy_);
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    FlagSet* set = &fs;
    strata::gpu::launch_host(v->copy_, [set] { raise_flag(set->flag, set->value); });
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

bool Verifier::read_logits(int rows, float* host, std::string& err) const {
    if (le_ < g_->n_layers) return next_ != nullptr && next_->read_logits(rows, host, err);
    if (!strata::gpu::copy(host, head_logits_, (size_t) rows * (size_t) n_vocab_ * 4)) {
        err = "verify: reading the head's logits back failed";
        return false;
    }
    return true;
}

bool Verifier::wait_commit(std::string& err) {
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        if (!strata::gpu::event_sync(commit_done_)) {
            err = std::string("verify: commit: ") + strata::gpu::last_error();
            return false;
        }
    }
    return next_ == nullptr || next_->wait_commit(err);
}

// ================================ BATCH WINDOWS (see init_slots; upstream PR #559) ================================

bool Verifier::init_slots(const std::vector<SessionState*>& slots, std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    if (segmented_) { err = "verify: batch windows need a GPU whose running kernels see the host's writes"; return false; }
    if (slots.empty()) {
        err = "verify: init_slots needs at least one session";
        return false;
    }
    for (SessionState* x : slots) {
        if (x == nullptr) { err = "verify: a slot's session is null"; return false; }
        if (!session_slot_compatible(*g_, *ss_, *x, err)) return false;
    }
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    if (!mapped((size_t) (S * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    const uint64_t a = ((uint64_t) S * CB * 4 + 255) & ~255ull;
    if (!strata::gpu::alloc_device(&arena_b_, a + (uint64_t) S * std::max<int64_t>(nQ, 1) * TS * 4)) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    commitb_ = (int32_t*) arena_b_;
    tail_snap_b_ = reinterpret_cast<float*>(static_cast<uint8_t*>(arena_b_) + a);
    slots_ = slots;
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    std::fprintf(stderr, "strata verify: batch windows of up to %lld sequences\n", (long long) S);
    return true;
}

bool Verifier::enable_batch_outbound(std::string& err) {
    if (!BatchOutbound::enabled() || last_stage()) return true;
    const OnDevice on_device(device_);
    if (slots_.empty() || !next_ || !exec_bm_.empty() ||
        !outbound_.init(hand_out_, handoff_rows_ > 0 ? handoff_capacity() : slots_.size(),
                        (size_t) handoff_floats(*g_))) {
        err = std::string("verify: async batch outbound initialization: ") + strata::gpu::last_error();
        return false;
    }
    std::fprintf(stderr, "strata verify: async batch outbound enabled, %zu group-owned hand-off rows\n", slots_.size());
    return true;
}

int Verifier::batch_outbound_poll(int base, std::string& err) {
    const OnDevice on_device(device_);
    const int r = outbound_.poll(base);
    if (r < 0) err = r == -2 ? "verify: batch outbound copy timed out" :
                             std::string("verify: batch outbound copy: ") + strata::gpu::last_error();
    if (r != 1) return r;
    // rows() completes the host-to-host copy into the receiver's own context's host USM.  Only then may its
    // graph consume the group.  No foreign-context event, or source device pointer, is handed to the receiver.
    next_->rows(hand_out_, base, outbound_.count(base));
    if (!outbound_.acknowledge(base)) { err = "verify: batch outbound acknowledgement failed"; return -1; }
    return 1;
}

bool Verifier::capture_batch(const int* rows, int S, int hbase, std::string& err, bool res) {
    strata::gpu::Graph*& ex = exec_bm_[batch_key(rows, S, hbase, res)];
    if (ex != nullptr) return true;
    if (!strata::gpu::begin_capture(cs_)) { err = "verify: begin batch capture failed"; return false; }
    batch_rec_ = true;
    recording_res_ = res;   // STRATA_BATCH_RESIDENT_GRAPH: the device plans every layer, nothing rings the host
    row_base_ = hbase;
    for (int t = 0; t < S; ++t) brow_[t] = rows[t];
    std::string rerr;
    bool ok = false;
    try {
        ok = record_window(S, cs_, rerr);
    } catch (const std::exception& e) {
        rerr = std::string("verify: batch capture: ") + e.what();
    }
    batch_rec_ = false;
    recording_res_ = false;
    row_base_ = 0;
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        err = rerr;
        return false;
    }
    if (!strata::gpu::end_capture(cs_, &ex)) {
        err = std::string("verify: end batch capture: ") + strata::gpu::last_error();
        return false;
    }
    const bool synced = strata::gpu::stream_sync(cs_);
    std::string list;
    for (int t = 0; t < S; ++t) list += (t ? "," : "") + std::to_string(rows[t]);
    std::fprintf(stderr, "strata verify: captured the batch window over slots %s (sync %s; %zu graph nodes, layers %lld..%lld)\n",
                 list.c_str(), synced ? "ok" : strata::gpu::last_error(), strata::gpu::graph_nodes(ex), (long long) lb_,
                 (long long) le_ - 1);
    return true;
}

bool Verifier::capture_commit_batch(const int* rows, int S, int hbase, std::string& err, bool res) {
    strata::gpu::Graph*& cex = commit_bm_[batch_key(rows, S, hbase, res)];
    if (cex != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ZV = g.ssm_value_dim, ID = g.idx_key_dim, MT = max_t_;
    const int64_t CB = 2 + MT, nQ = g.n_qsa_layers();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (!strata::gpu::begin_capture(cs_)) { err = "verify: begin batch commit capture failed"; return false; }
    bool ok = true;
    try {
        for (int t = 0; t < S; ++t)
            if (t == 0 || rows[t] != rows[t - 1])
                copy_i32_from_mapped(commitb_ + (size_t) rows[t] * CB, m_commitb_ + (size_t) rows[t] * CB, CB, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    SessionState& sx = *slots_[(size_t) rows[first]];
                    const int32_t* keep = commitb_ + (size_t) rows[first] * CB;
                    float* state = sx.gdn_state + (size_t) (gdn_index - sx.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    gdn_conv_commit(conv, qkv_L_ + (size_t) gdn_index * MT * C + (size_t) first * C, (int) C, keep, cs_);
                    gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C + (size_t) first * C, (int) C,
                                        gate_L_ + (size_t) gdn_index * MT * HV + (size_t) first * HV,
                                        beta_L_ + (size_t) gdn_index * MT * HV + (size_t) first * HV,
                                        z_ + (size_t) first * ZV, (const float*) wnm->data, EPS,
                                        y_dummy_ + (size_t) first * ZV, (int) g.ssm_k_heads, (int) HV, t - first, keep,
                                        cs_, t - first > 1 ? t - first : 0);   // a 1-row group: the plain batch commit
                }
                ++gdn_index;
            } else {
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    const QsaState& st = slots_[(size_t) rows[first]]->qsa_states[qsa_index];
                    copy_from_mapped(st.idx_tail, tail_snap_b_ + ((size_t) rows[first] * nQ + qsa_index) * TS, TS, cs_);
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    for (int u = first; u < t; ++u)
                        native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + u) * ID,
                                                  commitb_ + (size_t) rows[first] * CB + 2 + (u - first), 0,
                                                  (const float*) wikn->data, EPS, ib, s, st.max_cells, rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss_->ple.ready() && ple_stage())
            for (int t = 0; t < S;) {
                const int first = t;
                while (t < S && rows[t] == rows[first]) ++t;
                copy_indexed(slots_[(size_t) rows[first]]->ple_hist, hist_snap_ + (size_t) first * HS, HS,
                             commitb_ + (size_t) rows[first] * CB + 1, HS, cs_);
            }
    } catch (const std::exception& e) {
        err = std::string("verify batch commit: ") + e.what();
        ok = false;
    }
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        return false;
    }
    if (!strata::gpu::end_capture(cs_, &cex)) {
        err = std::string("verify: batch commit capture: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    int rows[strata::kernels::kVerifyMaxT] = {};
    for (int t = 0; t < S && t < strata::kernels::kVerifyMaxT; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}

bool Verifier::stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos,
                           std::string& err, const int* keep) {
    homelab_profile::Span hp_stage("verify.stage_batch");
    using namespace strata::kernels;
    // a slot may own a contiguous group of rows (--batch-mtp), also across the stages of a layer split, whose hand-off
    // buffers hold max(selected window capacity, total slots) rows, indexed by global slot base
    if (S < 1 || S > max_t_ || hbase < 0 || ((next_ != nullptr || lb_ > 0) && (size_t) hbase + (size_t) S > handoff_capacity())) {
        err = "verify: batch rows out of range (init_slots)";
        return false;
    }
    const ModelGeometry& g = *g_;
    for (int t = 0; t < S; ++t) {
        if (rows[t] < 0 || rows[t] >= (int) slots_.size()) {
            err = "verify: a batch row's slot is out of range";
            return false;
        }
        for (int u = 0; u < t - 1; ++u)
            if (rows[u] == rows[t] && rows[t - 1] != rows[t]) {
                err = "verify: a slot's proposed rows must be contiguous";
                return false;
            }
        if (t > 0 && rows[t] == rows[t - 1] && pos[t] != pos[t - 1] + 1) {
            err = "verify: proposed rows must have consecutive positions";
            return false;
        }
        if (pos[t] < 0 || pos[t] >= slots_[(size_t) rows[t]]->max_cells) {
            err = "verify: slot " + std::to_string(rows[t]) + " runs past its context";
            return false;
        }
    }
    // STRATA_BATCH_RESIDENT_GRAPH: every expert of this stage's layers in VRAM (a layer split that holds the whole
    // model, or a card that does): the zero-doorbell graph.  STRATA_BATCH_SELF_COMMIT: one row a slot keeps itself.
    const bool res = batch_res_graph_on() && hits_.h_res != nullptr && all_resident();
    const bool self = batch_self_commits(rows, S);
    // --batch-mtp (a limit): slot rotation creates new layouts; bound the captured graph pairs, evicting the least
    // recently used layout.  Without a limit every layout is kept.
    const std::vector<int> key = batch_key(rows, S, hbase, res);
    if (batch_graph_limit_ > 0) {
        if (exec_bm_.find(key) == exec_bm_.end() && exec_bm_.size() >= batch_graph_limit_) {
            if (!strata::gpu::stream_sync(cs_)) {
                err = "verify: synchronizing before batch graph eviction failed";
                return false;
            }
            auto old = exec_bm_.begin();
            uint64_t oldest = UINT64_MAX;
            for (auto it = exec_bm_.begin(); it != exec_bm_.end(); ++it) {
                const auto u = bm_used_.find(it->first);
                const uint64_t used = u == bm_used_.end() ? 0 : u->second;
                if (used < oldest) { oldest = used; old = it; }
            }
            const std::vector<int> old_key = old->first;
            if (old->second) strata::gpu::graph_destroy(old->second);
            exec_bm_.erase(old);
            bm_used_.erase(old_key);
            const auto commit_old = commit_bm_.find(old_key);
            if (commit_old != commit_bm_.end()) {
                if (commit_old->second) strata::gpu::graph_destroy(commit_old->second);
                commit_bm_.erase(commit_old);
            }
        }
        bm_used_[key] = ++bm_tick_;
    }
    homelab_profile::Span hp_capture("verify.capture_batch");
    if (!capture_batch(rows, S, hbase, err, res) || (!self && !capture_commit_batch(rows, S, hbase, err, res))) return false;
    hp_capture.end();
    res_b_ = res;
    self_b_ = self;
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    // a pad row (keep 0, its slot's only row): attended at position 0 (one cell, a valid read), its K/V appended at
    // `pos` (the caller's: past the slot's history), no indexer append, and its slot's GDN and PLE histories kept
    const auto pad = [&](int t) {
        return keep != nullptr && keep[rows[t]] == 0 && (t == 0 || rows[t - 1] != rows[t]) &&
               (t + 1 == S || rows[t + 1] != rows[t]);
    };
    for (int t = 0; t < S; ++t) {
        const bool p = pad(t);
        const int64_t at = p ? 0 : pos[t];
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + (size_t) t * kStepCount, at, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) at;
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) at;
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) at;
        h_rowx_[t] = (int32_t) pos[t];
        h_rowx_[max_t_ + t] = p ? -1 : (int32_t) pos[t];
        h_rowx_[2 * max_t_ + t] = p ? 0 : 1;
    }
    if (ss_->ple.ready() && ple_stage()) {
        uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];   // (not `rows`: that is the slots of the window's rows)
        for (int t = 0; t < S; ++t) {
            const SessionState& sx = *slots_[(size_t) rows[t]];
            int32_t prev[2] = {sx.ple_prev[0], sx.ple_prev[1]};
            if (t > 0 && rows[t] == rows[t - 1]) {   // a proposal row follows its slot's row before it
                prev[0] = t > 1 && rows[t - 2] == rows[t] ? tokens[t - 2] : sx.ple_prev[1];
                prev[1] = tokens[t - 1];
            }
            ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + (size_t) t * PLE_N_HEADS);
        }
        ple_late_n_ = 0;
        *(volatile uint32_t*) (h_flagB_ + kPleFlagWord) = 0;
        if (lb_ == 0 && batch_ple_late_on()) {   // STRATA_BATCH_PLE_LATE: read after the launch (ple_late_publish)
            std::memcpy(ple_late_rows_, ple_rows, sizeof(uint32_t) * (size_t) S * PLE_N_HEADS);
            ple_late_n_ = S;
        } else {
            homelab_profile::Span hp_ple("verify.ple_gather_batch");
            if (!ss_->ple.table->gather_batch(ple_rows, (size_t) S, h_ple_, err)) return false;
        }
    }
    // the commit's rows: each slot owns a contiguous group, all of it kept unless `keep` (window-row prefill's padded
    // run) or commit_slot_prefixes says less
    const int64_t CB = 2 + max_t_;
    for (int t = 0; t < S;) {
        const int first = t;
        while (t < S && rows[t] == rows[first]) ++t;
        const int n = keep != nullptr ? keep[rows[first]] : t - first;
        if (n < (t - first == 1 ? 0 : 1) || n > t - first) {   // 0: a pad row (its slot's only row)
            err = "verify: a slot keeps more rows than it has, or none";
            return false;
        }
        if (n == 0 && !pad_rows_harmless()) {
            err = "verify: a pad row (keep 0) needs STRATA_BATCH_GDN_SLOTS with STRATA_BATCH_SELF_COMMIT";
            return false;
        }
        int32_t* c = h_commitb_ + (size_t) rows[first] * CB;
        c[0] = n;
        c[1] = n - 1;
        for (int64_t j = 0; j < max_t_; ++j) c[2 + j] = j < n ? (int32_t) pos[first + j] : -1;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = S;
    row_base_ = hbase;   // commit_slots' graph key until the next stage_batch
    last_batch_ = true;
    for (int t = 0; t < S; ++t) { last_tokens_[t] = tokens[t]; last_pos_b_[t] = pos[t]; last_rows_[t] = rows[t]; }
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::ple_late_publish(std::string& err) {
    if (ple_late_n_ == 0) return true;
    homelab_profile::Span hp_ple("verify.ple_gather_batch_late");
    const int n = ple_late_n_;
    ple_late_n_ = 0;
    const bool ok = ss_->ple.table->gather_batch(ple_late_rows_, (size_t) n, h_ple_, err);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    *(volatile uint32_t*) (h_flagB_ + kPleFlagWord) = 1u;   // also on a failed read: the window must not hang
    return ok;
}

bool Verifier::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                             void* user, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    homelab_profile::Window hp_window(S, S);
    homelab_profile::Span hp_run("verify.run");
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    if (outbound_.device_rows() && !outbound_.writable(0, S)) {
        err = "verify: batch outbound rows still owned by a transfer";
        return false;
    }
    if (!stage_batch(rows, S, 0, tokens, pos, err)) return false;
    homelab_profile::Span hp_submit("verify.graph_submit");
    prof_mark(0);
    if (!strata::gpu::graph_launch(exec_bm_[batch_key(rows, S, 0, res_b_)], cs_)) {
        err = std::string("verify: batch launch: ") + strata::gpu::last_error();
        return false;
    }
    prof_mark(1);
    hp_submit.end();
    if (!ple_late_publish(err)) return false;
    homelab_profile::Span hp_idle("verify.stream_idle");
    (void) strata::gpu::stream_idle(cs_);
    hp_idle.end();
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    for (int64_t k = 0; !res_b_ && k < le_ - lb_; ++k) {   // the resident graph rings nothing
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        homelab_profile::Span hp_wait("verify.marker_wait", (int) l);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify batch: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const strata::gpu::Status st = strata::gpu::stream_status(cs_);
                if (st == strata::gpu::Status::Failed) {
                    err = "verify batch: layer " + std::to_string(l) + ": " + strata::gpu::last_error();
                    return false;
                }
                if (st == strata::gpu::Status::Done && *seq < want) {
                    err = "verify batch: layer " + std::to_string(l) + " never rang (graph finished)";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) { err = "verify batch: timed out at layer " + std::to_string(l); return false; }
        }
        const Clock::time_point b = Clock::now();
        hp_wait.end();
        homelab_profile::Span hp_pool("verify.pool_and_publish", (int) l);
        cur_layer_ = want - 1;
        set_plan_slot(0);
        progress_at("verify batch: the CPU experts of layer", l);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
        hp_pool.end();
    }
    homelab_profile::Span hp_finish("verify.final_compute_sync");
    if (!strata::gpu::stream_sync(cs_)) { err = std::string("verify batch: ") + strata::gpu::last_error(); return false; }
    hp_finish.end();
    homelab_profile::Span hp_copy("verify.final_copy_sync");
    strata::gpu::stream_sync(copy_);
    hp_copy.end();   // no host function of this window may raise flag B in the next one
    ++windows;
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the rows go on to the next stage, as in run()
        if (next_ == nullptr) return true;
        if (outbound_.device_rows()) {   // a synchronous caller may replay the same batch graph: keep its contract
            if (!outbound_.start(0, S, cs_) || !outbound_.wait(0) || batch_outbound_poll(0, err) != 1) {
                if (err.empty()) err = std::string("verify: synchronous batch outbound copy: ") + strata::gpu::last_error();
                return false;
            }
        } else {
            next_->rows(hand_out_, 0, S);
        }
        return next_->run_slot_rows(rows, S, tokens, pos, pool, out, err);
    }
    homelab_profile::Span hp_sample("verify.sample_rows");
    if (!sample_rows(S, err)) return false;
    hp_sample.end();
    for (int t = 0; t < S; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(S, last_pos_b_);
    progress_at("decode");
    progress_beat();
    return true;
}

bool Verifier::commit_slots(std::string& err) {
    std::vector<int> keep(slots_.size(), 0);
    for (int t = 0; t < last_t_; ++t) ++keep[(size_t) last_rows_[t]];
    return commit_slot_prefixes(keep.data(), err);
}

bool Verifier::commit_slot_prefixes(const int* keep, std::string& err) {
    homelab_profile::Span hp_commit("verify.commit_prefixes");
    const OnDevice on_device(device_);
    if (!last_batch_ || last_t_ < 1) { err = "verify: commit_slots without a batch window"; return false; }
    const int S = last_t_;
    const Clock::time_point t0 = Clock::now();
    const int64_t CB = 2 + max_t_;
    for (int t = 0; t < S;) {   // one prefix a slot: rejected draft rows must not enter the recurrent state
        const int first = t;
        while (t < S && last_rows_[t] == last_rows_[first]) ++t;
        const int n = keep[last_rows_[first]];
        if (n < 1 || n > t - first) {
            err = "verify: an accepted prefix outside its slot group";
            return false;
        }
        int32_t* c = h_commitb_ + (size_t) last_rows_[first] * CB;
        c[0] = n;
        c[1] = n - 1;
        for (int j = 0; j < max_t_; ++j) c[2 + j] = j < n ? (int32_t) last_pos_b_[first + j] : -1;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // STRATA_BATCH_SELF_COMMIT: the window kept its rows (every keep is 1, checked above): no graph, no sync
    if (!self_b_ && (!strata::gpu::graph_launch(commit_bm_[batch_key(last_rows_, S, row_base_, res_b_)], cs_) ||
                     !strata::gpu::stream_sync(cs_))) {
        err = std::string("verify: batch commit: ") + strata::gpu::last_error();
        return false;
    }
    if (ple_stage())
        for (int t = 0; t < S;) {
            const int first = t;
            while (t < S && last_rows_[t] == last_rows_[first]) ++t;
            SessionState& sx = *slots_[(size_t) last_rows_[first]];
            for (int u = first; u < first + keep[last_rows_[first]]; ++u) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = last_tokens_[u];
            }
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit_slot_prefixes(keep, err);
}

// the one-launch GDN forms skip a pad row's commit (GdnSlotTable::keep); the commit graph's kernels keep nothing
// for n_keep 0; only the per-slot self-commit (gdn_conv_l2_multi's bool) cannot skip a row
bool Verifier::pad_rows_harmless() { return !batch_self_commit_on() || batch_gdn_slots_on(); }

bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    int rows[strata::kernels::kVerifyMaxT] = {};
    for (int t = 0; t < S && t < strata::kernels::kVerifyMaxT; ++t) rows[t] = base + t;
    return batch_launch_rows(rows, S, base, tokens, pos, nullptr, err);
}

bool Verifier::batch_launch_rows(const int* rows, int S, int base, const int32_t* tokens, const int64_t* pos,
                                 const int* keep, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    if (outbound_.device_rows() && !outbound_.writable(base, S)) {
        err = "verify: batch outbound rows still owned by a transfer";
        return false;
    }
    const bool tl = (batch_timeline() > 0 || batch_raw_windows());
    auto ns_now = [] { return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); };
    if (tl) {
        bt_cur_ = BatchTimes{};
        bt_cur_.rows = S;
        bt_cur_.out_scope = outbound_.device_rows() ? "device-pack" : "mapped-host";
        bt_cur_.h_begin = ns_now();
        for (int i = 0; i < 2; ++i)
            if (tl_ev_[i] == nullptr && !strata::gpu::event_create(&tl_ev_[i], true)) tl_ev_[i] = nullptr;
    }
    if (!stage_batch(rows, S, base, tokens, pos, err, keep)) return false;
    // the commit right behind the window: every row of a batch window is kept (or its slot's `keep`)
    const strata::gpu::Graph* bg = exec_bm_[batch_key(rows, S, base, res_b_)];
    if (tl) {
        // Clear both hand-off and calibration pairs before this replay; a missing stamp must not reuse last
        // window's value.  This reset is queued before the timeline's start tag, outside the measured device span.
        if (prof_on_ && !last_stage() &&
            !strata::gpu::memset_async(prof_ + g_->n_layers * kProfPer, 0, 4 * sizeof(*prof_), cs_)) {
            err = std::string("verify: hand-off timeline reset: ") + strata::gpu::last_error();
            return false;
        }
        bt_cur_.h_submit = ns_now();
        bt_cur_.nodes = strata::gpu::graph_nodes(bg);
        bt_cur_.dev = tl_ev_[0] != nullptr && tl_ev_[1] != nullptr && strata::gpu::event_record(tl_ev_[0], cs_);
    }
    prof_mark(0);
    const bool le = strata::gpu::graph_launch(bg, cs_);
    prof_mark(1);
    if (!le || (!self_b_ && !strata::gpu::graph_launch(commit_bm_[batch_key(rows, S, base, res_b_)], cs_))) {
        err = std::string("verify: batch launch: ") + strata::gpu::last_error();
        return false;
    }
    if (tl) {
        bt_cur_.dev = bt_cur_.dev && strata::gpu::event_record(tl_ev_[1], cs_);
        bt_cur_.h_ret = ns_now();
    }
    if (!ple_late_publish(err)) return false;
    (void) strata::gpu::stream_idle(cs_);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens): the kept rows only
        for (int t = 0; t < S;) {
            const int first = t;
            while (t < S && rows[t] == rows[first]) ++t;
            const int n = keep != nullptr ? keep[rows[first]] : t - first;
            SessionState& sx = *slots_[(size_t) rows[first]];
            for (int u = first; u < first + n; ++u) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = tokens[u];
            }
        }
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = res_b_ ? 0 : le_ - lb_;   // the resident graph rings nothing
    b_last_ = Clock::now();
    return true;
}

int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    const OnDevice on_device(device_);
    volatile uint32_t* const seq = h_seq_;
    const int S = last_t_;
    while (b_k_ < b_steps_) {
        const uint32_t want = (uint32_t) (b_k_ + 1);
        if (*seq < want) {
            const auto now = Clock::now();
            if (now - b_last_ > std::chrono::milliseconds(2)) {
                const strata::gpu::Status st = strata::gpu::stream_status(cs_);
                if (st == strata::gpu::Status::Failed) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + ": " + strata::gpu::last_error();
                    b_running_ = false;
                    return -1;
                }
                if (st == strata::gpu::Status::Done && *seq < want) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (graph finished)";
                    b_running_ = false;
                    return -1;
                }
                if (now - b_last_ > std::chrono::seconds(20)) {
                    err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_);
                    b_running_ = false;
                    return -1;
                }
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot(0);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + b_k_);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ms_pool += ms_since(b);
        b_last_ = Clock::now();
        ++b_k_;
    }
    strata::gpu::Status st = strata::gpu::stream_status(cs_);
    if (st == strata::gpu::Status::Done) st = strata::gpu::stream_status(copy_);   // (copy_: no host function of this
    if (st == strata::gpu::Status::Failed) {                                         // window may raise flag B in the next)
        err = std::string("verify batch: the window's commit: ") + strata::gpu::last_error();
        b_running_ = false;
        return -1;
    }
    if (st == strata::gpu::Status::Pending) {
        if (Clock::now() - b_last_ > std::chrono::seconds(20)) {
            err = "verify batch: the window's commit did not finish";
            b_running_ = false;
            return -1;
        }
        return 0;
    }
    if ((batch_timeline() > 0 || batch_raw_windows())) {   // STRATA_BATCH_TIMELINE: this window's times, the device's from its tags
        bt_cur_.h_notice = (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
        bt_cur_.dev = bt_cur_.dev && strata::gpu::event_time_ns(tl_ev_[0], &bt_cur_.d_start) &&
                      strata::gpu::event_time_ns(tl_ev_[1], &bt_cur_.d_end);
    }
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
        note_nonfinite(S, last_pos_b_);
    } else if (next_ != nullptr && !outbound_.device_rows()) {   // the next stage's rows of this group
        next_->rows(hand_out_, row_base_, S);
    }
    if (prof_on_) {   // STRATA_VERIFY_PROFILE=1: the batch window's stage stamps (a blocking copy: profiling runs only)
        const bool copied = strata::gpu::copy(prof_h_.data(), prof_, prof_h_.size() * 8);
        if (copied) accumulate_profile(prof_h_.data());
        if ((batch_timeline() > 0 || batch_raw_windows()) && !last_stage()) {
            bt_cur_.out_clock = strata::kernels::gpu_stamp_clock();
            if (copied) {
                const size_t i = (size_t) g_->n_layers * kProfPer;
                float ms = 0;
                // These tags bracket this window, not an accumulated or another device's clock rate.
                if (!prof_tag_ok_[0] || !prof_tag_ok_[1] ||
                    !strata::gpu::event_elapsed_ms(&ms, prof_ev_[0], prof_ev_[1])) ms = 0;
                bt_cur_.set_handoff(prof_h_[i + 2], prof_h_[i], prof_h_[i + 1], prof_h_[i + 3], ms);
            }
        }
    }
    if ((batch_timeline() > 0 || batch_raw_windows())) bt_last_ = bt_cur_;
    if (outbound_.device_rows() && !outbound_.start(row_base_, S, cs_)) {
        err = std::string("verify: batch outbound submission: ") + strata::gpu::last_error();
        b_running_ = false;
        return -1;
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return outbound_.device_rows() ? 2 : 1;   // compute free is NOT group delivery
}

void Verifier::BatchTimes::set_handoff(uint64_t start, uint64_t out_start, uint64_t out_end, uint64_t end,
                                      double window_ms) {
    out_ticks = 0;
    out_ms = -1;
    if (start == 0 || out_start < start || out_end <= out_start || end < out_end) return;
    out_ticks = out_end - out_start;
    if (!std::isfinite(window_ms) || window_ms <= 0) return;
    // The clock can follow DVFS (sub_group in particular): this window's average rate is only an estimate for its
    // hand-off.  Keep the raw ticks and scope; never label this profiling-tag ns or compare ticks between devices.
    out_ms = window_ms * ((double) out_ticks / (double) (end - start));
}

bool Verifier::batch_raw_windows() {
    static const bool on = env_flag("STRATA_BATCH_RAW_WINDOWS", false);
    return on;
}

int Verifier::batch_timeline() {
    static const int every = [] {
        const char* e = std::getenv("STRATA_BATCH_TIMELINE");
        return e != nullptr ? std::max(0, std::atoi(e)) : 0;
    }();
    return every;
}

bool Verifier::device_clock_offset(double* host_minus_dev_ns) {
    const OnDevice on_device(device_);
    if (tl_q_ == nullptr && (tl_q_ = strata::gpu::stream_create(false)) == nullptr) return false;
    if (tl_ev_[2] == nullptr && !strata::gpu::event_create(&tl_ev_[2], true)) { tl_ev_[2] = nullptr; return false; }
    auto ns_now = [] { return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); };
    double best_rt = 1e30, best = 0;
    for (int r = 0; r < 7; ++r) {
        const double h0 = ns_now();
        if (!strata::gpu::event_record(tl_ev_[2], tl_q_) || !strata::gpu::stream_sync(tl_q_)) return false;
        const double h1 = ns_now();
        uint64_t d = 0;
        if (!strata::gpu::event_time_ns(tl_ev_[2], &d)) return false;
        if (h1 - h0 < best_rt) { best_rt = h1 - h0; best = 0.5 * (h0 + h1) - (double) d; }
    }
    *host_minus_dev_ns = best;
    return true;
}

bool Verifier::sample_rows(int S, std::string& err) {
    bool any = false;
    for (int t = 0; t < S; ++t) {
        strata::kernels::SamplerParams sp = slot_sp_[(size_t) last_rows_[t]];
        if (sp.greedy || sp.temperature <= 0.0f) continue;
        sp.counter = (uint64_t) last_pos_b_[t];   // Philox(seed, position): the solo window's draw for this position
        sp.penalty_last_n = 0;
        strata::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, sp,
                                       m_out_ + t, cs_);
        any = true;
    }
    if (any && !strata::gpu::stream_sync(cs_)) { err = "verify batch: the row sampling failed"; return false; }
    return true;
}

// #879: STRATA_NAN_GUARD=0 turns the guard off (no kernel is recorded; the graphs are captured once, so it is read once)
bool Verifier::nan_guard_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_NAN_GUARD");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    return on;
}

namespace {
// STRATA_NAN_INJECT=<n>[,<n>...] (debug): the windows (counted from 1 over the process) the guard reports as non-finite
const std::vector<long long>& nan_inject_list() {
    static const std::vector<long long> list = [] {
        std::vector<long long> v;
        if (const char* e = std::getenv("STRATA_NAN_INJECT"))
            for (const char* q = e; *q != 0;) {
                char* end = nullptr;
                const long long n = std::strtoll(q, &end, 10);
                if (end == q) break;
                if (n > 0) v.push_back(n);
                q = *end == ',' ? end + 1 : end;
            }
        return v;
    }();
    return list;
}
std::atomic<long long> g_nf_windows{0};
}  // namespace

void Verifier::note_nonfinite(int T, const int64_t* pos_b) {
    nf_row_ = -1;
    nf_pos_ = -1;
    nf_mask_ = 0;
    if (!nan_guard_on() || h_nf_ == nullptr || !last_stage()) return;   // (a stage without the head checks nothing)
    const volatile uint32_t* f = h_nf_;
    for (int t = 0; t < T && t < strata::kernels::kVerifyMaxT; ++t)
        for (int b = 0; b < strata::kernels::kNonfiniteBlocks; ++b)
            if (f[t * strata::kernels::kNonfiniteBlocks + b] != 0) { nf_mask_ |= 1u << t; break; }
    const std::vector<long long>& inject = nan_inject_list();
    if (!inject.empty()) {
        const long long k = g_nf_windows.fetch_add(1) + 1;
        if (nf_mask_ == 0 && std::find(inject.begin(), inject.end(), k) != inject.end()) {
            nf_mask_ = 1u;
            std::fprintf(stderr, "strata verify: STRATA_NAN_INJECT: window %lld reported as non-finite\n", k);
        }
    }
    for (int t = 0; t < T && nf_row_ < 0; ++t)
        if ((nf_mask_ >> t) & 1u) nf_row_ = t;
    if (nf_row_ >= 0) nf_pos_ = pos_b != nullptr ? pos_b[nf_row_] : last_pos0_ + nf_row_;
}

bool Verifier::copy_logits(int t, float* host) const {
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    return strata::gpu::copy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float));
}

bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) {
    if (next_ != nullptr) return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized its stream before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    if (!strata::gpu::copy(h.data(), head_logits_, h.size() * sizeof(float))) {
        err = "window_logprobs: the head logits copy failed";
        return false;
    }
    static const int topk = [] {
        const char* v = std::getenv("STRATA_LOGPOS_TOPK");
        return v != nullptr ? (int) std::clamp(std::strtol(v, nullptr, 10), 0L, 256L) : 0;
    }();
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab_; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;   // the second skips extra_id: no cancellation when it holds ~all mass
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(std::max(sum, 1.0));   // sum holds the top token's exp(0) = 1
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) pos0 + t, (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

namespace { bool g_commit_async = false; }
void Verifier::set_commit_async(bool on) { g_commit_async = on && std::getenv("STRATA_COMMIT_SYNC") == nullptr; }

bool Verifier::commit(int n_keep, std::string& err) {
    homelab_profile::Span hp_commit("verify.commit_solo_maybe_async");
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    if (last_t_ == 1 && !last_batch_ && !always_publish_ && one_token_self_commit()) {
        // a one-token window has advanced the state itself (record_window): no commit graph
        if (ple_stage())
            for (int t = 0; t < n_keep; ++t) {
                ss_->ple_prev[0] = ss_->ple_prev[1];
                ss_->ple_prev[1] = last_tokens_[t];
            }
        ms_commit += ms_since(t0);
        return next_ == nullptr || next_->commit(n_keep, err);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const bool le = strata::gpu::graph_launch(commit_exec_, cs_);
    if (!le) { err = std::string("verify: commit launch: ") + strata::gpu::last_error(); return false; }
    // set_commit_async: no wait here - the next window runs on the same queue after it, and the drafter (its own
    // queue) reads only this window's final rows and its own K/V.  h_commit_ is next written after the next window's
    // results are read, i.e. after this graph has run.  Everything else waits on commit_done_ (wait_commit, which
    // goes on into a layer split's later stages).  Each stage's commit graph writes its own layers' state on its own
    // queue, so a layer split's stages commit without a wait as well (upstream kept the wait there).
    if (!g_commit_async) {
        if (!strata::gpu::stream_sync(cs_)) { err = std::string("verify: commit: ") + strata::gpu::last_error(); return false; }
    } else {
        if (!strata::gpu::event_record(commit_done_, cs_)) {
            err = std::string("verify: commit event: ") + strata::gpu::last_error();
            return false;
        }
        commit_pending_ = true;
    }
    if (ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err);
}

// ================================ PIPELINED WINDOWS (see verify.hpp) ================================
//
// The window is run()'s, step for step: the same graph, the same staging, the same per-layer service, the same commit
// graph.  Only the waits are split up: the host polls instead of spinning, so it can serve the other stage's window in
// between.  A one-token window does not commit itself here (always_publish_): a speculative window may be rolled back.

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
}  // namespace

void Verifier::diag_pipelined(std::FILE* f, const char* name) const {
    // host-side state only: a GPU call from the watchdog's thread can wait behind a window that waits for the host
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    std::fprintf(f, "  %s: %s T=%d pos %lld served %lld/%lld; GPU rang %u, flags served %u A %u B %u; commit launched %d\n",
                 name, fl_active_ ? "IN FLIGHT" : "idle", last_t_, (long long) last_pos0_, (long long) fl_k_,
                 (long long) fl_total_, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_), (int) commit_live_);
}

bool Verifier::capture_all(std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr) { err = "verify: capture_all before init"; return false; }
    if (segmented_) { err = "verify: pipelined windows need a GPU whose running kernels see the host's writes"; return false; }
    if (!always_publish_) { err = "verify: pipelined windows need set_always_publish before init"; return false; }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    if (!capture_commit(err)) return false;
    if ((ev_done_ == nullptr && !strata::gpu::event_create(&ev_done_)) ||
        (ev_commit_ == nullptr && !strata::gpu::event_create(&ev_commit_))) {
        err = "verify: event create failed";
        return false;
    }
    if (prof_on_ && prof_pin_ == nullptr && !strata::gpu::alloc_host(&prof_pin_, prof_h_.size() * 8, "verify"))
        prof_pin_ = nullptr;   // the pipelined windows go unprofiled
    return true;
}

// The host staging of a pipelined window: run()'s, with the PLE rows from `ple_prev` (the window's verifier is idle,
// so its mapped rows are free to write)
bool Verifier::pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + (size_t) t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) (pos0 + t);
    }
    if (ss.ple.ready() && ple_stage()) {
        uint32_t rows[kVerifyMaxT * PLE_N_HEADS];
        int32_t prev[2] = {ple_prev[0], ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows + (size_t) t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
        if (!ss.ple.table->gather_batch(rows, (size_t) T, h_ple_, err)) return false;
    }
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    pl_prev_[0] = ple_prev[0];
    pl_prev_[1] = ple_prev[1];
    return true;
}

bool Verifier::prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err) {
    if (fl_active_) { err = "verify: a window is in flight on this verifier"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (!pl_stage(T, tokens, pos0, ple_prev, err)) return false;
    pl_prestaged_ = true;
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    if (fl_active_) { err = "verify: a window is already in flight on this verifier"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (exec_[T] == nullptr || commit_exec_ == nullptr || ev_done_ == nullptr) {
        err = "verify: pipelined window not prepared (capture_all)";
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    last_batch_ = false;
    bool staged = pl_prestaged_ && last_t_ == T && last_pos0_ == pos0;
    for (int t = 0; staged && t < T; ++t) staged = last_tokens_[t] == tokens[t];
    if (staged && ss.ple.ready() && ple_stage())
        staged = pl_prev_[0] == ss.ple_prev[0] && pl_prev_[1] == ss.ple_prev[1];
    pl_prestaged_ = false;
    if (!staged && !pl_stage(T, tokens, pos0, ss.ple_prev, err)) return false;
    // the previous commit's words (h_commit_) are rewritten by this window's: wait for that graph here, before this
    // window is launched, never in pl_commit_async behind it (on the B70 that wait did not return while the window
    // waited for this thread)
    if (commit_live_) {
        strata::gpu::Status st;
        while ((st = strata::gpu::event_status(ev_commit_)) == strata::gpu::Status::Pending) _mm_pause();
        if (st == strata::gpu::Status::Failed) {
            err = std::string("verify: the previous commit: ") + strata::gpu::last_error();
            return false;
        }
        commit_live_ = false;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    fl_T_ = T;
    fl_k_ = 0;
    fl_total_ = (le_ - lb_) * G;
    fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    ms_host += ms_since(t0);
    if (fl_prof_) prof_mark(0);
    if (!strata::gpu::graph_launch(exec_[T], cs_)) {
        err = std::string("verify: launch: ") + strata::gpu::last_error();
        return false;
    }
    if (fl_prof_) prof_mark(1);
    if (fl_prof_) strata::gpu::copy_async(prof_pin_, prof_, prof_h_.size() * 8, cs_);
    if (!strata::gpu::event_record(ev_done_, cs_)) { err = "verify: event record failed"; return false; }
    (void) strata::gpu::stream_idle(cs_);
    fl_active_ = true;
    fl_since_ms_ = fl_flush_ms_ = fl_launch_ms_ = now_ms();
    return true;
}

int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {
    if (!fl_active_ || fl_k_ >= fl_total_) return 1;
    // Nothing rang yet and neither the dead-graph check nor the timeout below is due: return before the device switch
    // (upstream 6b392641).  The pipelined loop asks both stages' windows this millions of times a second while they
    // wait on their doorbells; upstream's host stalled inside one of those device switches with both stages waiting
    // on it.  The same answer as the loop below gives.  (A pipelined window always rings: always_publish_ keeps the
    // zero-doorbell graph off, see init.)
    if (*(volatile uint32_t*) h_seq_ < (uint32_t) (fl_k_ + 1)) {
        const double now = now_ms();
        if (now - fl_flush_ms_ <= 2.0 && now - fl_since_ms_ <= 20000.0) return 0;
    }
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = fl_T_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    while (fl_k_ < fl_total_) {
        const int64_t l = lb_ + fl_k_ / G;
        const uint32_t want = (uint32_t) (fl_k_ + 1);
        if (*(volatile uint32_t*) h_seq_ < want) {
            const double now = now_ms();
            if (now - fl_flush_ms_ > 2.0) {   // notice a dead graph, as run() does
                fl_flush_ms_ = now;
                const strata::gpu::Status st = strata::gpu::event_status(ev_done_);
                if (st == strata::gpu::Status::Failed) {
                    err = "verify: layer " + std::to_string(l) + ": " + strata::gpu::last_error();
                    return -1;
                }
                if (st == strata::gpu::Status::Done && *(volatile uint32_t*) h_seq_ < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (graph finished)";
                    return -1;
                }
            }
            if (now - fl_since_ms_ > 20000.0) {
                err = "verify: timed out at layer " + std::to_string(l);
                return -1;
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        ms_wait += now_ms() - fl_since_ms_;
        const int grp = (int) (fl_k_ % G);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window (pipelined): the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ++fl_k_;
        ms_pool += ms_since(b);
        fl_since_ms_ = fl_flush_ms_ = now_ms();
    }
    return 1;
}

bool Verifier::done(std::string& err) {
    if (!fl_active_ || fl_k_ < fl_total_) return false;
    const OnDevice on_device(device_);
    strata::gpu::Status st = strata::gpu::event_status(ev_done_);
    if (st == strata::gpu::Status::Pending) {
        if (now_ms() - fl_since_ms_ > 20000.0) err = "verify: the window never finished";
        return false;
    }
    // no host function of this window may raise flag B in the next one
    if (st == strata::gpu::Status::Done) st = strata::gpu::stream_status(copy_);
    if (st == strata::gpu::Status::Failed) err = std::string("verify: the window: ") + strata::gpu::last_error();
    return st == strata::gpu::Status::Done;
}

bool Verifier::pl_finish(int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    fl_active_ = false;
    if (fl_prof_) accumulate_profile(prof_pin_);
    ++windows;
    const int T = fl_T_;
    if (le_ < g.n_layers) {   // an earlier stage: the hand-off is written
        // the caller launches the next stage's window of this parity only after it has finished the one before,
        // which read its rows
        if (next_ != nullptr) next_->rows(hand_out_, 0, T);
        return true;
    }
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr)) {   // run()'s host-side sampling, Philox(seed, pos0 + t)
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) last_pos0_;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (!strata::gpu::stream_sync(cs_)) {
            err = "verify: the head sampling failed";
            return false;
        }
    }
    if (out != nullptr)
        for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    progress_beat();
    return true;
}

bool Verifier::pl_commit_async(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    if (commit_exec_ == nullptr || ev_commit_ == nullptr) { err = "verify: pipelined commit not prepared"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (commit_live_) {   // a replay (pl_launch waited for the one before): its graph reads the words below
        strata::gpu::Status st;
        while ((st = strata::gpu::event_status(ev_commit_)) == strata::gpu::Status::Pending) _mm_pause();
        if (st == strata::gpu::Status::Failed) {
            err = std::string("verify: the commit before: ") + strata::gpu::last_error();
            return false;
        }
    }
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!strata::gpu::graph_launch(commit_exec_, cs_)) {
        err = std::string("verify: commit launch: ") + strata::gpu::last_error();
        return false;
    }
    if (!strata::gpu::event_record(ev_commit_, cs_)) { err = "verify: event record failed"; return false; }
    commit_live_ = true;
    if (ple_stage())
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return true;
}

void Verifier::absorb_stats(Verifier& o) {
    ms_wait += o.ms_wait; ms_pool += o.ms_pool; ms_host += o.ms_host; ms_commit += o.ms_commit;
    windows += o.windows;
    o.ms_wait = o.ms_pool = o.ms_host = o.ms_commit = 0;
    o.windows = 0;
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < kProfPer; ++i) { prof_sum_[k][i] += o.prof_sum_[k][i]; o.prof_sum_[k][i] = 0; }
    prof_windows_ += o.prof_windows_;
    o.prof_windows_ = 0;
    prof_ns_ += o.prof_ns_; prof_ticks_ += o.prof_ticks_;
    o.prof_ns_ = o.prof_ticks_ = 0;
}

// ================================ THE NEXT STAGE IN THIS PROCESS (StageLink) ================================

void Verifier::accept_rows(const float* from, int row0, int n) {
    float* in = const_cast<float*>(hand_in_);
    if (in == from) return;   // stages on one GPU share the hand-off
    const size_t HB = (size_t) handoff_floats(*g_);
    std::memcpy(in + (size_t) row0 * HB, from + (size_t) row0 * HB, (size_t) n * HB * sizeof(float));
}

void InProcessStage::rows(const float* hand_out, int row0, int n) { next_->accept_rows(hand_out, row0, n); }
bool InProcessStage::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, int32_t* out,
                         std::string& err) {
    return next_->run(T, tokens, pos0, pool, user_, out, err);
}
bool InProcessStage::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos,
                                   PoolMultiFn pool, int32_t* out, std::string& err) {
    return next_->run_slot_rows(rows, S, tokens, pos, pool, user_, out, err);
}
bool InProcessStage::commit(int n_keep, std::string& err) { return next_->commit(n_keep, err); }
bool InProcessStage::commit_slot_prefixes(const int* keep, std::string& err) {
    return next_->commit_slot_prefixes(keep, err);
}
bool InProcessStage::wait_commit(std::string& err) { return next_->wait_commit(err); }
void InProcessStage::set_sampling(const strata::kernels::SamplerParams& sp) { next_->set_sampling(sp); }
void InProcessStage::set_slot_sampling(int slot, const strata::kernels::SamplerParams& sp) {
    next_->set_slot_sampling(slot, sp);
}
void InProcessStage::set_head_sampling(bool on) { next_->set_head_sampling(on); }
bool InProcessStage::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    return next_->set_logit_bias(bias, err);
}
void InProcessStage::set_history(const int32_t* history, int history_len) { next_->set_history(history, history_len); }
bool InProcessStage::read_logits(int rows, float* host, std::string& err) const {
    return next_->read_logits(rows, host, err);
}
bool InProcessStage::copy_logits(int t, float* host) const { return next_->copy_logits(t, host); }
int InProcessStage::nonfinite_row() const { return next_->nonfinite_row(); }
int64_t InProcessStage::nonfinite_pos() const { return next_->nonfinite_pos(); }
bool InProcessStage::nonfinite_in(int t) const { return next_->nonfinite_in(t); }
bool InProcessStage::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                                     std::string& err) {
    return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
}
const float* InProcessStage::final_R_all() const { return next_->final_R_all(); }

}  // namespace strata::core
