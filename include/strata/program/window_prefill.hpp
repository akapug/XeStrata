// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/window_prefill.hpp - window-row prefill (homelab): a short prompt, or a returning
// conversation's new part, is fed straight into its batch slot as extra rows of the pipelined decode windows.
//
// --batch with pipelined slot groups (--batch-groups) reads every admitted prompt in the main session, one request at
// a time, with the decode windows drained, and then copies it into the slot.  STRATA_BATCH_PREFILL=1 (default off)
// takes a BGEN whose new part is short (STRATA_BATCH_PREFILL_MAX_TOKENS) another way: the slot starts from what it
// holds (its own conversation, one of its turn checkpoints, or nothing), and its prompt tokens go into the slot's
// group windows as a contiguous run of rows at consecutive positions, as a --batch-mtp slot's token and proposal do.
// A window holds one row for every slot up to its group's last busy one; the rows the window capacity
// (STRATA_BATCH_MAX_T) has above that go to the reading slots.  The slot's first token is the pick of its last prompt
// row.  Decode never stops for the read and there is no main-session read, no park and no copy.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.  Tested by window_prefill_test.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::program::window_prefill {

inline constexpr int64_t kMaxTokensDefault = 512;   ///< STRATA_BATCH_PREFILL_MAX_TOKENS: the longest new part taken
inline constexpr int kMaxRowsDefault = 16;          ///< STRATA_BATCH_PREFILL_ROWS: most rows of one slot in a window
inline constexpr int kMaxSlotsDefault = 2;          ///< STRATA_BATCH_PREFILL_SLOTS: reading slots given extra rows
inline constexpr int kGraphLimitDefault = 128;      ///< STRATA_BATCH_PREFILL_GRAPHS: captured layouts kept a stage

/// How a window's spare rows are shared among the reading slots of its group: `fifo` gives them to the slot admitted
/// first (it reaches its first token soonest), `even` splits them evenly.
enum class Split { Fifo, Even };

struct Settings {
    bool on = false;
    int64_t max_tokens = kMaxTokensDefault;
    int max_rows = kMaxRowsDefault;
    int max_slots = kMaxSlotsDefault;
    Split split = Split::Fifo;
    int graph_limit = kGraphLimitDefault;   ///< 0: keep every captured layout
};

namespace detail {
inline bool parse_int(const char* v, long long lo, long long hi, long long& out) {
    if (v == nullptr || v[0] == '\0') return true;   // unset: the default stays
    char* end = nullptr;
    const long long x = std::strtoll(v, &end, 10);
    if (end == v || *end != '\0' || x < lo || x > hi) return false;
    out = x;
    return true;
}
}  // namespace detail

/// The settings from their environment values (null or empty: the named default).  `window_rows` is the window
/// capacity (verify_max_t()).  False with `why` for a value out of range: the caller keeps the lever off.
inline bool parse_settings(const char* on, const char* max_tokens, const char* max_rows, const char* max_slots,
                           const char* split, const char* graphs, int window_rows, Settings& s, std::string& why) {
    s = Settings{};
    long long v = 0;
    if (!detail::parse_int(on, 0, 1, v)) { why = "STRATA_BATCH_PREFILL must be 0 or 1"; return false; }
    s.on = v != 0;
    v = s.max_tokens;
    if (!detail::parse_int(max_tokens, 1, 1LL << 20, v)) {
        why = "STRATA_BATCH_PREFILL_MAX_TOKENS must be 1 or more";
        return false;
    }
    s.max_tokens = v;
    v = std::min(s.max_rows, window_rows);
    if (!detail::parse_int(max_rows, 1, window_rows, v)) {
        why = "STRATA_BATCH_PREFILL_ROWS must be 1.." + std::to_string(window_rows) + " (the window's rows)";
        return false;
    }
    s.max_rows = (int) v;
    v = s.max_slots;
    if (!detail::parse_int(max_slots, 1, window_rows, v)) {
        why = "STRATA_BATCH_PREFILL_SLOTS must be 1.." + std::to_string(window_rows);
        return false;
    }
    s.max_slots = (int) v;
    if (split != nullptr && split[0] != '\0') {
        const std::string x = split;
        if (x == "fifo") s.split = Split::Fifo;
        else if (x == "even") s.split = Split::Even;
        else { why = "STRATA_BATCH_PREFILL_SPLIT must be fifo or even"; return false; }
    }
    v = s.graph_limit;
    if (!detail::parse_int(graphs, 0, 1 << 16, v)) { why = "STRATA_BATCH_PREFILL_GRAPHS must be 0 or more"; return false; }
    s.graph_limit = (int) v;
    return true;
}

/// A BGEN line: `BGEN <slot> <max_new> [key=value ...] <id,id,...>`, the keys as the request path reads them (the
/// unknown ones skipped).  Only what the window path needs or must refuse is kept; defaults as the request path's.
struct Request {
    int slot = -1;
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 20;                  ///< the sampler's own default (the request path's)
    unsigned long long seed = 0;
    int cvec = 1;
    bool ckpt = true;
    long long pin = 0;
    int logprobs = -1;
    bool penalties = false;          ///< a repeat / frequency / presence penalty was asked for
    bool logit_bias = false;
    bool img_keys = false;
    std::vector<int64_t> ids;
};

inline bool parse_bgen(const std::string& line, Request& r, std::string& why) {
    r = Request{};
    if (line.rfind("BGEN ", 0) != 0) { why = "not a BGEN line"; return false; }
    const char* p = line.c_str() + 5;
    char* e = nullptr;
    const long b = std::strtol(p, &e, 10);
    if (e == p) { why = "no slot"; return false; }
    p = e;
    const long long mn = std::strtoll(p, &e, 10);
    if (e == p || mn < 1) { why = "a bad max_new"; return false; }
    r.slot = (int) b;
    r.max_new = mn;
    p = e;
    for (;;) {   // key=value tokens until the first one without '='
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        if (p == start) break;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) { p = start; break; }
        const std::string key = tok.substr(0, eq);
        const char* val = tok.c_str() + eq + 1;
        const float fv = std::strtof(val, nullptr);
        if (key == "logit_bias") r.logit_bias = true;
        else if (key == "cvec") r.cvec = (int) std::strtol(val, nullptr, 10);
        else if (key == "ckpt") r.ckpt = std::strtol(val, nullptr, 10) != 0;
        else if (key == "pin") r.pin = std::max<long long>(0, std::strtoll(val, nullptr, 10));
        else if (key == "temperature") r.temperature = fv;
        else if (key == "top_p") r.top_p = fv;
        else if (key == "top_k") r.top_k = (int) std::strtol(val, nullptr, 10);
        else if (key == "min_p") r.min_p = fv;
        else if (key == "logprobs") r.logprobs = (int) std::clamp(std::strtol(val, nullptr, 10), -1L, 20L);
        else if (key == "penalty_repeat") r.penalties = r.penalties || fv != 1.0f;
        else if (key == "penalty_freq" || key == "penalty_present") r.penalties = r.penalties || fv != 0.0f;
        else if (key == "seed") r.seed = std::strtoull(val, nullptr, 10);
        else if (key == "img_keys") r.img_keys = true;
    }
    while (*p == ' ') ++p;
    while (*p != '\0') {
        const long long t = std::strtoll(p, &e, 10);
        if (e == p) { why = "a bad token list"; return false; }
        r.ids.push_back(t);
        p = e;
        if (*p == ',') ++p;
        else if (*p != '\0' && *p != ' ' && *p != '\n' && *p != '\r') { why = "a bad token list"; return false; }
        while (*p == ' ' || *p == '\n' || *p == '\r') ++p;
    }
    if (r.ids.empty()) { why = "no prompt tokens"; return false; }
    return true;
}

/// What a slot holds of a prompt: its whole conversation (`ids`, when `ids_ok`) or one of its checkpoints, whichever
/// is the longest start of the prompt of at most n - 1 tokens (the last prompt token always goes through a window, as
/// the request path's `starts_with` keeps it).  checkpoint -1: the conversation itself; tokens 0: nothing.
struct Prefix {
    int64_t tokens = 0;
    int checkpoint = -1;
};

inline bool starts_with(const std::vector<int64_t>& prompt, const std::vector<int32_t>& pre) {
    const int64_t L = (int64_t) pre.size();
    if (L < 1 || L > (int64_t) prompt.size() - 1) return false;
    for (int64_t i = 0; i < L; ++i)
        if (prompt[(size_t) i] != (int64_t) pre[(size_t) i]) return false;
    return true;
}

inline Prefix slot_prefix(const std::vector<int64_t>& prompt, const std::vector<int32_t>& ids, bool ids_ok,
                          const std::vector<const std::vector<int32_t>*>& checkpoints) {
    Prefix best;
    if (ids_ok && starts_with(prompt, ids)) best.tokens = (int64_t) ids.size();
    for (size_t i = 0; i < checkpoints.size(); ++i)
        if ((int64_t) checkpoints[i]->size() > best.tokens && starts_with(prompt, *checkpoints[i])) {
            best.tokens = (int64_t) checkpoints[i]->size();
            best.checkpoint = (int) i;
        }
    return best;
}

/// The turn checkpoint's place, as the request path chooses it: the last `turn_token` past `from` (the conversation
/// so far ends before it), moved back over a short turn of `tail_role_token`'s role right before it (#458; < 0: no
/// such role).  -1: none.
inline int64_t turn_boundary(const std::vector<int64_t>& ids, int64_t from, int64_t turn_token, int64_t tail_role_token) {
    if (turn_token < 0) return -1;
    const int64_t n = (int64_t) ids.size();
    int64_t at = -1;
    for (int64_t i = n - 1; i > from; --i)
        if (ids[(size_t) i] == turn_token) { at = i; break; }
    if (at > 0 && tail_role_token >= 0)
        for (int64_t i = at - 1; i > from; --i)
            if (ids[(size_t) i] == turn_token) {
                if (i + 1 < n && ids[(size_t) i + 1] == tail_role_token) at = i;
                break;
            }
    return at;
}

/// One slot of a group as the next window sees it.
struct SlotIn {
    enum Kind { Idle, Decode, Prefill } kind = Idle;
    int64_t left = 0;     ///< Prefill: prompt tokens not fed yet (>= 1)
    int64_t to_cut = 0;   ///< Prefill: tokens until its turn checkpoint (0: none ahead); a window ends there
    int64_t room = 0;     ///< Prefill: cells from its next position to the end of its context
    uint64_t since = 0;   ///< Prefill: admission order (fifo)
};

/// Its rows in the window: `rows` contiguous rows, the first `keep` of them kept (the rest pad a short run up to a
/// power of two, so few window layouts are captured; they are not committed).  An idle slot below the group's last
/// busy one gets the identity layout's one row (rows 1, keep 1 here); the caller makes it a pad row (pad_row).
struct SlotOut {
    int rows = 0;
    int keep = 0;
};

inline int pow2_floor(int64_t n) {
    int r = 1;
    while ((int64_t) r * 2 <= n) r *= 2;
    return r;
}
inline int64_t pow2_ceil(int64_t n) {
    int64_t r = 1;
    while (r < n) r *= 2;
    return r;
}

/// Plans a group's window.  Every slot up to the last busy one has a row; the `cap` - S rows above them go to the
/// reading slots in `split` order, at most `max_slots` of them taking more than one row.  A reading slot takes a
/// power of two of rows (at most max_rows, at most 1 + its share of the spare rows, at most its room), keeping its
/// prompt tokens and its turn checkpoint's place: its run ends at the checkpoint, and its last run is padded up to the
/// next power of two when that fits.  Returns S (0: nothing busy; -1: S > cap).  `out` gets S entries.
inline int plan_window(const std::vector<SlotIn>& in, int cap, const Settings& s, std::vector<SlotOut>& out) {
    int S = 0;
    for (int t = 0; t < (int) in.size(); ++t)
        if (in[(size_t) t].kind != SlotIn::Idle) S = t + 1;
    out.assign((size_t) S, SlotOut{1, 1});
    if (S == 0) return 0;
    if (S > cap) return -1;
    std::vector<int> order;
    for (int t = 0; t < S; ++t)
        if (in[(size_t) t].kind == SlotIn::Prefill) order.push_back(t);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return in[(size_t) a].since < in[(size_t) b].since; });
    int64_t spare = cap - S;
    const int sharers = std::min<int>((int) order.size(), std::max(s.max_slots, 1));
    const int64_t share = s.split == Split::Even && sharers > 0 ? spare / sharers : spare;
    int chosen = 0;
    for (const int t : order) {
        const SlotIn& x = in[(size_t) t];
        int64_t want = std::min<int64_t>(std::max<int64_t>(x.left, 1), std::max(s.max_rows, 1));
        if (x.to_cut > 0) want = std::min(want, x.to_cut);
        const int64_t budget = chosen < s.max_slots ? std::min(spare, share) : 0;
        const int64_t avail = std::max<int64_t>(1, std::min<int64_t>({1 + budget, std::max<int64_t>(x.room, 1),
                                                                      std::max(s.max_rows, 1)}));
        int rows = 0;
        if (want <= avail && pow2_ceil(want) <= avail) rows = (int) pow2_ceil(want);
        else rows = pow2_floor(std::min(want, avail));
        SlotOut& o = out[(size_t) t];
        o.rows = rows;
        o.keep = (int) std::min<int64_t>(want, rows);
        if (rows > 1) {
            ++chosen;
            spare -= rows - 1;
        }
    }
    return S;
}

inline constexpr bool kPadKeepsIdleDefault = true;   ///< STRATA_BATCH_PAD_KEEPS_IDLE

/// An idle slot's row in a group window (every slot up to the group's last busy one has a row: the captured layouts
/// are fixed).  It used to feed token 0 at position 0 and keep it, which overwrote the slot's conversation (its GDN
/// histories and state, its first K/V cell and indexer block, its PLE history), so the slot stopped being a cache.
/// With `keeps_idle` (STRATA_BATCH_PAD_KEEPS_IDLE, default on) and a verifier whose window can skip the row
/// (Verifier::pad_rows_harmless) it is a pad row: keep 0, its K/V into cell `held` (the slot's history length, past
/// it; the slot's next token overwrites that cell), nothing else written.  `cells`: the slot's context; a full slot
/// has no cell past its history and keeps the old row (`writes_slot`: the caller drops its cache flag).
struct PadRow {
    int64_t pos = 0;
    int keep = 1;
    bool writes_slot = true;
};
inline PadRow pad_row(bool keeps_idle, bool harmless, int64_t held, int64_t cells) {
    if (keeps_idle && harmless && held >= 0 && held < cells) return PadRow{held, 0, false};
    return PadRow{};
}
/// STRATA_BATCH_PAD_KEEPS_IDLE (null or empty: the default).  False with `why` for a value other than 0 or 1.
inline bool parse_pad_keeps_idle(const char* v, bool& out, std::string& why) {
    long long x = kPadKeepsIdleDefault ? 1 : 0;
    if (!detail::parse_int(v, 0, 1, x)) { why = "STRATA_BATCH_PAD_KEEPS_IDLE must be 0 or 1"; return false; }
    out = x != 0;
    return true;
}

}  // namespace strata::program::window_prefill
