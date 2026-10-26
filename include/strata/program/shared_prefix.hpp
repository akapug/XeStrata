// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/shared_prefix.hpp - shared-prefix reuse across batch slots (homelab, STRATA_SHARED_PREFIX).
//
// Many agents start with the same long system prompt and tool list.  Without this lever a batch slot reuses only its
// own conversation, and the main session only the chain it holds, so a new agent reads the shared part again whenever
// the main session moved on (every slot-sourced turn clears its chain).  With STRATA_SHARED_PREFIX=1 (default off):
//   * A prompt read on the request path also stops at its LEARNED CUT: the longest prefix it shares with any token
//     sequence the engine has seen (the slots' conversations, the main session's, the parked ones), aligned down to
//     the prompt chunk so the read keeps its chunk grid.  The checkpoint there is marked `shared` (so is the system
//     prompt root when it lies inside that common part).
//   * A batch slot keeps the deepest shared checkpoint below its turn checkpoint as its ROOT, by reference: every
//     slot whose K/V is a byte copy of the same read holds one copy of that state (its origin).
//   * The request path looks for a prefix in every slot - idle or decoding - through their checkpoints and roots,
//     and copies it from there (conversation_prefix_copy); the window-row prefill path starts a slot from another
//     slot's root and reads only the rest in its windows.
// The K/V of a shared prefix always comes from a slot whose own history wrote it together with that running state
// (one origin), never mixed from two reads.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.  Tested by shared_prefix_test.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::program::shared_prefix {

inline constexpr int64_t kMinTokensDefault = 2048;   ///< STRATA_SHARED_PREFIX_MIN_TOKENS: shortest prefix shared
inline constexpr int64_t kAlignDefault = 0;          ///< STRATA_SHARED_PREFIX_ALIGN: learned cut granule (0: the chunk)
inline constexpr int kMaxRootsDefault = 8;           ///< STRATA_SHARED_PREFIX_MAX: distinct roots the slots hold
inline constexpr int64_t kMaxMibDefault = 2048;      ///< STRATA_SHARED_PREFIX_MAX_MIB: host RAM those roots take

struct Settings {
    bool on = false;
    int64_t min_tokens = kMinTokensDefault;
    int64_t align = kAlignDefault;
    int max_roots = kMaxRootsDefault;
    int64_t max_mib = kMaxMibDefault;
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

/// The settings from their environment values (null or empty: the named default).  False with `why` for a value out
/// of range: the caller keeps the lever off.
inline bool parse_settings(const char* on, const char* min_tokens, const char* align, const char* max_roots,
                           const char* max_mib, Settings& s, std::string& why) {
    s = Settings{};
    long long v = 0;
    if (!detail::parse_int(on, 0, 1, v)) { why = "STRATA_SHARED_PREFIX must be 0 or 1"; return false; }
    s.on = v != 0;
    v = s.min_tokens;
    if (!detail::parse_int(min_tokens, 1, 1LL << 30, v)) {
        why = "STRATA_SHARED_PREFIX_MIN_TOKENS must be 1 or more";
        return false;
    }
    s.min_tokens = v;
    v = s.align;
    if (!detail::parse_int(align, 0, 1LL << 30, v)) { why = "STRATA_SHARED_PREFIX_ALIGN must be 0 or more"; return false; }
    s.align = v;
    v = s.max_roots;
    if (!detail::parse_int(max_roots, 0, 1 << 16, v)) { why = "STRATA_SHARED_PREFIX_MAX must be 0 or more"; return false; }
    s.max_roots = (int) v;
    v = s.max_mib;
    if (!detail::parse_int(max_mib, 0, 1LL << 30, v)) {
        why = "STRATA_SHARED_PREFIX_MAX_MIB must be 0 or more";
        return false;
    }
    s.max_mib = v;
    return true;
}

/// How many leading tokens `a` and `b` share.
template <class A, class B> int64_t common_prefix(const std::vector<A>& a, const std::vector<B>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && (int64_t) a[i] == (int64_t) b[i]) ++i;
    return (int64_t) i;
}

/// A held prefix of `pre` tokens starts a prompt of `n` (it leaves the last prompt token to a window, as every reuse
/// path does).
inline bool starts(int64_t pre, int64_t common, int64_t n) { return pre >= 1 && pre <= common && pre <= n - 1; }

/// The learned cut of a prompt of `n` tokens whose longest prefix shared with a sequence seen is `common`, read from
/// `read_from`: `common` aligned down to `align` (0: `chunk`, the prompt path's chunk, so a read from token 0 keeps its
/// chunk grid), at least `min_tokens`, past `read_from` and before the last token.  -1: none.
inline int64_t learned_cut(int64_t n, int64_t common, int64_t read_from, int64_t chunk, const Settings& s) {
    const int64_t a = std::max<int64_t>(s.align > 0 ? s.align : chunk, 1);
    const int64_t at = (std::min(common, n - 1) / a) * a;
    if (at < s.min_tokens || at <= read_from) return -1;
    return at;
}

/// Whether the root a slot is about to keep fits beside the distinct roots the slots hold now (`held` of them,
/// `held_bytes` bytes); one already held (same origin) always fits.
inline bool root_fits(int held, size_t held_bytes, size_t bytes, bool already_held, const Settings& s) {
    if (already_held) return true;
    if (held + 1 > s.max_roots) return false;
    const size_t cap = (size_t) s.max_mib << 20;
    return bytes <= cap && held_bytes <= cap - bytes;
}

/// One batch slot as a prefix source.  Its K/V below a checkpoint of its own history (its turn checkpoints, its root)
/// stays put while its conversation only grows: a decoding slot is a source too, a slot whose first window has not
/// set up all its stages yet is not, nor one tainted by non-finite logits (#879), one with pictures, one read with the
/// other control-vector setting, or an idle one whose state stopped being a cache (a pad row wrote it).
struct SlotState {
    bool active = false, cached = false, tainted = false, img = false, cvec_match = true;
    bool setting_up = false;    ///< window-row prefill: stages still to set up at their launch
    int64_t held = 0;           ///< tokens its sessions hold (committed)
};
inline bool source_ok(const SlotState& x) {
    if (x.tainted || x.img || !x.cvec_match || x.setting_up) return false;
    return x.active || x.cached;
}
/// A checkpoint of `len` tokens of a source slot that holds `held` and whose conversation starts with it.
inline bool checkpoint_ok(const SlotState& x, int64_t len, bool prefix_of_held) {
    return source_ok(x) && len >= 1 && len <= x.held && prefix_of_held;
}

/// Window-row prefill: whether a slot starts from another slot's root (`shared` tokens) instead of what it holds
/// itself (`own`), for a prompt of `n` with at most `max_new_part` tokens read in the windows.  Only where it fits the
/// windows and either the slot's own prefix does not, or the root saves at least `min_tokens` more rows.
inline bool window_takes_shared(int64_t own, int64_t shared, int64_t n, int64_t max_new_part, const Settings& s) {
    if (!s.on || shared <= own || n - shared > max_new_part) return false;
    return n - own > max_new_part || shared - own >= s.min_tokens;
}

}  // namespace strata::program::shared_prefix
