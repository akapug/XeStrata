// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/slot_park.hpp - an idle batch slot's conversation written out before the slot is reused
// (homelab kvib, STRATA_SLOT_PARK).
//
// Under slot pressure serve/server.py's pick_slot gives a request a free slot holding its prompt's start, else an empty
// one, else the least recently used idle one; the request's copy_to_slot then overwrites the conversation that slot
// held, and its next turn reads its whole history again.  With STRATA_SLOT_PARK=1 (default off) the request path first
// writes that conversation to --conversation-save - a folder, or kvib:// (other machines' RAM, strata/core/kvib.hpp) -
// so the next turn restores it from there (ConversationDisk::best / load, the existing path).  Window-row prefill
// declines such an admission so the request path does the park.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.  Tested by slot_park_test.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::program::slot_park {

inline constexpr int64_t kMinTokensDefault = 2048;   ///< STRATA_SLOT_PARK_MIN_TOKENS: shorter ones are read again

struct Settings {
    bool on = false;
    int64_t min_tokens = kMinTokensDefault;
};

/// STRATA_SLOT_PARK (0/1) and STRATA_SLOT_PARK_MIN_TOKENS; null or empty keeps a default.  false: `why` names the bad
/// value and `s` is the defaults (off).
inline bool parse_settings(const char* on, const char* min_tokens, Settings& s, std::string& why) {
    s = Settings{};
    auto parse = [](const char* v, long long lo, long long hi, long long& out) {
        if (v == nullptr || v[0] == '\0') return true;
        char* end = nullptr;
        const long long x = std::strtoll(v, &end, 10);
        if (end == v || *end != '\0' || x < lo || x > hi) return false;
        out = x;
        return true;
    };
    long long a = 0, m = s.min_tokens;
    if (!parse(on, 0, 1, a)) { why = "STRATA_SLOT_PARK must be 0 or 1"; return false; }
    if (!parse(min_tokens, 1, 1LL << 30, m)) {
        why = "STRATA_SLOT_PARK_MIN_TOKENS must be 1 or more";
        return false;
    }
    s.on = a != 0;
    s.min_tokens = m;
    return true;
}

/// homelab background park: the serve thread only captures the slot (the GPU part); a writer thread serialises and
/// writes the image (strata/core/conversation_park_queue.hpp).  Off by default: the park writes on the serve thread.
inline constexpr int64_t kQueueJobsDefault = 2;        ///< STRATA_SLOT_PARK_QUEUE: images queued, the one being written included
inline constexpr int64_t kQueueMibDefault = 16384;     ///< STRATA_SLOT_PARK_QUEUE_MIB: host memory those images may hold

struct AsyncSettings {
    bool on = false;
    int64_t jobs = kQueueJobsDefault;
    int64_t mib = kQueueMibDefault;
};

/// STRATA_SLOT_PARK_ASYNC (0/1), STRATA_SLOT_PARK_QUEUE (1-64) and STRATA_SLOT_PARK_QUEUE_MIB (1 or more); null or
/// empty keeps a default.  false: `why` names the bad value and `s` is the defaults (off).
inline bool parse_async_settings(const char* on, const char* jobs, const char* mib, AsyncSettings& s, std::string& why) {
    s = AsyncSettings{};
    auto parse = [](const char* v, long long lo, long long hi, long long& out) {
        if (v == nullptr || v[0] == '\0') return true;
        char* end = nullptr;
        const long long x = std::strtoll(v, &end, 10);
        if (end == v || *end != '\0' || x < lo || x > hi) return false;
        out = x;
        return true;
    };
    long long a = 0, j = s.jobs, m = s.mib;
    if (!parse(on, 0, 1, a)) { why = "STRATA_SLOT_PARK_ASYNC must be 0 or 1"; return false; }
    if (!parse(jobs, 1, 64, j)) { why = "STRATA_SLOT_PARK_QUEUE must be 1 to 64"; return false; }
    if (!parse(mib, 1, 1LL << 30, m)) { why = "STRATA_SLOT_PARK_QUEUE_MIB must be 1 or more"; return false; }
    s.on = a != 0;
    s.jobs = j;
    s.mib = m;
    return true;
}

/// Whether `prompt` goes on with the conversation a slot holds: all of `held`, or `deepest` (its deepest turn
/// checkpoint's tokens; a client renders the history again without the last reply's thinking).  Anything else that
/// takes the slot overwrites that conversation.
template <class P>
bool continued(const std::vector<int32_t>& held, const std::vector<int32_t>* deepest, const std::vector<P>& prompt) {
    auto holds = [&](const std::vector<int32_t>& pre) {
        return !pre.empty() && prompt.size() >= pre.size() &&
               std::equal(pre.begin(), pre.end(), prompt.begin(),
                          [](int32_t a, const P& t) { return (int64_t) a == (int64_t) t; });
    };
    return holds(held) || (deepest != nullptr && holds(*deepest));
}

/// What a slot is, as the park needs it.
struct Slot {
    bool active = false, cached = false, img = false, tainted = false, partial = false;
    int lent = 0;
    int64_t tokens = 0;
};

/// Whether an idle slot's conversation is worth writing out before its reuse (the folder is enabled).
inline bool parkable(const Settings& s, bool disk_on, const Slot& x) {
    return s.on && disk_on && !x.active && x.cached && !x.img && !x.tainted && !x.partial && x.lent == 0 &&
           x.tokens >= s.min_tokens;
}

} // namespace strata::program::slot_park
