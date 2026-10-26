// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/read_pipe.hpp - pipelined, chunked batch reads (homelab, STRATA_BATCH_READ_PIPE).
//
// With --batch, a prompt the request path reads (a new part over STRATA_BATCH_PREFILL_MAX_TOKENS, or any part when
// window-row prefill is off) used to go to the prompt path one chunk per call: every call waits for every layer-split
// stage before it returns, so stage 1 sat idle while stage 0 read and the other way round.  --prefill auto picks
// 32768-token chunks, so nearly every read was one chunk, every decoding slot stopped for the whole read, and a
// waiting short request could not make it give way (BYIELD needs a chunk boundary).
//
// STRATA_BATCH_READ_PIPE=1 (default off):
//   - the read's chunk is its tokens / STRATA_BATCH_READ_PARTS (default 4), rounded up to 256 tokens, at least
//     STRATA_BATCH_READ_CHUNK_MIN (default 4096: at and above the prompt path's 2048 every streamed expert goes through
//     the GPU ring, below it the CPU takes a share), at most the --prefill chunk.  The loan of cache slots is sized
//     for that chunk, so a shorter chunk also leaves the decode windows more resident experts;
//   - one prompt-path call takes STRATA_BATCH_READ_GROUP chunks (default 2) while slots decode, the whole part when
//     none does.  Inside a call stage 0 reads chunk i + 1 while stage 1 reads chunk i;
//   - between two calls a waiting shorter request can take the read's place (BYIELD), and the slots decode for
//     STRATA_BATCH_DECODE_SHARE of the call's time (the pipelined groups when --batch-groups pipelines them).
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.  Tested by read_pipe_test.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace strata::program::read_pipe {

inline constexpr int64_t kGrain = 256;               ///< chunks are whole multiples of this (the loan's rounding)
inline constexpr int kPartsDefault = 4;              ///< STRATA_BATCH_READ_PARTS: chunks a read is cut into
inline constexpr int64_t kChunkMinDefault = 4096;    ///< STRATA_BATCH_READ_CHUNK_MIN: the smallest chunk
inline constexpr int kGroupDefault = 2;              ///< STRATA_BATCH_READ_GROUP: chunks a call while slots decode

struct Settings {
    bool on = false;
    int parts = kPartsDefault;
    int64_t chunk_min = kChunkMinDefault;
    int group = kGroupDefault;
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
/// of range: the caller keeps the lever off.  chunk_min is rounded up to kGrain.
inline bool parse_settings(const char* on, const char* parts, const char* chunk_min, const char* group, Settings& s,
                           std::string& why) {
    s = Settings{};
    long long v = 0;
    if (!detail::parse_int(on, 0, 1, v)) { why = "STRATA_BATCH_READ_PIPE must be 0 or 1"; return false; }
    s.on = v != 0;
    v = s.parts;
    if (!detail::parse_int(parts, 1, 64, v)) { why = "STRATA_BATCH_READ_PARTS must be 1..64"; return false; }
    s.parts = (int) v;
    v = s.chunk_min;
    if (!detail::parse_int(chunk_min, kGrain, 1LL << 20, v)) {
        why = "STRATA_BATCH_READ_CHUNK_MIN must be " + std::to_string(kGrain) + ".." + std::to_string(1LL << 20);
        return false;
    }
    s.chunk_min = (v + kGrain - 1) / kGrain * kGrain;
    v = s.group;
    if (!detail::parse_int(group, 1, 64, v)) { why = "STRATA_BATCH_READ_GROUP must be 1..64"; return false; }
    s.group = (int) v;
    return true;
}

inline int64_t round_up(int64_t n) { return (std::max<int64_t>(n, 1) + kGrain - 1) / kGrain * kGrain; }

/// The prompt path's chunk for a read of `tokens` with the --prefill chunk `cap`: off, as before (the read rounded up
/// to the grain, at most cap); on, tokens / parts rounded up, at least chunk_min, at most cap, never past the read.
inline int64_t chunk_for(int64_t tokens, int64_t cap, const Settings& s) {
    const int64_t whole = std::min(cap, round_up(tokens));
    if (!s.on) return whole;
    const int64_t part = round_up((std::max<int64_t>(tokens, 1) + s.parts - 1) / s.parts);
    return std::min(whole, std::max(part, s.chunk_min));
}

/// Where one prompt-path call that starts at q ends, in a part that ends at b0, with chunk C: off, one chunk (as
/// before); on, `group` chunks while slots decode, else the rest of the part.
inline int64_t call_end(int64_t q, int64_t b0, int64_t C, bool decoding, const Settings& s) {
    const int64_t c = std::max<int64_t>(C, 1);
    if (!s.on) return std::min(b0, q + c);
    if (!decoding) return b0;
    return std::min(b0, q + c * s.group);
}

}  // namespace strata::program::read_pipe
