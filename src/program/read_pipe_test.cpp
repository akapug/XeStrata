// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/read_pipe_test.cpp - pipelined batch reads' pure rules (header-only, CPU):
//
//   1. the settings: named defaults, the range checks, chunk_min rounded to the grain;
//   2. the read's chunk: off as before (the read rounded up, the --prefill chunk the ceiling), on a quarter of the read
//      rounded up, the floor, the ceiling, never past the read;
//   3. a call's end: off one chunk, on `group` chunks while slots decode and the whole part when none does; every read
//      is covered once, in order, by calls on the chunk grid.
#include "strata/program/read_pipe.hpp"

#include <cstdio>
#include <vector>

namespace rp = strata::program::read_pipe;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
rp::Settings on(int parts = rp::kPartsDefault, int64_t chunk_min = rp::kChunkMinDefault, int group = rp::kGroupDefault) {
    rp::Settings s;
    s.on = true;
    s.parts = parts;
    s.chunk_min = chunk_min;
    s.group = group;
    return s;
}
// the calls a read of [a0, b0) makes with chunk C
std::vector<std::pair<int64_t, int64_t>> calls(int64_t a0, int64_t b0, int64_t C, bool decoding, const rp::Settings& s) {
    std::vector<std::pair<int64_t, int64_t>> v;
    for (int64_t q = a0; q < b0;) {
        const int64_t r = rp::call_end(q, b0, C, decoding, s);
        if (r <= q) return {};   // no progress: a failure the caller checks
        v.emplace_back(q, r);
        q = r;
    }
    return v;
}
}  // namespace

int main() {
    std::printf("1. settings\n");
    {
        rp::Settings s;
        std::string why;
        check(rp::parse_settings(nullptr, nullptr, nullptr, nullptr, s, why) && !s.on &&
                  s.parts == rp::kPartsDefault && s.chunk_min == rp::kChunkMinDefault && s.group == rp::kGroupDefault,
              "unset: off, the named defaults");
        check(rp::parse_settings("", "", "", "", s, why) && !s.on, "empty values: the defaults");
        check(rp::parse_settings("1", "8", "3000", "3", s, why) && s.on && s.parts == 8 && s.chunk_min == 3072 &&
                  s.group == 3,
              "set: on, parts 8, chunk_min 3000 rounded up to 3072, group 3");
        check(!rp::parse_settings("2", nullptr, nullptr, nullptr, s, why) && !why.empty(), "PIPE=2 refused");
        check(!rp::parse_settings("1", "0", nullptr, nullptr, s, why), "PARTS=0 refused");
        check(!rp::parse_settings("1", "65", nullptr, nullptr, s, why), "PARTS=65 refused");
        check(!rp::parse_settings("1", nullptr, "255", nullptr, s, why), "CHUNK_MIN below the grain refused");
        check(!rp::parse_settings("1", nullptr, "4k", nullptr, s, why), "CHUNK_MIN=4k refused (not a number)");
        check(!rp::parse_settings("1", nullptr, nullptr, "0", s, why), "GROUP=0 refused");
    }
    std::printf("2. the read's chunk\n");
    {
        const rp::Settings off;
        check(rp::chunk_for(30000, 32768, off) == 30208, "off: a 30,000-token read is one 30,208 chunk (as before)");
        check(rp::chunk_for(40000, 32768, off) == 32768, "off: the --prefill chunk is the ceiling");
        check(rp::chunk_for(1, 32768, off) == 256, "off: a 1-token read takes the grain");
        const rp::Settings s = on();
        check(rp::chunk_for(30000, 32768, s) == 7680, "on: 30,000 tokens / 4 = 7,500, rounded up to 7,680");
        check(rp::chunk_for(16384, 32768, s) == 4096, "on: 16,384 / 4 = 4,096");
        check(rp::chunk_for(10000, 32768, s) == 4096, "on: 10,000 / 4 = 2,500: the 4,096 floor");
        check(rp::chunk_for(3000, 32768, s) == 3072, "on: a 3,000-token read stays one chunk (never past the read)");
        check(rp::chunk_for(200000, 32768, s) == 32768, "on: a 200k read: the --prefill chunk is the ceiling");
        check(rp::chunk_for(30000, 4096, s) == 4096, "on: a smaller --prefill chunk still caps it");
        check(rp::chunk_for(30000, 32768, on(1)) == 30208, "on, parts 1: the whole read (pipelining off, chunk as before)");
        bool grain = true;
        for (int64_t n = 1; n < 70000; n += 997)
            for (const int64_t cap : {256LL, 2048LL, 8192LL, 32768LL}) {
                const int64_t c = rp::chunk_for(n, cap, s);
                grain = grain && c % rp::kGrain == 0 && c >= rp::kGrain && c <= cap && c <= rp::round_up(n);
            }
        check(grain, "on: every chunk a multiple of 256, at most the cap and the read rounded up");
    }
    std::printf("3. a call's end\n");
    {
        const rp::Settings off;
        const auto v0 = calls(100, 30100, 32768, true, off);
        check(v0.size() == 1 && v0[0] == std::make_pair<int64_t, int64_t>(100, 30100), "off: one chunk, one call");
        const auto v1 = calls(0, 30000, 4096, true, off);
        check(v1.size() == 8 && v1.back().first == 28672, "off: chunk 4096, a call per chunk (8 calls)");
        const rp::Settings s = on();
        const int64_t C = rp::chunk_for(30000, 32768, s);
        const auto d = calls(0, 30000, C, true, s);
        check(d.size() == 2 && d[0].second == 2 * C && d[1].second == 30000,
              "on, decoding: 4 chunks in 2 calls of 2 (a decode turn between)");
        const auto q = calls(0, 30000, C, false, s);
        check(q.size() == 1 && q[0].second == 30000, "on, nothing decoding: the whole part in one call");
        const auto g3 = calls(500, 30500, 4096, true, on(4, 4096, 3));
        bool grid = !g3.empty();
        for (const auto& [a, b] : g3) grid = grid && ((a - 500) % 4096 == 0) && (b == 30500 || (b - 500) % (3 * 4096) == 0);
        check(grid && g3.size() == 3, "on, group 3, from 500: calls start on the read's chunk grid, 3 chunks each");
        bool cover = true;
        for (int64_t n = 1; n < 50000; n += 1231)
            for (const bool dec : {false, true}) {
                const int64_t c = rp::chunk_for(n, 32768, s);
                const auto v = calls(7, 7 + n, c, dec, s);
                int64_t at = 7;
                for (const auto& [a, b] : v) { cover = cover && a == at && b > a; at = b; }
                cover = cover && at == 7 + n;
            }
        check(cover, "every read covered once, in order, by calls that move forward");
    }
    std::printf(g_fail == 0 ? "read_pipe_test: PASS\n" : "read_pipe_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
