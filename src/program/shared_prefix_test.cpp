// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/shared_prefix_test.cpp - shared-prefix reuse's pure rules (header-only, CPU):
//
//   1. the settings: named defaults (off), the range checks, a bad value refused by name;
//   2. the common prefix of two token lists, and whether a held prefix starts a prompt (never its last token);
//   3. the learned cut: the common part aligned down to the chunk (or STRATA_SHARED_PREFIX_ALIGN), at least the
//      minimum, past the read's start, before the last token;
//   4. the root registry's limits (count and bytes; a root of a held origin always fits);
//   5. which slots are sources (a decoding one too; not one still setting up, tainted, with pictures, of the other
//      control-vector setting, or idle and no cache) and which of their checkpoints;
//   6. when a window-row prefill slot starts from another slot's root instead of its own start.
#include "strata/program/shared_prefix.hpp"

#include <cstdio>

namespace sp = strata::program::shared_prefix;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-96s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
sp::Settings on() {
    sp::Settings s;
    s.on = true;
    return s;
}
}  // namespace

int main() {
    std::printf("shared_prefix_test\n");
    {
        sp::Settings s;
        std::string why;
        check(sp::parse_settings(nullptr, nullptr, nullptr, nullptr, nullptr, s, why) && !s.on &&
                  s.min_tokens == sp::kMinTokensDefault && s.align == sp::kAlignDefault &&
                  s.max_roots == sp::kMaxRootsDefault && s.max_mib == sp::kMaxMibDefault,
              "settings: unset is off with the named defaults");
        check(sp::parse_settings("", "", "", "", "", s, why) && !s.on, "settings: empty values are unset");
        check(sp::parse_settings("1", "512", "256", "3", "100", s, why) && s.on && s.min_tokens == 512 && s.align == 256 &&
                  s.max_roots == 3 && s.max_mib == 100,
              "settings: every value read");
        check(sp::parse_settings("0", nullptr, nullptr, "0", "0", s, why) && !s.on && s.max_roots == 0 && s.max_mib == 0,
              "settings: 0 roots and 0 MiB are legal (no slot keeps a root)");
        const char* bad[][5] = {{"2", 0, 0, 0, 0}, {"1", "0", 0, 0, 0}, {"1", 0, "-1", 0, 0},
                                {"1", 0, 0, "x", 0}, {"1", 0, 0, 0, "-5"}, {"1", "12a", 0, 0, 0}};
        const char* names[] = {"STRATA_SHARED_PREFIX ", "STRATA_SHARED_PREFIX_MIN_TOKENS", "STRATA_SHARED_PREFIX_ALIGN",
                               "STRATA_SHARED_PREFIX_MAX ", "STRATA_SHARED_PREFIX_MAX_MIB", "STRATA_SHARED_PREFIX_MIN_TOKENS"};
        for (int i = 0; i < 6; ++i) {
            why.clear();
            const bool refused = !sp::parse_settings(bad[i][0], bad[i][1], bad[i][2], bad[i][3], bad[i][4], s, why);
            std::string name = names[i];
            if (name.back() == ' ') name.pop_back();
            check(refused && why.rfind(name + " ", 0) == 0, "settings: a bad value is refused by its name");
        }
    }
    {
        const std::vector<int64_t> p{1, 2, 3, 4, 5, 6};
        check(sp::common_prefix(p, std::vector<int32_t>{1, 2, 3, 9}) == 3, "common prefix: up to the first difference");
        check(sp::common_prefix(p, std::vector<int32_t>{}) == 0 && sp::common_prefix(p, std::vector<int32_t>{7}) == 0,
              "common prefix: empty and disjoint");
        check(sp::common_prefix(p, std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8}) == 6, "common prefix: bounded by the shorter");
        check(sp::starts(3, 3, 6) && !sp::starts(4, 3, 6) && !sp::starts(0, 3, 6), "starts: within the common part, not empty");
        check(sp::starts(5, 6, 6) && !sp::starts(6, 6, 6), "starts: never the prompt's last token (a window reads it)");
    }
    {
        sp::Settings s = on();
        s.min_tokens = 2048;
        check(sp::learned_cut(30000, 20123, 0, 8192, s) == 16384, "learned cut: aligned down to the chunk");
        check(sp::learned_cut(30000, 20123, 16384, 8192, s) == -1, "learned cut: none at or below where the read starts");
        check(sp::learned_cut(30000, 20123, 100, 2048, s) == 18432, "learned cut: another chunk, another grid");
        check(sp::learned_cut(30000, 5000, 0, 8192, s) == -1, "learned cut: aligned below the minimum, none");
        s.align = 512;
        check(sp::learned_cut(30000, 20123, 0, 8192, s) == 19968, "learned cut: STRATA_SHARED_PREFIX_ALIGN wins over the chunk");
        check(sp::learned_cut(20124, 25000, 0, 8192, s) == 19968 && sp::learned_cut(2100, 9000, 0, 8192, s) == 2048 &&
                  sp::learned_cut(2048, 9000, 0, 8192, s) == -1,
              "learned cut: capped before the last token");
        s.align = 1;
        check(sp::learned_cut(20000, 30000, 0, 0, s) == 19999, "learned cut: align 1 (chunk 0) cuts at the common part");
    }
    {
        sp::Settings s = on();
        s.max_roots = 2;
        s.max_mib = 300;
        const size_t mib = (size_t) 1 << 20;
        check(sp::root_fits(0, 0, 118 * mib, false, s), "roots: the first fits");
        check(sp::root_fits(1, 118 * mib, 118 * mib, false, s), "roots: the second fits");
        check(!sp::root_fits(2, 236 * mib, 1 * mib, false, s), "roots: a third is past the count");
        check(!sp::root_fits(1, 200 * mib, 118 * mib, false, s), "roots: past the bytes");
        check(!sp::root_fits(0, 0, 301 * mib, false, s), "roots: one bigger than the cap");
        check(sp::root_fits(2, 400 * mib, 118 * mib, true, s), "roots: one of a held origin always fits (no new copy)");
        s.max_roots = 0;
        check(!sp::root_fits(0, 0, 1, false, s), "roots: STRATA_SHARED_PREFIX_MAX 0 keeps none");
    }
    {
        sp::SlotState x;
        x.cached = true;
        x.held = 30000;
        check(sp::source_ok(x), "source: an idle cache");
        sp::SlotState d = x;
        d.active = true;
        d.cached = false;
        check(sp::source_ok(d), "source: a decoding slot (its K/V below its checkpoints stays put)");
        sp::SlotState t = d;
        t.setting_up = true;
        check(!sp::source_ok(t), "source: not a slot whose first window has not set up every stage");
        for (int i = 0; i < 4; ++i) {
            sp::SlotState y = x;
            if (i == 0) y.tainted = true;
            if (i == 1) y.img = true;
            if (i == 2) y.cvec_match = false;
            if (i == 3) y.cached = false;
            check(!sp::source_ok(y), "source: not tainted, pictures, other control vector, or idle without its cache");
        }
        check(sp::checkpoint_ok(x, 20000, true) && sp::checkpoint_ok(x, 30000, true), "checkpoint: up to what it holds");
        check(!sp::checkpoint_ok(x, 30001, true) && !sp::checkpoint_ok(x, 100, false) && !sp::checkpoint_ok(x, 0, true),
              "checkpoint: not past what it holds, not off its conversation, not empty");
    }
    {
        sp::Settings s = on();
        s.min_tokens = 2048;
        check(sp::window_takes_shared(0, 20000, 20300, 512, s), "window: the root where the slot holds nothing");
        check(!sp::window_takes_shared(0, 20000, 20600, 512, s), "window: not when the rest does not fit the windows");
        check(sp::window_takes_shared(19900, 22000, 22100, 512, s), "window: the root where its own start does not fit");
        sp::Settings low = s;
        low.min_tokens = 64;
        check(sp::window_takes_shared(19900, 20000, 20300, 512, low), "window: a root that saves at least the minimum");
        check(!sp::window_takes_shared(19900, 20000, 20100, 512, s), "window: its own start when that fits and saves little");
        check(!sp::window_takes_shared(20000, 20000, 20100, 512, s), "window: a root no longer than its own start");
        sp::Settings off = s;
        off.on = false;
        check(!sp::window_takes_shared(0, 20000, 20300, 512, off), "window: off by default");
    }
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
