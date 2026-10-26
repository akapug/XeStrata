// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/window_prefill_test.cpp - window-row prefill's pure rules (header-only, CPU):
//
//   1. the settings: named defaults, the range checks, the window capacity bound;
//   2. a BGEN line parsed as the request path parses it (keys, defaults, the token list, refusals);
//   3. what a slot holds of a prompt (its conversation or a checkpoint, at most n - 1 tokens);
//   4. the turn checkpoint's place (the last turn token past the reuse, #458's tail role);
//   5. a window's plan: a row for every slot up to the last busy one, the capacity respected, every reading slot
//      fed at least one real token, power-of-two runs, the checkpoint cut, the padding, fifo and even splits, the
//      cap on slots taking extra rows, the room at the end of a slot's context.
#include "strata/program/window_prefill.hpp"

#include <cstdio>
#include <numeric>

namespace wp = strata::program::window_prefill;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
wp::SlotIn dec() { wp::SlotIn s; s.kind = wp::SlotIn::Decode; return s; }
wp::SlotIn idle() { return wp::SlotIn{}; }
wp::SlotIn pre(int64_t left, uint64_t since, int64_t to_cut = 0, int64_t room = 1 << 20) {
    wp::SlotIn s;
    s.kind = wp::SlotIn::Prefill;
    s.left = left;
    s.since = since;
    s.to_cut = to_cut;
    s.room = room;
    return s;
}
int rows_of(const std::vector<wp::SlotOut>& o) {
    int r = 0;
    for (const auto& x : o) r += x.rows;
    return r;
}
bool pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }
}  // namespace

int main() {
    std::printf("window_prefill_test\n");
    {
        wp::Settings s;
        std::string why;
        check(wp::parse_settings(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 32, s, why) && !s.on &&
                  s.max_tokens == wp::kMaxTokensDefault && s.max_rows == wp::kMaxRowsDefault &&
                  s.max_slots == wp::kMaxSlotsDefault && s.split == wp::Split::Fifo &&
                  s.graph_limit == wp::kGraphLimitDefault,
              "settings: unset -> off with the named defaults");
        check(wp::parse_settings("1", "300", "8", "3", "even", "0", 32, s, why) && s.on && s.max_tokens == 300 &&
                  s.max_rows == 8 && s.max_slots == 3 && s.split == wp::Split::Even && s.graph_limit == 0,
              "settings: every value given");
        check(wp::parse_settings("1", nullptr, nullptr, nullptr, nullptr, nullptr, 8, s, why) && s.max_rows == 8,
              "settings: the default rows bounded by an 8-row window");
        check(!wp::parse_settings("1", nullptr, "33", nullptr, nullptr, nullptr, 32, s, why) && !why.empty(),
              "settings: more rows than the window refused");
        check(!wp::parse_settings("2", nullptr, nullptr, nullptr, nullptr, nullptr, 32, s, why), "settings: on=2 refused");
        check(!wp::parse_settings("1", "0", nullptr, nullptr, nullptr, nullptr, 32, s, why), "settings: 0 tokens refused");
        check(!wp::parse_settings("1", "12x", nullptr, nullptr, nullptr, nullptr, 32, s, why), "settings: junk refused");
        check(!wp::parse_settings("1", nullptr, nullptr, nullptr, "lifo", nullptr, 32, s, why), "settings: unknown split");
    }
    {
        wp::Request r;
        std::string why;
        check(wp::parse_bgen("BGEN 3 100 1,2,3", r, why) && r.slot == 3 && r.max_new == 100 && r.ids.size() == 3 &&
                  r.ids[2] == 3 && r.temperature == 0.0f && r.top_k == 20 && r.ckpt && r.cvec == 1 && r.pin == 0 &&
                  r.logprobs == -1 && !r.penalties && !r.logit_bias,
              "BGEN: plain, the request path's defaults");
        check(wp::parse_bgen("BGEN 0 5 temperature=0.7 top_p=0.9 top_k=40 min_p=0.05 seed=7 ckpt=0 cvec=0 foo=1 9,8",
                             r, why) &&
                  r.temperature > 0.69f && r.top_p < 0.91f && r.top_k == 40 && r.min_p > 0.04f && r.seed == 7 &&
                  !r.ckpt && r.cvec == 0 && r.ids.size() == 2 && r.ids[0] == 9,
              "BGEN: sampling keys, ckpt, cvec, an unknown key skipped");
        check(wp::parse_bgen("BGEN 1 5 penalty_repeat=1.1 logprobs=3 pin=40 logit_bias=1:2 img_keys=ab 4", r, why) &&
                  r.penalties && r.logprobs == 3 && r.pin == 40 && r.logit_bias && r.img_keys,
              "BGEN: the keys the window path refuses are seen");
        check(wp::parse_bgen("BGEN 1 5 penalty_repeat=1.0 penalty_freq=0 4", r, why) && !r.penalties,
              "BGEN: neutral penalties are no penalties");
        check(!wp::parse_bgen("BGEN 1 0 4", r, why), "BGEN: max_new 0 refused");
        check(!wp::parse_bgen("BGEN 1 5", r, why), "BGEN: no tokens refused");
        check(!wp::parse_bgen("BGEN 1 5 4,x", r, why), "BGEN: a bad token refused");
        check(!wp::parse_bgen("BGENI 1 5 4", r, why), "BGEN: BGENI (pictures) is not this path's");
        check(wp::parse_bgen("BGEN 2 5 4,5\n", r, why) && r.ids.size() == 2, "BGEN: a trailing newline");
    }
    {
        const std::vector<int64_t> prompt = {1, 2, 3, 4, 5, 6};
        const std::vector<int32_t> ids = {1, 2, 3, 4}, longer = {1, 2, 3, 4, 5, 6}, other = {1, 2, 9};
        const std::vector<int32_t> c1 = {1, 2}, c2 = {1, 2, 3, 4, 5};
        check(wp::slot_prefix(prompt, ids, true, {}).tokens == 4 && wp::slot_prefix(prompt, ids, true, {}).checkpoint == -1,
              "prefix: the slot's conversation");
        check(wp::slot_prefix(prompt, ids, false, {}).tokens == 0, "prefix: not a cache -> nothing");
        check(wp::slot_prefix(prompt, longer, true, {}).tokens == 0, "prefix: the whole prompt is not a start (n - 1 at most)");
        check(wp::slot_prefix(prompt, other, true, {&c1}).tokens == 2 && wp::slot_prefix(prompt, other, true, {&c1}).checkpoint == 0,
              "prefix: a checkpoint when the conversation went another way");
        const auto p = wp::slot_prefix(prompt, ids, true, {&c1, &c2});
        check(p.tokens == 5 && p.checkpoint == 1, "prefix: the longest of conversation and checkpoints");
    }
    {
        const int64_t T = 100, R = 7;
        const std::vector<int64_t> ids = {T, 1, 2, T, 3, 4, T, R, 5, T, 6};
        check(wp::turn_boundary(ids, 0, T, -1) == 9, "turn: the last turn token");
        check(wp::turn_boundary(ids, 9, T, -1) == -1, "turn: none past the reuse");
        check(wp::turn_boundary(ids, 0, T, R) == 6, "turn: back over a tail-role turn (#458)");
        check(wp::turn_boundary(ids, 0, -1, -1) == -1, "turn: no turn token");
    }
    wp::Settings s;
    s.on = true;
    {
        std::vector<wp::SlotOut> o;
        check(wp::plan_window({idle(), idle()}, 32, s, o) == 0 && o.empty(), "plan: nothing busy");
        std::vector<wp::SlotIn> d(16, dec());
        check(wp::plan_window(d, 32, s, o) == 16 && rows_of(o) == 16, "plan: decode only, one row a slot");
        check(wp::plan_window(std::vector<wp::SlotIn>(33, dec()), 32, s, o) == -1, "plan: more slots than rows refused");
    }
    {   // 16 decoding slots, one reading: 16 spare rows
        std::vector<wp::SlotIn> in(16, dec());
        in[5] = pre(300, 1);
        std::vector<wp::SlotOut> o;
        const int S = wp::plan_window(in, 32, s, o);
        check(S == 16 && o[5].rows == 16 && o[5].keep == 16 && rows_of(o) == 31, "plan: a reader takes max_rows (16)");
        in[5] = pre(5, 1);
        wp::plan_window(in, 32, s, o);
        check(o[5].rows == 8 && o[5].keep == 5, "plan: its last run padded up to a power of two (5 -> 8)");
        in[5] = pre(5, 1, 0, 6);
        wp::plan_window(in, 32, s, o);
        check(o[5].rows == 4 && o[5].keep == 4, "plan: no padding past its context's end");
        in[5] = pre(300, 1, 3);
        wp::plan_window(in, 32, s, o);
        check(o[5].keep == 3 && o[5].rows == 4, "plan: the run ends at the turn checkpoint");
    }
    {   // a full window: every reader still reads one real token
        std::vector<wp::SlotIn> in(32, dec());
        in[0] = pre(300, 2);
        in[31] = pre(300, 1);
        std::vector<wp::SlotOut> o;
        wp::plan_window(in, 32, s, o);
        check(o[0].rows == 1 && o[0].keep == 1 && o[31].rows == 1 && o[31].keep == 1 && rows_of(o) == 32,
              "plan: no spare rows -> one real prompt row each");
    }
    {   // fifo vs even, and the cap on slots with extra rows
        std::vector<wp::SlotIn> in(16, dec());
        in[1] = pre(300, 3);
        in[2] = pre(300, 1);
        in[3] = pre(300, 2);
        std::vector<wp::SlotOut> o;
        wp::plan_window(in, 32, s, o);   // fifo, 16 spare: slot 2 first (since 1)
        check(o[2].rows == 16 && o[3].rows == 2 && o[1].rows == 1 && rows_of(o) <= 32,
              "plan: fifo gives the spare rows to the first admitted, the next the rest");
        wp::Settings e = s;
        e.split = wp::Split::Even;
        wp::plan_window(in, 32, e, o);   // 16 spare over 2 sharers (max_slots 2): 8 each -> 8 rows (1 + 8 -> 8)
        check(o[2].rows == 8 && o[3].rows == 8 && o[1].rows == 1 && rows_of(o) <= 32,
              "plan: even splits them among max_slots readers");
        e.max_slots = 3;
        wp::plan_window(in, 32, e, o);   // 16 / 3 = 5 each -> 1 + 5 -> 4 rows
        check(o[2].rows == 4 && o[3].rows == 4 && o[1].rows == 4 && rows_of(o) <= 32, "plan: even over three readers");
        wp::Settings f = s;
        f.max_rows = 4;
        wp::plan_window(in, 32, f, o);   // fifo, 4 rows max, 2 slots with extra rows
        check(o[2].rows == 4 && o[3].rows == 4 && o[1].rows == 1, "plan: max_slots caps the readers with extra rows");
    }
    {   // invariants over many layouts
        bool ok = true;
        for (int cap : {8, 16, 24, 32})
            for (int n = 1; n <= cap; ++n)
                for (int mask = 0; mask < 16; ++mask) {
                    std::vector<wp::SlotIn> in((size_t) n, dec());
                    for (int t = 0; t < n; ++t) {
                        if ((mask >> (t % 4)) & 1) in[(size_t) t] = pre(1 + (t * 37 + mask) % 70, (uint64_t) (t ^ mask), (t * 5) % 9, 4 + t);
                        else if ((t + mask) % 5 == 0) in[(size_t) t] = idle();
                    }
                    std::vector<wp::SlotOut> o;
                    const int S = wp::plan_window(in, cap, s, o);
                    if (S < 0) { ok = false; continue; }
                    ok = ok && rows_of(o) <= cap;
                    int extra = 0;
                    for (int t = 0; t < S; ++t) {
                        const auto& x = in[(size_t) t];
                        const auto& y = o[(size_t) t];
                        ok = ok && y.rows >= 1 && y.keep >= 1 && y.keep <= y.rows && pow2(y.rows);
                        if (x.kind == wp::SlotIn::Prefill) {
                            ok = ok && y.keep <= x.left && y.rows <= x.room && y.rows <= s.max_rows;
                            if (x.to_cut > 0) ok = ok && y.keep <= x.to_cut;
                            extra += y.rows > 1;
                        } else {
                            ok = ok && y.rows == 1;
                        }
                    }
                    ok = ok && extra <= s.max_slots;
                }
        check(ok, "plan: capacity, >=1 real row a reader, keep <= left/cut, rows <= room, powers of two, max_slots");
    }
    {   // pad rows (STRATA_BATCH_PAD_KEEPS_IDLE): an idle slot's row past its history, kept nowhere
        const wp::PadRow p = wp::pad_row(true, true, 37, 4096);
        check(p.pos == 37 && p.keep == 0 && !p.writes_slot, "pad: its K/V at its history length, nothing kept");
        const wp::PadRow e = wp::pad_row(true, true, 0, 4096);
        check(e.pos == 0 && e.keep == 0 && !e.writes_slot, "pad: an empty slot pads at cell 0, keep 0");
        const wp::PadRow l = wp::pad_row(true, true, 4095, 4096);
        check(l.pos == 4095 && l.keep == 0, "pad: the last cell is past a history one short of the context");
        for (const wp::PadRow& o : {wp::pad_row(true, true, 4096, 4096), wp::pad_row(false, true, 37, 4096),
                                    wp::pad_row(true, false, 37, 4096), wp::pad_row(true, true, -1, 4096)})
            check(o.pos == 0 && o.keep == 1 && o.writes_slot, "pad: full slot, setting off or no skip: the old row");
        bool on = false;
        std::string why;
        check(wp::parse_pad_keeps_idle(nullptr, on, why) && on == wp::kPadKeepsIdleDefault && on,
              "pad: STRATA_BATCH_PAD_KEEPS_IDLE defaults on");
        check(wp::parse_pad_keeps_idle("0", on, why) && !on, "pad: 0 turns it off");
        check(wp::parse_pad_keeps_idle("1", on, why) && on, "pad: 1 turns it on");
        check(!wp::parse_pad_keeps_idle("2", on, why) && why.find("STRATA_BATCH_PAD_KEEPS_IDLE") != std::string::npos,
              "pad: a bad value is refused by name");
    }
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
