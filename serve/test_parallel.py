# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""#465: "parallel": N - several requests decode together in the engine's batch slots (--batch N).

A fake engine (a Python script speaking the engine's GEN / BGEN / BT / BDONE / BADM / BSTOP lines) runs behind the
real StrataEngine and Service, so the server's side is tested without a GPU: requests up to N run at once, more
wait for a slot, /metrics shows the slots, each request's history row is its own, and a conversation's next turn
goes back to the slot that holds it."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, engine_args, parallel_args, serve

# The fake engine: one token every STEP seconds per active slot (a "window" serves every active slot at once);
# GEN (solo) streams T lines; BGEN reads the prompt (one T line, DONE), answers BADM and continues in the slot.
# Every reply is the bytes of "ok, done." then <|im_end|> (ByteTokenizer: 257).
FAKE_BATCH = r'''import queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
fail = "--fail-window" in args        # #997: the first window over two slots fails
WINDOW = "--window" in args           # homelab window-row prefill: BGEN answers BADM 2, the first token a BT
LAG = int(args[args.index("--stop-lag") + 1]) if "--stop-lag" in args else 0   # homelab: a BSTOP ends its slot this
lag = {}                                                                       # many windows later (no tokens meanwhile)
STEP = 0.02
CH = 32
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.1.39" + (f" batch_slots={fit}" if fit >= 2 else "") +
      (" slot_cache=1" if "--slotcache" in args else ""), flush=True)
print("READY 4096 stop", flush=True)
LONG = list(b"LONGREPLY")
def continued(ids):        # (the whole reply, how much of it the prompt already ends with: a request continued)
    ids = [int(x) for x in ids]
    long_one = any(ids[i:i + len(LONG)] == LONG for i in range(len(ids)))
    R = list(b"ok, " + b"la " * 15 + b"done." if long_one else b"ok, done.") + [257]
    return R, max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k])
def reply(ids):            # the rest of the reply after what the prompt already ends with
    R, k = continued(ids)
    return R[k:]
active = {}          # slot -> [tokens left, max_new, produced]
stopped = set()      # BSTOPped slots: they end "cancel"
# #879: a prompt with NFSOLO / NFADMIT ends its first GEN / BGEN with DONE ... nonfinite (nothing emitted); one with
# NFSLOT ends its slot with BDONE ... nonfinite after its second token
fired, nf_slots = set(), set()
def window():        # one batch window: every active slot one token
    if fail and len(active) >= 2:     # as the engine: the reason on stdout, then exit code 1
        print("ERR verify batch: layer 34 never rang (graph finished)", flush=True)
        sys.exit(1)
    for b in sorted(active):
        if lag.get(b):
            lag[b] -= 1
            continue
        left, max_new, produced = active[b]
        if b in nf_slots and produced >= 2:
            nf_slots.discard(b)
            print(f"BDONE {b} {produced} nonfinite 1.0", flush=True)
            del active[b]
            continue
        t = left.pop(0) if left else 257
        produced += 1
        print(f"BT {b} {t}", flush=True)
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            print(f"BDONE {b} {produced} {fin} 1.0", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
# DONE's 15th field: the prompt tokens the engine read (a continued prompt: the 1 it copies)
READF = "--readfield" in args
REUSE = "--reuse" in args     # DONE's reused field as the engine gives it: a continued prompt is all held but its last
def reused(ids):              # token (the request's own output included), a fresh one the first 3 tokens
    return 0 if not REUSE else len(ids) - 1 if continued(ids)[1] else 3
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        if b in active:
            active[b][1] = 0
            stopped.add(b)
            lag[b] = LAG
        continue
    if line.startswith(("GEN ", "BGEN ")):
        f = line.split()
        slot = int(f[1]) if f[0] == "BGEN" else None
        max_new = int(f[2] if f[0] == "BGEN" else f[1])
        ids = f[-1].split(",")
        if bytes(int(x) & 255 for x in ids).find(b"REFUSEME") >= 0:   # #1059: refused up front, then idle (no DONE)
            print("ERR refused: images are not enabled", flush=True)
            continue
        # #1994: the engine's own view of its slots differs: BUSYSLOT is refused once, BUSYALL always
        pbx = bytes(int(x) & 255 for x in ids)
        if slot is not None and (b"BUSYALL" in pbx or (b"BUSYSLOT" in pbx and "busyslot" not in fired)):
            fired.add("busyslot")
            print(f"ERR BGEN: no such free slot (--batch {slots}) or a bad max_new", flush=True)
            if log:
                log.write(f"REFUSED {slot}\n"); log.flush()
            continue
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        pb = bytes(int(x) & 255 for x in ids)
        mark = b"NFSOLO" if f[0] == "GEN" else b"NFADMIT"
        if mark in pb and mark not in fired:
            fired.add(mark)
            print(f"DONE 0 {len(ids)} 5.0 0.0 nonfinite 0 0 0", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        if slot is not None and b"NFSLOT" in pb:
            nf_slots.add(slot)
        toks = reply(ids)
        if WINDOW and slot is not None:   # the slot reads the prompt in its windows: nothing generated here
            print(f"DONE 0 {len(ids)} 0.0 0.0 length 0 0 {reused(ids)} 0 0 0 0 0.0 {len(ids) - reused(ids)} 0",
                  flush=True)
            active[slot] = [toks, max_new, 0]
            print(f"BADM {slot} 2", flush=True)
            continue
        stop.clear()
        # the prompt read: CH tokens a chunk, a PP line each; a BYIELD <s> gives way at a chunk boundary
        given = None
        for q in range(CH, len(ids) - 1, CH):
            time.sleep(STEP / 2)
            print(f"PP {q} {len(ids)} 1 1", flush=True)
            held = []
            while True:
                try:
                    l2 = lines.get_nowait()
                except queue.Empty:
                    break
                if l2 and l2.startswith("BYIELD "):
                    ys = int(l2.split()[1])
                    if len(ids) - 1 - q > CH and (slot is None or ys == slot) and given is None:
                        given = (ys, q)
                else:
                    held.append(l2)
            for l2 in held:
                lines.put(l2)
            if given:
                break
            if active and (q // CH) % 4 == 0:     # the slots decode between the prompt's chunks (as the engine)
                window()
        if given:
            if log:
                log.write(f"YIELD {given[0]} {given[1]}\n"); log.flush()
            print(f"YIELDED {given[0]} {given[1]}", flush=True)
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 {reused(ids)}", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        n = 1 if slot is not None else max_new
        out = 0
        for t in toks[:n]:
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        print(f"DONE {out} {len(ids)} 5.0 {out * STEP * 1000:.1f} {fin} 0 0 {reused(ids)}" +
              (f" 0 0 0 0 0.0 {len(ids) - reused(ids)}" if READF else ""), flush=True)
        if slot is not None:
            cont = fin == "length" and max_new > 1
            if cont:
                active[slot] = [toks[1:], max_new, 1]
            print(f"BADM {slot} {1 if cont else 0}", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
'''


class ParallelArgs(unittest.TestCase):
    def test_parallel_config_key(self):
        self.assertEqual(parallel_args({"parallel": 4}, []), ["--batch", "4"])
        self.assertEqual(parallel_args({"parallel": 1}, []), [])
        self.assertEqual(parallel_args({}, []), [])
        self.assertEqual(parallel_args({"parallel": 2}, ["--batch", "3"]), [])       # the args' own value wins
        self.assertEqual(parallel_args({"parallel": "4"}, []), [])                   # said and ignored
        self.assertEqual(parallel_args({"parallel": True}, []), [])
        self.assertEqual(parallel_args({"parallel": 12}, []), ["--batch", "12"])     # the engine warns and caps

    def test_research_parallel_operand_is_not_capped_by_warning_threshold(self):
        self.assertEqual(parallel_args({"parallel": 32}, []), ["--batch", "32"])
        for flag in ("--batch", "--slots"):
            args = [flag, "32"]
            with self.subTest(flag=flag):
                self.assertEqual(parallel_args({"parallel": 2}, args), [])
                self.assertEqual(engine_args({"args": args}), args)

    def test_parallel_above_the_window_says_so_unless_batch_mtp(self):
        import contextlib
        import io
        import os

        def said(args, env):
            buf = io.StringIO()
            old = os.environ.get("STRATA_BATCH_MTP")
            if env is None:
                os.environ.pop("STRATA_BATCH_MTP", None)
            else:
                os.environ["STRATA_BATCH_MTP"] = env
            try:
                with contextlib.redirect_stdout(buf):
                    r = parallel_args({"parallel": 12}, args)
            finally:
                if old is None:
                    os.environ.pop("STRATA_BATCH_MTP", None)
                else:
                    os.environ["STRATA_BATCH_MTP"] = old
            self.assertEqual(r, ["--batch", "12"])
            return "at most" in buf.getvalue()
        self.assertTrue(said([], None))                  # 0.1.39: the window holds 8 rows
        self.assertTrue(said([], "0"))
        self.assertFalse(said(["--batch-mtp"], None))    # --batch-mtp waves more through the window
        self.assertFalse(said([], "1"))

        args = engine_args({"args": ["--pack", "p"], "parallel": 2})
        self.assertEqual(args[-2:], ["--batch", "2"])


class PickSlot(unittest.TestCase):
    def engine(self, n):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_used = [0.0] * n
        e.slot_live = [None] * n
        e.slot_group = [0] * n
        return e

    def test_the_slot_that_holds_the_conversation(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5, 6]), 2)       # the longest held start
        self.assertEqual(e.pick_slot([1, 2, 3, 9]), 0)
        self.assertEqual(e.pick_slot([7, 8]), 1)                    # no match: the empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8]), 0)                    # then the one used longest ago
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([7, 8]))
        e.slot_busy = [False] * 3
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)                 # a held prompt is never the WHOLE prompt

    def test_uniform_contexts_do_not_enter_mixed_slot_fallback_policy(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        cases = [([1, 2, 3, 4, 5, 6], [False] * 3, 2),
                 ([7, 8], [False] * 3, 1),
                 ([7, 8], [False, True, False], 0),
                 ([7, 8], [True] * 3, None)]
        for prompt, busy, expected in cases:
            e.slot_busy = busy
            for contexts in (None, [200] * 3):
                e.slot_ctx = contexts
                with self.subTest(prompt=prompt, busy=busy, contexts=contexts):
                    self.assertEqual(e.pick_slot(prompt, 10), expected)

    def test_retained_mixed_context_idle_big_slots_need_canonical_fallback(self):
        import inspect
        import textwrap

        # Retained mixed-context operands: two big slots idle, every small slot busy, no matching prefix.
        e = self.engine(32)
        e.max_context = 262144
        e.slot_ctx = [262144] * 2 + [131072] * 30
        e.slot_busy = [False] * 2 + [True] * 30
        prompt = [1, 2, 3]
        before = list(e.slot_busy)
        self.assertEqual(e.pick_slot(prompt, 10), 0)
        self.assertEqual(e.slot_busy, before)
        e.slot_busy[0] = True
        self.assertEqual(e.pick_slot(prompt, 10), 1)
        # Mutate only our canonical method's fallback, never execute retained deployment DATA.
        source = textwrap.dedent(inspect.getsource(StrataEngine.pick_slot))
        fallback = "] or fits"
        self.assertEqual(source.count(fallback), 1)
        scope = dict(StrataEngine.pick_slot.__globals__)
        exec(compile(source.replace(fallback, "]"), "<picker-without-fallback>", "exec"), scope)
        self.assertIsNone(scope["pick_slot"](e, prompt, 10))
        self.assertEqual(e.slot_busy, [True, False] + [True] * 30)

    def test_a_rewritten_next_turn_goes_to_the_slot_of_its_chat(self):
        # #2518: two chats in two slots; each next turn renders the history again but not the reply it decoded or
        # the assistant header, so the held stream (prompt + reply) is no start of it.  The engine's turn checkpoint
        # sits at the last <|im_start|> of the old prompt: the prompt that agrees up to there belongs to that slot.
        # Before, both went to the least recently used slot in turn, and the first one overwrote the other chat's
        # cache.
        T = StrataEngine.TURN_ID
        hist_a, hist_b = list(range(1000, 1600)), list(range(2000, 2600))
        held_a = hist_a + [T, 7, 8, 9] + [50, 51, 52]                  # prompt (ends with the assistant header) + reply
        held_b = hist_b + [T, 7, 8, 9] + [60, 61]
        e = self.engine(2)
        e.slot_held = [held_a, held_b]
        e.slot_used = [10.0, 20.0]                                      # slot 0 was used longer ago
        next_a = hist_a + [T, 5, 6, 7, 8, 9] + list(range(3000, 3100))  # the history again, then new turns
        next_b = hist_b + [T, 5, 6, 7, 8, 9] + list(range(3000, 3100))
        self.assertEqual(e.pick_slot(next_b), 1)                        # the LRU slot 0 holds the other chat
        self.assertEqual(e.pick_slot(next_a), 0)
        e.slot_busy[0] = True                                           # a busy slot is never picked
        self.assertEqual(e.pick_slot(next_a), 1)
        e.slot_busy[0] = False
        # a chat that agrees with neither (a new one, or one that rewrote its history) keeps the old order
        self.assertEqual(e.pick_slot(list(range(5000, 5200))), 0)
        self.assertEqual(e.pick_slot(hist_a[:300] + [T, 5, 6] + list(range(3000, 3100))), 0)
        e.slot_held[0] = []                                             # an empty slot still comes before evicting one
        self.assertEqual(e.pick_slot(list(range(5000, 5200))), 0)
        # the whole held stream continued is still the longer match
        e.slot_held = [held_a, held_b + [T, 1, 2]]
        self.assertEqual(e.pick_slot(held_b + [T, 1, 2, 3, 4]), 1)

    def test_batch_groups_spread_requests_over_the_groups(self):
        # #1249: with --batch-groups 2 over 4 slots (groups {0,1} and {2,3}) a second request goes to the other group,
        # and a short held prefix does not pull it back into the busy one
        e = self.engine(4)
        e.slot_groups = 2
        e.slot_group = [0, 0, 1, 1]
        e.slot_busy[0] = True
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5]), 2)
        e.slot_held[1] = list(range(1, 601))                       # a long prefix is worth the unbalanced group
        self.assertEqual(e.pick_slot(list(range(1, 700))), 1)
        e.slot_groups, e.slot_group = 1, [0] * 4                   # without groups: any held prefix, as before
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5]), 1)

    def test_mixed_slot_contexts_prefer_small_slots_and_keep_long_conversation_affinity(self):
        # --slot-context 100 --big-slots 1 (max context 200): slot 0 holds 200 cells, slots 1-2 hold 100.  A short
        # request prefers a small slot, but can use an idle big one when the small slots are busy.  A request that
        # needs more than 100 takes only the big one (else waits); its next turn keeps the held prefix.
        e = self.engine(3)
        e.max_context = 200
        e.slot_ctx = [200, 100, 100]
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 1)              # small: the small slots, never the big one
        e.slot_busy[1] = e.slot_busy[2] = True
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 0)              # small slots full: borrow the idle big one
        e.slot_busy[1] = e.slot_busy[2] = False
        long = list(range(1000, 1080))
        self.assertEqual(e.pick_slot(long, 40), 0)                  # 80 + 40 + 8 > 100: only the big slot fits
        e.slot_busy[0] = True
        self.assertIsNone(e.pick_slot(long, 40))                    # big slot busy: it waits for it
        e.slot_busy[0] = False
        e.slot_held[0] = long + [7, 8]                              # the main agent lives in the big slot now
        self.assertEqual(e.pick_slot(long + [7, 8, 9], 5), 0)        # its next turn goes back there even when short
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 1)              # a new short chat still prefers a small slot
        # open-ended (no max_tokens: max_new is the whole context's room) takes a small slot and ends where it does,
        # as long as the slot leaves SLOT_ROOM_MIN to answer in
        e.slot_held[0] = []

        def room(p):
            return 200 - 8 - len(p)

        with mock.patch("serve.server.SLOT_ROOM_MIN", 16):
            self.assertEqual(e.pick_slot([1, 2, 3], room([1, 2, 3])), 1)
            self.assertEqual(e.slot_left(1, [1, 2, 3], room([1, 2, 3])), 100 - 8 - 3)
            self.assertEqual(e.slot_left(0, [1, 2, 3], room([1, 2, 3])), room([1, 2, 3]))
            self.assertEqual(e.slot_left(1, [1, 2, 3], 10), 10)            # max_tokens given: never cut
            self.assertEqual(e.pick_slot(long, room(long)), 0)               # 80 + 16 + 8 > 100: the big slot
        e.slot_ctx = [200, 200, 200]                                # one context: as before
        e.slot_busy[1] = e.slot_busy[2] = True
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 0)

    def test_short_request_borrows_only_the_idle_big_slot_without_preempting(self):
        # The stranded mixed-context shape: one big slot busy, one idle, every small slot busy.
        e = self.engine(4)
        e.max_context = 200
        e.slot_ctx = [200, 200, 100, 100]
        e.slot_busy = [True, False, True, True]
        before = list(e.slot_busy)
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 1)
        self.assertEqual(e.slot_busy, before)                       # the caller, not the picker, reserves it
        e.slot_busy[1] = True                                     # the short request now owns the borrowed slot
        long = list(range(1000, 1080))
        self.assertIsNone(e.pick_slot(long, 40))                   # no eviction of either running big request
        e.slot_busy[2] = False
        self.assertIsNone(e.pick_slot(long, 40))                   # a free small slot cannot hold its budget
        self.assertEqual(e.slot_left(1, long, 40), 40)              # an explicit budget is not shortened
        e.slot_busy[1] = False                                    # the borrower has finished
        self.assertEqual(e.pick_slot(long, 40), 1)
        e.slot_busy = [False, True, True, True]                    # not an assumption that big slot 1 is idle
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 0)
        e.slot_busy[0] = True
        self.assertIsNone(e.pick_slot([1, 2, 3], 10))

    def test_idle_big_slot_fallback_keeps_empty_then_lru_order(self):
        e = self.engine(4)
        e.max_context = 200
        e.slot_ctx = [200, 200, 100, 100]
        e.slot_busy = [False, False, True, True]
        e.slot_held[0] = [8, 9]
        e.slot_used = [0.0, 10.0, 0.0, 0.0]
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 1)              # empty before an older cached slot
        e.slot_held[1] = [6, 7]
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 0)              # both cached: least recently used
        e.slot_used[0] = 20.0
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 1)
        e.slot_held[0] = [1, 2]
        e.slot_busy[2] = False
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 0)              # normal held-prefix affinity still wins
        e.slot_held[0] = []
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 2)              # otherwise keep the free small preference

    def test_idle_big_slot_fallback_keeps_group_balance_and_hole_order(self):
        e = self.engine(6)
        e.max_context = 200
        e.slot_ctx = [200, 200, 200, 200, 100, 100]
        e.slot_groups = 2
        e.slot_group = [0, 0, 0, 1, 1, 1]
        e.slot_busy = [False, True, False, False, True, True]
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 0)              # less busy group before group 1
        e.slot_order = [2, 0, 1, 3, 4, 5]
        e.slot_gs = 3
        self.assertEqual(e.pick_slot([1, 2, 3], 10), 0)              # hole before the earlier idle tail

    def test_idle_big_slot_fallback_never_changes_fit_or_explicit_budget(self):
        e = self.engine(4)
        e.max_context = 200
        e.slot_ctx = [200, 150, 100, 100]
        e.slot_busy = [True, False, True, True]
        prompt = list(range(80))
        self.assertIsNone(e.pick_slot(prompt, 70))                  # 80 + 70 + slack exceeds every free slot
        e.slot_busy[0] = False
        self.assertEqual(e.pick_slot(prompt, 70), 0)
        self.assertEqual(e.slot_left(0, prompt, 70), 70)
        e.slot_busy = [True, True, False, False]
        self.assertIsNone(e.pick_slot(prompt, 40))                  # never shrink the budget to fit a small slot
        e.slot_busy = [False] * 4
        self.assertIsNone(e.pick_slot(list(range(201)), 0))         # no fitting slot even when every slot is idle
        e.slot_busy[2] = e.slot_busy[3] = True
        with mock.patch("serve.server.SLOT_ROOM_MIN", 16):
            self.assertEqual(e.pick_slot([1, 2, 3], 189), 0)        # open-ended fallback retains its slot's room
            self.assertEqual(e.slot_left(0, [1, 2, 3], 189), 189)

    def test_a_request_waits_for_its_own_ending_slot(self):
        # homelab: the thinking budget ends a pass and sends the conversation again at once; its slot is still ending
        # (BSTOP sent, BDONE due).  With `own` the picker waits for it (None) instead of overwriting slot 1's chat.
        e = self.engine(2)
        mine, other = list(range(1000, 1600)), list(range(5000, 5700))
        e.slot_busy[0], e.slot_pending, e.slot_held[1] = True, {0: mine}, other
        prompt = mine + [7, 8, 9]
        self.assertEqual(e.pick_slot(prompt), 1)                        # without `own`: as before
        self.assertIsNone(e.pick_slot(prompt, 0, True))
        e.slot_pending = {0: list(range(9000, 9600))}                   # another chat's ending slot: no wait
        self.assertEqual(e.pick_slot(prompt, 0, True), 1)
        e.slot_pending = {0: mine[:100]}                                # under SLOT_PREFIX_MIN: not worth a wait
        self.assertEqual(e.pick_slot(mine[:100] + [7], 0, True), 1)
        e.slot_pending, e.slot_held[1] = {0: mine}, mine + [7]          # a free slot that holds more: that one
        self.assertEqual(e.pick_slot(prompt + [10], 0, True), 1)

    def test_an_empty_slot_before_evicting_another_chat(self):
        # homelab: a hole (--batch-groups) or the less busy group never wins over an empty slot by evicting a chat
        e = self.engine(4)
        e.slot_groups, e.slot_group, e.slot_gs, e.slot_order = 2, [0, 0, 1, 1], 2, [0, 2, 1, 3]
        e.slot_busy[1] = True
        e.slot_held[0] = [8, 9]                                         # slot 0: a hole that holds another chat
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)
        e.slot_gs = 0
        e.slot_busy = [False, False, True, False]
        e.slot_held[1] = [6, 7]                                         # group 0 idle but all held; group 1 busier
        self.assertEqual(e.pick_slot([1, 2, 3]), 3)
        e.slot_held[3] = [4, 5]                                         # nothing empty: as before
        self.assertIn(e.pick_slot([1, 2, 3]), (0, 1))

    def test_slots_view(self):
        e = self.engine(2)
        e.slot_held[1] = [1, 2]
        e.slot_live[0] = {"slot": 0, "state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        v = e.slots_view()
        self.assertEqual(v[0]["state"], "decoding")
        self.assertEqual(v[0]["generated"], 4)
        self.assertEqual(v[1], {"slot": 1, "state": "idle", "held_tokens": 2})


class ParallelService(unittest.TestCase):
    """The real StrataEngine and Service over HTTP, the fake engine behind them."""

    def start(self, slots, fit=None, slot_cache=False, fail=False, reuse=False, more=()):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        extra = ["--batch", str(slots)] + (["--fit", str(fit)] if fit is not None else []) + ["--log", str(self.log)]
        extra += ["--slotcache"] if slot_cache else []
        extra += ["--fail-window"] if fail else []
        extra += ["--reuse"] if reuse else []
        extra += list(more)
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "test_chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens=64):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def message(self, text, max_tokens=64):
        body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens}
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def _two_at_once(self, text_b):
        """A long request first, then one that has to go to a batch slot (the engine is busy with the first)."""
        out = {}

        def go(k, text):
            try:
                out[k] = ("ok", self.chat(text, max_tokens=64))
            except urllib.error.HTTPError as e:
                out[k] = ("http", e.code, e.read().decode(), dict(e.headers))
        ta = threading.Thread(target=go, args=("a", "LONGREPLY please"))
        ta.start()
        time.sleep(0.15)
        tb = threading.Thread(target=go, args=("b", text_b))
        tb.start()
        ta.join(30)
        tb.join(30)
        return out

    def test_bgen_refused_by_the_engine_is_retried_not_a_400(self):
        """#1994: the engine refuses a BGEN ("no such free slot") although the server saw the slot free: the request
        goes to another slot (or waits) and is answered; the same conversation's next turn works too."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.2
        out = self._two_at_once("BUSYSLOT hello")
        self.assertEqual(out["b"][0], "ok", out["b"])
        self.assertIn("REFUSED", self.log.read_text())
        again = self.chat("BUSYSLOT hello")                     # the conversation is not poisoned
        self.assertEqual(again["choices"][0]["message"]["content"], "ok, done.")
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])

    def test_bgen_refused_for_good_is_a_retryable_503_and_frees_the_slots(self):
        """#1994: when the engine keeps refusing, the answer is a 503 (retryable), never a 400, and no slot stays
        busy."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.05
        out = self._two_at_once("BUSYALL hello")
        self.assertEqual(out["b"][0], "http", out["b"])
        self.assertEqual(out["b"][1], 503, out["b"])
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])
        self.assertEqual(self.chat("fine again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_a_burst_of_connections_is_not_reset(self):
        """30-40 clients at once got "connection reset by peer" with the listen backlog of 5 (4 x R9700 burst): the
        server listens with a deep backlog (STRATA_HTTP_BACKLOG, 256), so 60 simultaneous requests all get answers."""
        import serve.server as server
        self.assertGreaterEqual(server.Server.request_queue_size, 64)
        self.start(2)
        errs, ok = [], []

        def one(i):
            try:
                self.chat(f"burst {i}", max_tokens=8)
                ok.append(i)
            except Exception as e:                      # noqa: BLE001
                errs.append(repr(e))
        th = [threading.Thread(target=one, args=(i,)) for i in range(60)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        self.assertEqual((len(ok), errs[:2]), (60, []))

    def test_stop_strings_in_a_batch_slot(self):
        """#454: a stop string cuts the answer in --batch mode too, and the slot is freed for the next request."""
        self.start(2)
        body = {"messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64, "reasoning_effort": "none",
                "stop": ["la la"]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            c = json.loads(r.read().decode())["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok, ", "stop"))
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_turns_batching_off(self):
        self.start(4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")

    def test_concurrent_requests_share_the_slots(self):
        self.start(2)
        # the first one decodes long enough for the others to arrive while it runs (also on a busy PC)
        texts = ["first question LONGREPLY", "second question, longer", "a third one waits for a slot"]
        reply = {t: "ok, " + "la " * 15 + "done." if "LONGREPLY" in t else "ok, done." for t in texts}
        out, errors, seen = {}, [], []

        def go(t):
            try:
                out[t] = self.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            m = self.get("/metrics")["live"]
            seen.append(m)
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        for t in texts:
            self.assertEqual(out[t]["choices"][0]["message"]["content"], reply[t])
        # the slots were reported, and a request was admitted while another decoded in a slot (the fake logs how many
        # slots were active at each admission)
        self.assertTrue(any(m.get("parallel") == 2 and len(m.get("slots", [])) == 2 for m in seen))
        admitted = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertIn(1, admitted)
        # every request recorded its own row; nothing is left running
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})
            self.assertFalse(self.svc.status["busy"])
        self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                         sorted(len(reply[t]) + 1 for t in texts))
        # never more than two in the slots at once (the fake logs how many were active at each admission)
        active = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertTrue(all(a <= 1 for a in active), active)

    def race(self, first, second, delay=0.15, max_tokens=64):
        """`first` sent, `second` after `delay` seconds; -> {text: (reply, seconds it took)}"""
        res = {}

        def go(t):
            t0 = time.time()
            r = self.chat(t, max_tokens=max_tokens)
            res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
        a = threading.Thread(target=go, args=(first,))
        a.start()
        time.sleep(delay)
        go(second)
        a.join(30)
        return res

    def test_a_long_prompt_gives_way_to_a_short_one(self):
        """#656: a long solo prompt read gives way at a chunk boundary; the short request is answered first, then the
        long one goes on (in the slot its part waited in) and is answered too."""
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        # the long read takes ~1 s (3,000 tokens, 32 a chunk, 10 ms a chunk): the short one did not wait for it
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)
        self.assertFalse(any(self.engine.slot_busy))

    def test_three_requests_long_long_short_do_not_deadlock(self):
        """ENGINE_REVIEW finding 1 (0.1.41 check): two long prompts fill both slots, a short one arrives a moment later
        and wants one of them (the yield path).  The reviewer's repro: long, long, short at 0.25 s steps.  Every
        request must be answered; none may wait on the control lock that the yielding read holds."""
        self.start(2)
        texts = ["long " * 600, "lung " * 600, "short"]
        res, errors = {}, []

        def go(t):
            t0 = time.time()
            try:
                r = self.chat(t, max_tokens=64)
                res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
            except Exception as e:      # noqa: BLE001 - a timeout is the failure this test looks for
                errors.append((t[:6], repr(e)))
        threads = []
        for t in texts:
            th = threading.Thread(target=go, args=(t,))
            th.start()
            threads.append(th)
            time.sleep(0.25)
        for th in threads:
            th.join(45)
        self.assertFalse(any(th.is_alive() for th in threads), "a request never finished (deadlock)")
        self.assertEqual(errors, [])
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 3)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)

    def test_an_admission_gives_way_too(self):
        """The same while another request decodes in a slot: the long one is being admitted, a short one waits."""
        self.start(3)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_no_way_given_to_a_long_one(self):
        """Two prompts of about the same length: no BYIELD (it only goes to a much shorter one)."""
        self.start(2)
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_left_alone_it_goes_back_to_the_solo_path(self):
        """A request decoding in a slot whose neighbour finished goes back to the solo path (MTP drafts), continued
        from its slot - only with an engine that keeps the slots' conversations (INFO slot_cache=1)."""
        self.start(2, slot_cache=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(res["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        # solo, then both in slots, then the long one alone again: a GEN after the BGENs
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN", kinds)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)

    def test_next_turn_goes_back_to_its_slot(self):
        self.start(2)
        a = "conversation A, first turn"
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat(a)
        th.join(30)
        deadline = time.time() + 10                     # a slot left at its stop token is freed at its BDONE
        while not [h for h in self.engine.slot_held if h] and time.time() < deadline:
            time.sleep(0.02)
        held = [h for h in self.engine.slot_held if h]
        self.assertTrue(held, "a finished slot request keeps what its slot holds")
        # the held tokens are the prompt and every token fed: all of the reply but its last token
        reply = list(b"ok, done.") + [257]
        mine = [h for h in held
                if h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation A") >= 0]
        self.assertEqual(len(mine), 1, held)
        slot = self.engine.slot_held.index(mine[0])
        self.assertEqual(self.engine.pick_slot(mine[0] + [10, 11]), slot)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")

    def test_window_prefill_admission_streams_its_first_token_as_a_bt(self):
        """homelab: BADM <slot> 2 - the slot reads the prompt in its windows, so the admission says DONE with nothing
        generated and every token (the first one too) comes as a BT; the reply, the totals and what the slot holds
        are as for BADM 1."""
        self.start(2, more=("--window",))
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat("conversation W, first turn")
        th.join(30)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(r1["x"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        deadline = time.time() + 10
        while len([h for h in self.engine.slot_held if h]) < 2 and time.time() < deadline:
            time.sleep(0.02)
        reply = list(b"ok, done.") + [257]
        mine = [h for h in self.engine.slot_held
                if h and h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation W") >= 0]
        self.assertEqual(len(mine), 1, self.engine.slot_held)
        with self.svc.status_lock:
            self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                             sorted([len(reply), len(list(b"ok, " + b"la " * 15 + b"done.")) + 1]))
            self.assertEqual(self.svc.live_reqs, {})

    def flip_window_at_release(self):
        """Another admission takes ctl the moment this one lets go and writes its own BADM mode: model that by
        flipping the engine's _ctl_window inside ctl.release()."""
        e, real = self.engine, self.engine.ctl

        class Flip:
            def acquire(self, *a, **k):
                return real.acquire(*a, **k)

            def release(self):
                e._ctl_window = not e._ctl_window
                real.release()

            def locked(self):
                return real.locked()
        e.ctl = Flip()

    def held_reply(self, marker, reply):
        deadline = time.time() + 10
        while time.time() < deadline:
            mine = [h for h in self.engine.slot_held
                    if h and h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(marker) >= 0]
            if mine:
                return mine
            time.sleep(0.02)
        return []

    def test_an_admission_keeps_its_own_window_mode_when_the_next_one_flips_it(self):
        """homelab (review): the mode an admission continues in is read under ctl.  Before,
        it was read after ctl.release(), so a next admission's BADM (or its reset) could make a window-row admission
        decode as a conventional one, or the converse, and the slot's held history tracked the wrong prefix."""
        reply = list(b"ok, done.") + [257]
        for window in (True, False):
            with self.subTest(window=window):
                self.start(2, more=("--window",) if window else ())
                self.flip_window_at_release()
                r1 = {}
                th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
                th.start()
                time.sleep(0.05)
                first = self.chat("conversation W, first turn")
                th.join(30)
                self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")
                self.assertEqual(r1["x"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
                self.assertEqual(len(self.held_reply(b"conversation W", reply)), 1, self.engine.slot_held)
                with self.svc.status_lock:
                    self.assertEqual(self.svc.live_reqs, {})
                self.tearDown()
                self.httpd = self.engine = self.tmp = None

    def own_slot_race(self, lag, wait_s):
        """homelab: conversation A decodes in slot 0 (slot 2 busy all along, so A takes the slot path), B waits idle in
        slot 1 for its next turn.  A's consumer leaves after 4 tokens (the thinking budget) and A comes back at once
        with a wrap-up; the engine ends slot 0 `lag` windows after the BSTOP.  -> (the re-submit's slot, seconds)"""
        self.start(3, more=("--stop-lag", str(lag)))
        e = self.engine
        e.slot_own_wait_s = wait_s
        with e.slot_cv:
            e.slot_busy[2] = True
            e.slot_claims.add(2)
        b_held = list(b"conversation B " * 50)
        e.slot_held[1] = list(b_held)

        def first(ids, n):
            g, out = e.generate(ids, 64, {}, threading.Event()), []
            for t in g:
                if t is not None:
                    out.append(t)
                if len(out) >= n:
                    break
            g.close()
            return out
        a = list(b"conversation A, a long history " * 25) + [StrataEngine.TURN_ID] + list(b"assistant LONGREPLY")
        out = first(a, 4)
        t0 = time.monotonic()
        first(a + out + list(b" wrap up."), 1)
        took = time.monotonic() - t0
        bgens = [x.split() for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertEqual([int(x[1]) for x in bgens[:1]], [0])
        if int(bgens[1][1]) == 0:
            self.assertEqual(e.slot_held[1], b_held)                    # B's chat is still there for its next turn
        return int(bgens[1][1]), took

    def test_a_resubmit_goes_back_to_its_own_ending_slot(self):
        """homelab: before, the re-submit took B's idle slot 1 at once (its own was not free yet): B's history was
        overwritten and A's whole prompt read again.  Now it waits for slot 0's BDONE and continues there."""
        slot, took = self.own_slot_race(lag=10, wait_s=2.0)
        self.assertEqual(slot, 0)
        self.assertLess(took, 2.0)

    def test_the_wait_for_the_own_slot_is_bounded(self):
        """homelab: a slot that takes longer than slot_own_wait_s to end is not waited for: after the bound the
        request takes a free slot as before (here BDONE would come ~4 s after the BSTOP)."""
        slot, took = self.own_slot_race(lag=200, wait_s=0.3)
        self.assertEqual(slot, 1)
        self.assertGreaterEqual(took, 0.3)
        self.assertLess(took, 3.0)

    def test_a_failed_window_names_its_error(self):
        # #997 #890: a batch window fails while every request reads its own slot's lines, so no request reads the
        # engine's ERR line before it exits; the server's note on the death gives it instead of a guess
        self.start(2, fail=True)
        errors = []

        def go(t):
            try:
                self.chat(t)
            except urllib.error.HTTPError as e:
                errors.append(e.code)

        threads = [threading.Thread(target=go, args=(t,)) for t in ("first question LONGREPLY", "second question")]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(len(errors), 2, errors)
        note = self.engine.death_note()
        self.assertIn("exited (code 1)", note)
        self.assertIn("verify batch: layer 34 never rang (graph finished)", note)

    def test_nonfinite_in_batch_mode(self):
        """#879 in --batch mode: DONE ... nonfinite on the solo path or at an admission (nothing emitted yet) sends the
        request again once; BDONE ... nonfinite in a slot ends the reply that had started, as "length"."""
        self.start(2)
        self.assertEqual(self.chat("NFSOLO alone")["choices"][0]["message"]["content"], "ok, done.")
        gens = [x for x in self.log.read_text().splitlines() if x.startswith("GEN")]
        self.assertEqual(len(gens), 2, gens)                              # sent again once
        res = self.race("first question LONGREPLY", "NFADMIT second")
        self.assertEqual(res["NFADMIT second"][0], "ok, done.")
        self.assertEqual(res["first question LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertTrue(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))
        out = {}

        def go(t):
            body = {"messages": [{"role": "user", "content": t}], "max_tokens": 64, "reasoning_effort": "none"}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out[t] = json.loads(r.read().decode())["choices"][0]
        a = threading.Thread(target=go, args=("third question LONGREPLY",))
        a.start()
        time.sleep(0.15)
        go("NFSLOT fourth")
        a.join(30)
        c = out["NFSLOT fourth"]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok", "length"))
        self.assertEqual(out["third question LONGREPLY"]["message"]["content"], "ok, " + "la " * 15 + "done.")
        # the engine dropped that slot's state: the server does not route the conversation's next turn to it
        self.assertFalse([h for h in self.engine.slot_held if bytes(t & 255 for t in h).find(b"NFSLOT") >= 0])
        with self.svc.status_lock:
            self.assertEqual(self.svc.live_reqs, {})

    def test_a_continued_request_reports_its_own_prompt_reuse(self):
        """A request STOPped on the solo path and continued in a slot (or back on the solo path) is sent again as its
        prompt + what it generated, and the engine holds all of that: the usage must still split the request's OWN
        prompt (input_tokens + cache_read_input_tokens = the prompt, input_tokens >= 1).  input_tokens 0 makes Claude
        Code count the prompt twice (message_start's input_tokens + cache_read_input_tokens): "Prompt is too long"
        at half the context."""
        self.start(2, slot_cache=True, reuse=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.message("LONGREPLY please", 200)))
        long.start()
        time.sleep(0.05)
        self.message("short", 200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertIn("BGEN", kinds)                    # it was continued: the case under test
        u = res["long"]["usage"]
        prompt = self.svc.history[[h["output_tokens"] for h in self.svc.history].index(
            max(h["output_tokens"] for h in self.svc.history))]["prompt_total"]
        self.assertGreaterEqual(u["input_tokens"], 1, u)
        self.assertEqual(u["input_tokens"] + u["cache_read_input_tokens"], prompt, u)
        self.assertEqual(u["cache_read_input_tokens"], 3, u)    # the first read's reuse, not the continuation's

    def test_a_slot_request_ended_by_its_stop_token_reports_the_slot_leg(self):
        """#1869: the consumer leaves at the stop token, before the BDONE: the request kept the admission's one token
        and its few ms.  Its figures are the slot leg's now (finish=stop was the broken case, finish=length read the
        BDONE)."""
        import serve.server as server
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        with mock.patch.object(server, "EOS_IDS", {257}):          # the fake engine's stop token
            a = threading.Thread(target=lambda: res.setdefault("a", self.chat("List three primes please", 200)))
            b = threading.Thread(target=lambda: res.setdefault("b", self.chat("Name two rivers in Germany", 200)))
            a.start()
            b.start()
            a.join(30)
            b.join(30)
        self.assertEqual(len(self.svc.history), 2)
        for h in self.svc.history:
            self.assertEqual(h["engine_generated"], h["output_tokens"], h)
            self.assertGreaterEqual(h["decode_ms"], 1.0, h)
        for r in res.values():
            t = r["timings"]
            self.assertEqual(t["prompt_n"] + t["cache_n"], r["usage"]["prompt_tokens"], t)
            self.assertEqual(t["predicted_n"], 10, t)
        self.assertFalse(any(self.engine.slot_busy))                 # the slots are free (no BDONE left for a wait)

    def test_a_request_moved_between_solo_and_slot_keeps_its_totals(self):
        """#1869/#1870: a request that read its prompt and decoded on the solo path, was STOPped when another arrived
        and went on in a slot (its admission copies the held prefix: prompt_read 1, one token) reports the prompt it
        read and the decode time and tokens of ALL its legs, not the last one's."""
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.25)                                   # a few solo tokens first
        self.chat("short", max_tokens=200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)   # solo, then continued in a slot: the case under test
        n = res["long"]["usage"]["completion_tokens"]
        row = max(self.svc.history, key=lambda h: h["output_tokens"])
        self.assertEqual(row["engine_generated"], n, row)
        self.assertGreater(row["decode_ms"], 20.0 * 3, row)          # the solo leg's STEP-long tokens are in it
        self.assertGreater(row["prompt_read"], 1, row)                # the first read, not the copy leg's 1
        t = res["long"]["timings"]
        self.assertEqual(t["prompt_n"] + t["cache_n"], res["long"]["usage"]["prompt_tokens"], t)
        self.assertEqual(t["cache_n"], 3, t)                          # the first read's reuse only
        self.assertGreater(t["predicted_ms"], 60.0, t)

    def test_a_leaked_busy_slot_does_not_hang_the_next_request(self):
        """#1603: a slot whose busy flag has no owner (no request holds it, no BDONE wait is running) kept every later
        request waiting for a slot nothing would free: running 0, queued 0, and the frozen-engine check never looked.
        The waiting request is now counted (admitting), the slot is given back after ORPHAN_SLOT_S, and it is
        answered."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.4
        with e.slot_cv:
            e.slot_busy[0] = e.slot_busy[1] = True       # the leak: busy, claimed by nobody, nothing ending them
        # the solo path needs a quiet engine; a busy flag sends the request to the slot path, which waits
        res, errs = {}, []

        def go():
            try:
                res["r"] = self.chat("after the leak", max_tokens=16)
            except Exception as ex:                       # noqa: BLE001
                errs.append(ex)
        t = threading.Thread(target=go)
        t0 = time.time()
        t.start()
        seen = []
        while t.is_alive() and time.time() - t0 < 20:
            live = self.get("/metrics")["live"]
            seen.append((live.get("running"), live.get("queued"), live.get("admitting")))
            time.sleep(0.05)
        t.join(5)
        self.assertEqual(errs, [])
        self.assertFalse(t.is_alive(), "the request hung on a slot nobody owns")
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        # while it waited it was visible: running as a request, and counted as waiting for a slot - never 0 / 0 / 0
        self.assertTrue(any(r == 1 and a == 1 for r, q, a in seen), seen)
        first = next(i for i, x in enumerate(seen) if sum(v or 0 for v in x))
        self.assertTrue(all(sum(v or 0 for v in x) for x in seen[first:-1]), seen)   # never invisible once it arrived
        self.assertEqual(e.admitting, 0)
        self.assertEqual(e.slot_claims, set())

    def test_an_owned_busy_slot_is_never_taken(self):
        """The leak check frees only slots with no owner: a slot a request holds stays busy however long it waits."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.0
        with e.slot_cv:
            e.slot_busy[0] = True
            e.slot_claims.add(0)
            e.slot_busy[1] = True
            e.slot_releasing.add(1)
            self.assertEqual(e._reap_orphan_slots(), [])
            self.assertEqual(e.slot_busy, [True, True])
            e.slot_claims.discard(0)
            self.assertEqual(e._reap_orphan_slots(), [0])
            self.assertEqual(e.slot_busy, [False, True])

    def test_a_parallel_request_does_not_wait_for_the_fifo_on_a_loaded_engine(self):
        """#1603: parallel requests used to take the FIFO (held by an image encode, /v1/vram, load) just to find the
        engine loaded; between 'queued' and 'running' they were neither.  Loaded: no fifo.  Unloaded: they wait for it
        and are counted as queued the whole time."""
        self.start(2)
        self.assertEqual(self.chat("warm")["choices"][0]["message"]["content"], "ok, done.")
        self.svc.fifo.acquire()
        try:
            r = self.chat("with the fifo held")             # would block here before
            self.assertEqual(r["choices"][0]["message"]["content"], "ok, done.")
        finally:
            self.svc.fifo.release()
        import serve.server as server
        script, real = Path(self.tmp.name) / "fake_strata.py", server.subprocess.Popen
        patch = mock.patch.object(server.subprocess, "Popen",      # the reload starts the fake again
                                  lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        self.engine.unload()
        self.svc.fifo.acquire()
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("r", self.chat("reloads", max_tokens=16)))
        t.start()
        try:
            deadline = time.time() + 10
            live = {}
            while time.time() < deadline:
                live = self.get("/metrics")["live"]
                if live.get("queued"):
                    break
                time.sleep(0.05)
            self.assertEqual(live.get("queued"), 1, live)    # waiting for the fifo: queued, not nothing
            self.assertEqual(live.get("running"), 0, live)
        finally:
            self.svc.fifo.release()
        t.join(60)
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.get("/metrics")["live"]["queued"], 0)


if __name__ == "__main__":
    unittest.main()
