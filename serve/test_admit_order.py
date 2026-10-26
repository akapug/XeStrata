# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/test_admit_order.py - STRATA_ADMIT_SHORT_FIRST: new requests take the control lines short prompts first
(StrataEngine._take_control / _admit_next; no process, no GPU).

    python -m unittest serve.test_admit_order -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import StrataEngine  # noqa: E402


def engine_without_process(n=2):
    """A StrataEngine with the admission state of a started one and no process: alive() is `e.up`."""
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = n
    e.slot_busy = [False] * n
    e.slot_claims, e.slot_releasing, e._orphan_seen = set(), set(), {}
    e.slot_cv = threading.Condition()
    e.waiting, e.admitting, e.wait_lens, e.ctl_epoch = 0, 0, [], 0
    e.ctl = threading.Lock()
    e.gen = 1
    e.up = True
    e.alive = lambda: e.up
    return e


class QueueCase(unittest.TestCase):
    """Requests that queue for the control lines while a long read holds them (helpers only, no tests)."""

    def setUp(self):
        self.e = engine_without_process(16)
        self.order, self.cancels = [], []
        self.order_lock = threading.Lock()

    def tearDown(self):
        for c in self.cancels:
            c.set()

    def take(self, name, plen, **kw):
        """A request waiting for the control lines; once it has them it notes its name and gives them back."""
        c = threading.Event()
        self.cancels.append(c)

        def go():
            gen = self.e._take_control(c, plen, **kw)
            try:
                while True:
                    next(gen)
            except StopIteration as s:
                if s.value:
                    with self.order_lock:
                        self.order.append(name)
                    time.sleep(0.05)                # its read
                    self.e.ctl.release()
        threading.Thread(target=go, daemon=True).start()

    def until(self, cond, s=5.0):
        end = time.time() + s
        while time.time() < end and not cond():
            time.sleep(0.01)
        self.assertTrue(cond())

    def queue(self, *reqs):
        """While a long read holds the control lines, the requests arrive in this order; then the read ends."""
        self.assertTrue(self.e.ctl.acquire(timeout=1))
        for name, plen in reqs:
            self.take(name, plen, new=True)
            self.until(lambda n=len(self.order) + sum(1 for _ in self.e.wait_lens): self.e.waiting >= 1)
            time.sleep(0.02)                        # (distinct arrival times)
        self.until(lambda: self.e.waiting == len(reqs))
        self.e.ctl.release()
        self.until(lambda: len(self.order) == len(reqs), s=15.0)
        return self.order


class AdmitOrderTest(QueueCase):
    def test_short_first_puts_the_chat_before_the_agent_reads(self):
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0):
            order = self.queue(("agent1", 120000), ("agent2", 190000), ("chat", 500), ("agent3", 150000))
        self.assertEqual(order[0], "chat")
        self.assertEqual(order[1:], ["agent1", "agent2", "agent3"])      # arrival order among the long ones

    def test_a_long_request_that_waited_too_long_goes_first(self):
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 0.2):
            self.assertTrue(self.e.ctl.acquire(timeout=1))
            self.take("agent", 120000, new=True)
            self.until(lambda: self.e.waiting == 1)
            time.sleep(0.4)                         # past STRATA_ADMIT_MAX_WAIT_S
            self.take("chat", 500, new=True)
            self.until(lambda: self.e.waiting == 2)
            self.e.ctl.release()
            self.until(lambda: len(self.order) == 2, s=15.0)
        self.assertEqual(self.order, ["agent", "chat"])

    def test_off_by_default(self):
        self.assertEqual(server.ADMIT_SHORT_FIRST if "STRATA_ADMIT_SHORT_FIRST" not in __import__("os").environ
                         else 0, 0)
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 0):
            order = self.queue(("agent1", 120000), ("chat", 500))
        self.assertEqual(sorted(order), ["agent1", "chat"])            # both get their turn, in any order

    def test_a_continuation_is_not_held_back(self):
        """A request already admitted once (new=False: promoted from the solo path) is not part of the order."""
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0):
            self.assertTrue(self.e.ctl.acquire(timeout=1))
            self.take("chat", 500, new=True)
            self.take("continuation", 190000)
            self.until(lambda: self.e.waiting == 2)
            self.e.ctl.release()
            self.until(lambda: len(self.order) == 2, s=15.0)
        self.assertEqual(sorted(self.order), ["chat", "continuation"])


class AdmitPolicyTest(QueueCase):
    """homelab STRATA_ADMIT_POLICY=shortest-new: the new part (the prompt less what a slot holds) orders the reads."""

    def test_default_is_fifo(self):
        self.assertEqual(server.ADMIT_POLICY_DEFAULT, "fifo")
        if "STRATA_ADMIT_POLICY" not in __import__("os").environ:
            self.assertEqual(server.ADMIT_POLICY, "fifo")

    def test_fifo_counts_whole_prompts(self):
        self.e.slot_held = [list(range(1000)), []]
        with mock.patch.object(server, "ADMIT_POLICY", "fifo"):
            self.assertEqual(self.e._admit_len(list(range(1500))), 1500)

    def test_new_part_is_the_prompt_less_the_longest_held_start(self):
        conv = list(range(25000))
        self.e.slot_held = [list(range(100)), conv, [7] * 30]
        with mock.patch.object(server, "ADMIT_POLICY", "shortest-new"):
            self.assertEqual(self.e._admit_len(conv + list(range(500))), 500)      # the agent's next turn
            self.assertEqual(self.e._admit_len([9] * 30000), 30000)                # a cold agent
            # a client renders its history again without the reply's thinking: the turn checkpoint still holds
            turn = conv[:20000] + [server.StrataEngine.TURN_ID] + conv[20000:]
            self.e.slot_held = [turn + [5, 6, 7]]
            self.assertEqual(self.e._admit_len(turn[:20001] + [8] * 300), 300 + 1)

    def test_shortest_new_part_first_then_arrival(self):
        with mock.patch.object(server, "ADMIT_POLICY", "shortest-new"), \
                mock.patch.object(server, "ADMIT_SHORT_FIRST", 0), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0):
            order = self.queue(("cold1", 30000), ("turn1", 530), ("cold2", 11000), ("turn2", 4800), ("turn3", 530))
        self.assertEqual(order, ["turn1", "turn3", "turn2", "cold2", "cold1"])

    def test_shortest_new_ages_a_long_read(self):
        with mock.patch.object(server, "ADMIT_POLICY", "shortest-new"), \
                mock.patch.object(server, "ADMIT_SHORT_FIRST", 0), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 0.2):
            self.assertTrue(self.e.ctl.acquire(timeout=1))
            self.take("cold", 30000, new=True)
            self.until(lambda: self.e.waiting == 1)
            time.sleep(0.4)                         # past STRATA_ADMIT_MAX_WAIT_S
            self.take("turn", 500, new=True)
            self.until(lambda: self.e.waiting == 2)
            self.e.ctl.release()
            self.until(lambda: len(self.order) == 2, s=15.0)
        self.assertEqual(self.order, ["cold", "turn"])

    def test_a_long_read_gives_way_to_a_short_new_part(self):
        with mock.patch.object(server, "ADMIT_POLICY", "shortest-new"):
            self.e.wait_lens.append([530, time.monotonic(), True, None, threading.Event()])
            self.assertTrue(self.e._shorter_waiting(30000))
            self.assertFalse(self.e._shorter_waiting(1000))


class ReadRequeueTest(unittest.TestCase):
    """Drive the real selector/admission generator with a clock and control-line boundary, not a model/GPU.

    Age includes another prompt's ownership. A due waiter wins only an AVAILABLE boundary, not a deadline.
    """
    def setUp(self):
        self.e = engine_without_process()
        self.now = 100.0
        self.patches = [mock.patch.object(server, "BATCH_READ_REQUEUE_MAX_S", 20.0),
                        mock.patch.object(server, "ADMIT_SHORT_FIRST", 0),
                        mock.patch.object(server.time, "monotonic", lambda: self.now)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.e.slot_cv.wait = mock.Mock(side_effect=self.tick)

    def tick(self, timeout=None):
        self.now += 10.0

    def waiter(self, age=0, plen=100000, new=False, epoch=None, cancel=None):
        e = [plen, self.now - age, new, epoch, cancel or threading.Event()]
        self.e.wait_lens.append(e)
        self.e.waiting += 1
        return e

    def drive(self, cancel=None, **kw):
        g = self.e._take_control(cancel or threading.Event(), 100000, **kw)
        self.addCleanup(g.close)
        return g

    def result(self, g, value):
        with self.assertRaises(StopIteration) as done:
            next(g)
        self.assertIs(done.exception.value, value)

    def test_named_default_is_off_and_yield_budget_unchanged(self):
        self.assertEqual(server.BATCH_READ_REQUEUE_MAX_S_DEFAULT, 0.0)
        self.assertEqual(self.e.YIELDS_MAX, 2)

    def test_no_waiter_leaves_decode_unowned(self):
        self.assertIsNone(self.e._admit_next())
        self.assertTrue(self.e.ctl.acquire(blocking=False))
        self.e.ctl.release()                         # the selector neither acquires ctl nor runs decode

    def test_new_requests_and_plain_continuations_do_not_become_due(self):
        new = self.waiter(age=1000, new=True)
        plain = self.waiter(age=1000)
        self.assertIs(self.e._admit_next(), new)
        self.assertTrue(self.e._control_turn(plain))
        self.assertTrue(self.e._control_turn(new))

    def test_before_threshold_keeps_after_epoch_yield(self):
        held = self.waiter(age=19.99, epoch=0)
        new = self.waiter(new=True)
        self.assertFalse(self.e._control_turn(held))
        self.assertTrue(self.e._control_turn(new))
        self.now += 0.01
        self.assertTrue(self.e._control_turn(held))
        self.assertFalse(self.e._control_turn(new))

    def test_due_read_overrides_short_first_and_after_epoch(self):
        held = self.waiter(age=20, epoch=0)
        chat = self.waiter(plen=5, new=True)
        plain = self.waiter()
        for short_first in (0, 8192):
            with self.subTest(short_first=short_first), mock.patch.object(server, "ADMIT_SHORT_FIRST", short_first):
                self.assertIs(self.e._admit_next(), held)
                self.assertTrue(self.e._control_turn(held))
                self.assertFalse(self.e._control_turn(chat))
                self.assertFalse(self.e._control_turn(plain))

    def test_same_shape_reads_choose_oldest_then_stable_tie(self):
        older = self.waiter(age=60, epoch=0)
        tie = self.waiter(age=60, epoch=0)
        younger = self.waiter(age=20, epoch=0)
        self.assertIs(self.e._admit_next(), older)
        self.assertFalse(self.e._control_turn(tie))
        self.e.wait_lens.pop(0)
        self.e.waiting -= 1
        self.assertIs(self.e._admit_next(), tie)
        self.e.wait_lens.pop(0)
        self.e.waiting -= 1
        self.assertIs(self.e._admit_next(), younger)

    def test_off_does_not_prioritize_due_held_read(self):
        held = self.waiter(age=1000, epoch=0)
        new = self.waiter(new=True)
        with mock.patch.object(server, "BATCH_READ_REQUEUE_MAX_S", 0):
            self.assertFalse(self.e._control_turn(held))
            self.assertTrue(self.e._control_turn(new))
            self.e.ctl_epoch = 1
            self.assertTrue(self.e._control_turn(held))

    def test_no_due_read_preserves_short_first(self):
        held = self.waiter(age=19, epoch=0)
        long = self.waiter(new=True)
        chat = self.waiter(plen=5, new=True)
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192):
            self.assertIs(self.e._admit_next(), chat)
            self.assertFalse(self.e._control_turn(long))
            self.assertFalse(self.e._control_turn(held))
            self.assertTrue(self.e._control_turn(chat))

    def test_due_continuation_acquires_without_epoch_change(self):
        self.waiter(new=True)
        g = self.drive(after_epoch=0)
        self.assertIsNone(next(g))                  # yields until queued age reaches the threshold
        self.assertIsNone(next(g))
        self.result(g, True)                       # due selection overrides after_epoch, avoiding self-deadlock
        self.assertEqual(self.e.ctl_epoch, 1)
        self.assertTrue(self.e.ctl.locked())
        self.assertEqual(self.e.waiting, 1)
        self.e.ctl.release()

    def test_post_acquire_rechecks_newly_due_read_for_every_competitor(self):
        for new in (False, True):
            with self.subTest(new=new):
                self.e = engine_without_process()
                self.e.slot_cv.wait = mock.Mock(side_effect=self.tick)
                held = self.waiter(age=19, epoch=0)
                lock = self.e.ctl
                calls = []

                def acquire(timeout=None):
                    calls.append(True)
                    self.now += 1                    # held becomes due while competitor waits in acquire
                    return lock.acquire(blocking=False)

                self.e.ctl = mock.Mock(acquire=acquire, release=lock.release)
                g = self.drive(new=new)
                self.assertIsNone(next(g))
                self.assertEqual(len(calls), 1)
                self.assertFalse(lock.locked())       # post-acquire rejected and released competitor
                self.assertIs(self.e._admit_next(), held)
                g.close()
                self.assertEqual(self.e.waiting, 1)

    def test_current_owner_is_not_preempted(self):
        self.e.ctl.acquire()
        lock = self.e.ctl

        def unavailable(timeout=None):
            self.now += 10
            return False

        self.e.ctl = mock.Mock(acquire=unavailable, release=mock.Mock(wraps=lock.release))
        g = self.drive(after_epoch=0)
        for _ in range(3):
            self.assertIsNone(next(g))
        self.assertTrue(lock.locked())                # threshold passage does not release somebody else's lock
        g.close()
        self.e.ctl.release.assert_not_called()
        lock.release()
        self.assertEqual(self.e.waiting, 0)

    def test_canceled_due_waiter_does_not_block_other_reads(self):
        held = self.waiter(age=1000, epoch=0)
        held[4].set()
        chat = self.waiter(new=True)
        self.assertIs(self.e._admit_next(), chat)
        self.assertTrue(self.e._control_turn(chat))

    def test_cancellation_before_and_during_acquire_releases_ownership(self):
        for during in (False, True):
            with self.subTest(during=during):
                cancel = threading.Event()
                lock = self.e.ctl
                if during:
                    def acquire(timeout=None):
                        cancel.set()
                        return lock.acquire(blocking=False)
                    self.e.ctl = mock.Mock(acquire=acquire, release=lock.release)
                else:
                    cancel.set()
                self.result(self.drive(cancel=cancel, after_epoch=0), False)
                self.assertFalse(lock.locked())
                self.assertEqual(self.e.waiting, 0)
                self.assertEqual(self.e.wait_lens, [])
                self.assertEqual(self.e.ctl_epoch, 0)

    def test_engine_death_removes_waiter(self):
        self.e.up = False
        g = self.drive(after_epoch=0)
        with self.assertRaises(server.EngineDied):
            next(g)
        self.assertEqual(self.e.waiting, 0)
        self.assertEqual(self.e.wait_lens, [])
        self.assertFalse(self.e.ctl.locked())

    def test_closed_generator_removes_only_its_own_equal_waiter(self):
        older = self.waiter(epoch=0)
        g = self.drive(cancel=older[4], after_epoch=0)
        self.assertIsNone(next(g))
        own = self.e.wait_lens[1]
        self.assertEqual(older, own)                  # identical list values must not delete the older waiter
        g.close()
        self.assertEqual(self.e.waiting, 1)
        self.assertIs(self.e.wait_lens[0], older)


if __name__ == "__main__":
    unittest.main()
