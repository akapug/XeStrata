# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Real HTTP handler -> Service -> fake stdio worker; no sockets, model or GPU."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from serve.frontend import ChatTemplate, openai_to_messages
from serve.route_evidence import join_records
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, make_handler
from serve.test_parallel import FAKE_BATCH
import serve.server as server


class MemoryConnection:
    def __init__(self, request):
        self.input, self.output = io.BytesIO(request), io.BytesIO()

    def makefile(self, *args):
        return self.input

    def sendall(self, data):
        self.output.write(data)


class RouteEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workers = []
        self.addCleanup(self.close_workers)
        self.a = self.start_worker("a")
        self.tok = ByteTokenizer()
        self.svc = Service(self.a, self.tok, ChatTemplate(Path(__file__).parent / "test_chat_template.jinja"))
        self.svc.api_monitor = True
        self.svc.api_key = "fakekey"
        self.expected = self.a.route_identity()
        self.body = {"model": self.svc.model, "messages": [{"role": "user", "content": "unique fixture input"}],
                     "temperature": 0, "seed": 1, "stream": False, "reasoning_effort": "none"}

    def close_workers(self):
        for worker in self.workers:
            worker.unload()

    def start_worker(self, name):
        script = Path(self.tmp.name) / (name + ".py")
        # Extend the existing protocol fixture, not the server protocol. Only the fake worker retains IDs.
        code = FAKE_BATCH.replace("import queue, sys, threading, time",
                                  "import hashlib, json, queue, sys, threading, time")
        code = code.replace("toks = reply(ids)",
                            'toks = (list(b"unfinished thought") if b"CONTINUATION" in pb '
                            'and b"</think>" not in pb else list(("sentinel-" + '
                            'hashlib.sha256(json.dumps([int(x) for x in ids]).encode())'
                            '.hexdigest()[:12]).encode())) + [257]')
        code = code.replace('log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\\n"); log.flush()',
                            'log.write(json.dumps([int(x) for x in ids]) + "\\n"); log.flush()')
        script.write_text(code)
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            worker = StrataEngine("fixture-strata", ["--log", str(Path(self.tmp.name) / (name + ".log"))])
        self.workers.append(worker)
        return worker

    def post(self, body=None, handler=None):
        body = self.body if body is None else body
        data = json.dumps(body).encode()
        request = (b"POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\n"
                   b"Authorization: Bearer fakekey\r\nContent-Type: application/json\r\nContent-Length: " +
                   str(len(data)).encode() + b"\r\n\r\n" + data)
        conn = MemoryConnection(request)
        handler = handler or make_handler(self.svc)
        # The network-disconnect watcher is the only socket-specific operation skipped by this in-memory client.
        with mock.patch.object(handler, "_watch_client"):
            handler(conn, ("127.0.0.1", 0), None)
        head, raw = conn.output.getvalue().split(b"\r\n\r\n", 1)
        self.assertIn(b" 200 ", head)
        return json.loads(raw)

    def records(self):
        return [self.svc.request_records(row["id"]) for row in self.svc.request_records()]

    def join(self, response, records=None, boundary=None, worker=None, body=None):
        return join_records([self.body if body is None else body], [response],
                            self.records() if records is None else records,
                            [] if boundary is None else boundary, self.expected if worker is None else worker)

    def test_real_worker_input_sentinel_output_and_monitor_join(self):
        response = self.post()
        ids = json.loads((Path(self.tmp.name) / "a.log").read_text().splitlines()[0])
        messages, tools, kwargs = openai_to_messages(self.body)
        rendered = self.svc.template.render(messages, tools=tools, add_generation_prompt=True, **kwargs)
        self.assertEqual(ids, self.tok.encode(rendered, parse_special=True))
        sentinel = "sentinel-" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
        self.assertEqual(response["choices"][0]["message"]["content"], sentinel)
        joined = self.join(response)
        route = joined[0]["engine_route"]
        self.assertEqual(route["worker"], self.expected)
        self.assertEqual(route["calls"], 1)
        self.assertEqual(route["tokens"], len(sentinel) + 1)  # includes native stop, not heartbeat
        self.assertTrue(route["stable"])
        self.assertNotIn("fakekey", json.dumps(self.records()))

    def test_a_kept_alive_b_is_actual_service_worker(self):
        self.svc.engine = self.start_worker("b")
        response = self.post()
        self.assertTrue(self.a.alive())
        self.assertEqual(self.a.route_identity(), self.expected)
        self.assertFalse((Path(self.tmp.name) / "a.log").read_text())
        with self.assertRaisesRegex(ValueError, "chosen worker"):
            self.join(response)
        self.join(response, worker=self.svc.engine.route_identity())

    def test_bound_generate_routes_to_b_while_service_still_holds_a(self):
        b = self.start_worker("b")
        with mock.patch.object(self.a, "generate", b.generate):
            response = self.post()
        self.assertIs(self.svc.engine, self.a)
        self.assertTrue(self.a.alive())
        with self.assertRaisesRegex(ValueError, "chosen worker"):
            self.join(response)

    def test_canned_handler_reply_with_a_and_pipes_alive_is_refused(self):
        handler = make_handler(self.svc)
        canned = {"id": "canned", "choices": [{"finish_reason": "stop", "message": {"content": "sentinel"}}],
                  "usage": {"completion_tokens": 8}}
        with mock.patch.object(handler, "_openai", lambda h, req: h._json(200, canned)):
            response = self.post(handler=handler)
        self.assertTrue(self.a.alive())
        self.assertEqual(self.a.route_identity(), self.expected)
        self.assertFalse((Path(self.tmp.name) / "a.log").read_text())
        with self.assertRaisesRegex(ValueError, "worker consumption"):
            self.join(response)

    def test_mock_route_cannot_attest_expected_live_worker(self):
        self.svc.engine = MockEngine(self.tok, "canned", max_context=4096)
        response = self.post()
        self.assertTrue(self.a.alive())
        with self.assertRaisesRegex(ValueError, "worker consumption"):
            self.join(response)

    def test_stale_duplicate_missing_evicted_and_unrelated_rows_refused(self):
        response = self.post()
        rows = self.records()
        bad = [(rows, [rows[0]["id"]]), (rows * 2, []), ([], []), (rows, ["evicted"])]
        extra = copy.deepcopy(rows[0])
        extra["id"] = "unrelated"
        bad.append((rows + [extra], []))
        for records, boundary in bad:
            with self.subTest(records=len(records), boundary=boundary), self.assertRaises(ValueError):
                self.join(response, records, boundary)

    def test_truncation_input_output_and_completion_mismatches_refused(self):
        response = self.post()
        changes = [(k + "_truncated", True) for k in ("input", "output", "reasoning", "response")]
        changes += [("input", "{}"), ("response", json.dumps({**response, "usage": {}})),
                    ("state", "generating"), ("http_status", 500), ("stream", True)]
        for key, value in changes:
            rows = copy.deepcopy(self.records())
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.join(response, rows)
        for key in ("input", "output", "reasoning", "response"):
            rows = copy.deepcopy(self.records())
            del rows[0][key + "_truncated"]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.join(response, rows)

    def test_generation_pipe_and_zero_consumption_refused(self):
        response = self.post()
        for key, value in (("stable", False), ("calls", 0), ("tokens", 0), ("tokens", True)):
            rows = copy.deepcopy(self.records())
            rows[0]["engine_route"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.join(response, rows)
        worker = {**self.expected, "stdin_pipe": self.expected["stdout_pipe"]}
        with self.assertRaisesRegex(ValueError, "ambiguous pinned pipes"):
            self.join(response, worker=worker)
        worker = {**self.expected, "stdin_pipe": [0, 1]}
        with self.assertRaisesRegex(ValueError, "chosen worker"):
            self.join(response, worker=worker)
        body2 = {**self.body, "messages": [{"role": "user", "content": "second unique input"}]}
        response2 = self.post(body2)
        rows = self.records()
        rows[0]["engine_route"]["worker"]["generation"] = "replacement"
        with self.assertRaisesRegex(ValueError, "generation changed"):
            join_records([self.body, body2], [response, response2], rows, [], self.expected)

    def test_native_retry_accumulates_calls_and_monitor_off_retains_nothing(self):
        body = {**self.body, "messages": [{"role": "user", "content": "NFSOLO"}]}
        response = self.post(body)
        route = self.join(response, body=body)[0]["engine_route"]
        self.assertEqual(route["calls"], 2)
        self.assertGreater(route["tokens"], 0)
        self.svc.api_monitor = False
        self.svc.request_trace.record = None
        count = len(self.svc.api_requests)
        self.post()
        self.assertEqual(len(self.svc.api_requests), count)

    def test_reasoning_continuation_counts_worker_tokens_not_injected_close(self):
        self.svc.reasoning_close_retry = True
        body = {**self.body, "messages": [{"role": "user", "content": "CONTINUATION"}],
                "reasoning_effort": "low"}
        response = self.post(body)
        route = self.join(response, body=body)[0]["engine_route"]
        self.assertEqual(route["calls"], 2)
        self.assertEqual(route["tokens"], len("unfinished thought") + 1 + len("sentinel-") + 12 + 1)
        self.assertIn("sentinel-", response["choices"][0]["message"]["content"])
        self.assertEqual(len((Path(self.tmp.name) / "a.log").read_text().splitlines()), 2)

    def test_generation_changes_on_restart_and_monitor_boundary_can_be_retained(self):
        first = self.post()
        boundary = [self.records()[0]["id"]]
        body2 = {**self.body, "messages": [{"role": "user", "content": "next unique input"}]}
        second = self.post(body2)
        self.join(second, boundary=boundary, body=body2)
        self.assertNotEqual(first["id"], second["id"])
        real = server.subprocess.Popen
        script = Path(self.tmp.name) / "a.py"
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.a.restart()
        self.assertNotEqual(self.expected["generation"], self.a.route_identity()["generation"])
        self.assertEqual(self.a.route_identity()["wrapper_pid"], os.getpid())


if __name__ == "__main__":
    unittest.main()
