# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prepared gate controls: synthetic collector/HTTP only, never Docker or inference."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from serve import served_greedy16 as gate
from serve.route_evidence import join_records


class Reply(io.BytesIO):
    status = 200


class Gate(unittest.TestCase):
    def identity(self, enabled="0"):
        worker = {"wrapper_pid": 7, "worker_pid": 8, "stdin_pipe": [1, 2], "stdout_pipe": [1, 3]}
        process = {"worker": worker, "listener": {"pid": 7, "start": 100}, "child": {"pid": 8, "start": 101},
                   "engine": {"sha256": "d" * 64}, "interpreter": {"sha256": "b" * 64}, "closure_sha256": "c" * 64}
        return {"container": "f" * 64, "started": "start", "image": "sha256:" + "b" * 64,
                "source": "c" * 40, "engine_sha256": "d" * 64, "launch_sha256": "e" * 64,
                "contract_sha256": "a" * 64, "settings": {"STRATA_BATCH_DENSE_XMX": enabled},
                "worker": worker, "process": process}

    def receipt(self, enabled="0"):
        bodies = gate.requests("m")
        identity = self.identity(enabled)
        responses = [{"id": "reply-" + str(i), "choices": [{"finish_reason": "stop", "message":
                      {"content": "answer-" + str(i), "reasoning_content": "thought"}}],
                      "usage": {"completion_tokens": 8}} for i in range(gate.COUNT)]
        records = [{"id": "monitor-" + str(i), "path": "/v1/chat/completions", "stream": False,
                    "state": "completed", "http_status": 200, "input": json.dumps(body),
                    "response": json.dumps(response), "input_truncated": False, "output_truncated": False,
                    "reasoning_truncated": False, "response_truncated": False,
                    "engine_route": {"worker": {**identity["worker"], "generation": "spawn"},
                                     "calls": 1, "tokens": 8, "stable": True}}
                   for i, (body, response) in enumerate(zip(bodies, responses))]
        return {"schema": 2, "kind": "served-greedy16", "requests": bodies, "requests_sha256": gate.digest(bodies),
                "model_config_sha256": "a" * 64, "identity": identity,
                "responses": responses, "results": [gate.normalize(r) for r in responses],
                "monitor_records": records, "monitor_boundary": [],
                "route_joins": join_records(bodies, responses, records, [], identity["worker"])}

    def test_paired_gate_checks_all_sixteen_and_only_declared_delta(self):
        left, right = self.receipt(), self.receipt("1")
        self.assertTrue(gate.compare(left, right, ["STRATA_BATCH_DENSE_XMX"]))
        for i in range(gate.COUNT):
            candidate = copy.deepcopy(right)
            candidate["responses"][i]["choices"][0]["message"]["content"] += "different"
            candidate["results"][i] = gate.normalize(candidate["responses"][i])
            candidate["monitor_records"][i]["response"] = json.dumps(candidate["responses"][i])
            candidate["route_joins"] = join_records(candidate["requests"], candidate["responses"],
                                                    candidate["monitor_records"], [], candidate["identity"]["worker"])
            with self.subTest(i=i), self.assertRaisesRegex(ValueError, "output mismatch"):
                gate.compare(left, candidate, ["STRATA_BATCH_DENSE_XMX"])
        with self.assertRaisesRegex(ValueError, "variant keys"):
            gate.compare(left, left, ["STRATA_BATCH_DENSE_XMX"])

    def test_no_legacy_identity_only_or_tampered_join_can_pass(self):
        for key, value in (("schema", 1), ("kind", "identity-only"), ("route_joins", []), ("responses", []),
                           ("monitor_records", []), ("monitor_boundary", ["stale"])):
            receipt = self.receipt()
            receipt[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.validate(receipt)

    def test_reasoning_fields_incomplete_output_and_usage_refused(self):
        response = self.receipt()["responses"][0]
        for field in ("reasoning", "reasoning_content"):
            broken = copy.deepcopy(response)
            broken["choices"][0]["message"][field] = ["invalid"]
            with self.assertRaisesRegex(ValueError, "reasoning"):
                gate.normalize(broken)
        for tokens in (0, True, None, -1):
            broken = copy.deepcopy(response)
            broken["usage"]["completion_tokens"] = tokens
            with self.assertRaises(ValueError):
                gate.normalize(broken)
        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            gate.normalize(response)

    def test_proxy_disabled_and_redirect_refused_without_network(self):
        opener = mock.Mock()
        with mock.patch.object(gate.urllib.request, "build_opener", return_value=opener) as build:
            request = gate.urllib.request.Request("http://127.0.0.1:8090")
            gate.loopback_open(request, 1)
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertIsNone(build.call_args.args[1].redirect_request(
            request, None, 302, "redirect", {}, "http://elsewhere.invalid"))

    def capture(self, root, identity=None, mutate=None, old_rows=False, collector=None):
        fixture = self.receipt()
        seen, posted = [], []
        args = SimpleNamespace(timeout=5, model_config_sha256="a" * 64, out=str(Path(root) / "receipt.json"),
                               model="m", url="http://127.0.0.1:8090")
        if collector is not None:
            args._collector = collector
        by_body = {gate.digest(b): (r, m) for b, r, m in zip(
            fixture["requests"], fixture["responses"], fixture["monitor_records"])}

        def open_request(request, timeout):
            seen.append(request.full_url)
            if request.data is not None:
                body = json.loads(request.data)
                response, record = by_body[gate.digest(body)]
                posted.append(record)
                return Reply(json.dumps(response).encode())
            if "?id=" in request.full_url:
                record = next(r for r in posted if r["id"] == request.full_url.split("?id=")[1])
                if mutate:
                    record = mutate(copy.deepcopy(record))
                return Reply(json.dumps(record).encode())
            summaries = [{"id": r["id"]} for r in posted] + ([{"id": "old-row"}] if old_rows else [])
            return Reply(json.dumps({"requests": summaries}).encode())

        with mock.patch.object(gate, "identity", side_effect=identity or [fixture["identity"], fixture["identity"]]), \
                mock.patch.object(gate, "loopback_open", side_effect=open_request), \
                mock.patch.dict(os.environ, {"API_KEY": "fakekey"}), contextlib.redirect_stdout(io.StringIO()):
            gate.capture(args)
        return args, seen

    def test_capture_joins_same_sixteen_posts_and_reserves_fresh_path(self):
        with tempfile.TemporaryDirectory() as root:
            args, seen = self.capture(root)
            receipt = json.loads(Path(args.out).read_text())
            gate.validate(receipt)
            self.assertEqual(sum(url.endswith("/v1/chat/completions") for url in seen), gate.COUNT)
            self.assertEqual(len(receipt["route_joins"]), gate.COUNT)
            self.assertNotIn("fakekey", json.dumps(receipt))
            with mock.patch.object(gate, "identity") as identity, self.assertRaises(FileExistsError):
                gate.capture(args)
            identity.assert_not_called()

    def test_failed_route_or_changed_identity_leaves_no_success_json(self):
        for name in ("route", "identity", "evicted"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as root:
                mutate = (lambda r: {**r, "engine_route": {}}) if name == "route" else None
                identity = [self.identity(), {**self.identity(), "started": "changed"}] if name == "identity" else None
                if name == "evicted":
                    def missing_record(r):
                        return {"error": "record not found"}
                    mutate = missing_record
                with self.assertRaises((ValueError, KeyError)):
                    self.capture(root, identity, mutate)
                self.assertEqual((Path(root) / "receipt.json").read_text(), "")

    def test_collector_closes_before_success_and_old_prompt_details_are_not_fetched(self):
        with tempfile.TemporaryDirectory() as root:
            collector = mock.Mock()
            collector.close.side_effect = lambda: self.assertEqual((Path(root) / "receipt.json").read_text(), "")
            args, seen = self.capture(root, old_rows=True, collector=collector)
            collector.close.assert_called_once()
            self.assertFalse(hasattr(args, "_collector"))
            self.assertFalse(any("?id=old-row" in url for url in seen))
            receipt = json.loads(Path(args.out).read_text())
            self.assertEqual(receipt["monitor_records"][-1], {"id": "old-row"})
        with tempfile.TemporaryDirectory() as root:
            collector = mock.Mock()
            collector.close.side_effect = TimeoutError("fixture collector did not close")
            with self.assertRaises(TimeoutError):
                self.capture(root, collector=collector)
            self.assertEqual((Path(root) / "receipt.json").read_text(), "")

    def test_identity_integrates_persistent_collector_with_immutable_container(self):
        from serve.test_process_evidence import DockerCustody
        from serve.process_evidence import env_digest
        container, image, contract = DockerCustody().fixture()
        args = SimpleNamespace(image=image["Id"], source="c" * 40, engine_sha256="d" * 64,
                               engine=contract["engine"], container="mutable-name", url="http://127.0.0.1:8090",
                               collector_timeout=5, collector_cleanup_timeout=7, collector_lifetime=120)
        contract["engine_sha256"] = args.engine_sha256
        container["Config"]["Env"].append("STRATA_BATCH_DENSE_XMX=0")
        contract["docker_env_sha256"] = env_digest(container["Config"]["Env"])
        image["Config"] = {"Labels": {"homelab.engine": args.source, "homelab.strata_sha256": args.engine_sha256}}
        with tempfile.TemporaryDirectory() as root:
            args.settings = str(Path(root) / "settings.json")
            Path(args.settings).write_text(json.dumps({"STRATA_BATCH_DENSE_XMX": "0"}))
            args.runtime_contract = str(Path(root) / "runtime.json")
            Path(args.runtime_contract).write_text(json.dumps(contract))
            args.runtime_contract_sha256 = gate.digest(contract)
            process = self.identity()["process"]
            collector = mock.Mock(initial=process)
            collector.snapshot.return_value = process
            with mock.patch.object(gate, "docker_json", side_effect=[container, image, container, image]), \
                    mock.patch.object(gate, "Collector", return_value=collector) as create, \
                    mock.patch.object(gate, "loopback_open") as http:
                before = gate.identity(args)
                self.assertEqual(gate.identity(args), before)
            self.assertEqual(create.call_args.args[0], container["Id"])
            self.assertEqual(create.call_args.kwargs, {"timeout": 5, "cleanup_timeout": 7, "lifetime": 120})
            collector.snapshot.assert_called_once()
            http.assert_not_called()
            self.assertNotIn("fakekey", json.dumps(before))
            collector.close()
            del args._collector
            args.runtime_contract_sha256 = "0" * 64
            with mock.patch.object(gate, "docker_json") as docker, \
                    self.assertRaisesRegex(ValueError, "runtime contract"):
                gate.identity(args)
            docker.assert_not_called()
            args.runtime_contract_sha256 = gate.digest(contract)
            container["NetworkSettings"]["Ports"] = {"80/udp": [{"HostPort": "8090", "HostIp": "127.0.0.1"}]}
            with mock.patch.object(gate, "docker_json", side_effect=[container, image]), \
                    mock.patch.object(gate, "Collector") as create, self.assertRaisesRegex(ValueError, "not published"):
                gate.identity(args)
            create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
