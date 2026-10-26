#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prepared operation input, not an authorization or a submitted queue job.

Extends the existing one-prompt served parity contract: temperature 0, seed 1,
exact content/reasoning/token counts. Sixteen requests start together. Capture
checks the running container, immutable image and actual engine, not its tag.
The accepted operation packet must separately bind the loaded model and serving
configuration. Queue success never substitutes for this script's own receipt.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import urllib.parse
import urllib.request

from serve.route_evidence import join_records
from serve.process_evidence import (CAPTURE_LIFETIME_DEFAULT, CLEANUP_TIMEOUT_DEFAULT, Collector,
                                    READ_TIMEOUT_DEFAULT, SECRET_ENV_KEYS_DEFAULT, container_identity)

PROMPTS = (
    "List the first 12 prime numbers, comma-separated, then give their sum.",
    "Calculate 37 times 19. Give the result and a short check.",
    "Sort these integers ascending: 91, -4, 17, 0, 17, 6. Return a JSON array.",
    "Return valid JSON with the keys red, green and blue and values 1, 2 and 3.",
    "Write a Python function that returns the greatest common divisor of two positive integers.",
    "Explain in two sentences why a database transaction needs atomicity.",
    "Translate into French: The small cat is sleeping beside the window.",
    "Give the next five terms after 2, 3, 5, 8, 13 and explain the rule.",
    "A train travels 180 kilometres in 2 hours 15 minutes. Find its average speed in kilometres per hour.",
    "Write a SQL query counting orders per customer in a table named orders with a customer_id column.",
    "Name the three states of water at ordinary pressure and give one example of each.",
    "Summarize in one sentence: A village planted trees upstream. The roots held soil in place. "
    "Less sediment reached its reservoir.",
    "Find the error and correct it: def square(x): return x + x",
    "Give a four-line poem about rain, with no title.",
    "If all sparrows are birds and no birds are stones, can a sparrow be a stone? Explain briefly.",
    "Convert the hexadecimal number 2AF to decimal and show the arithmetic.",
)
COUNT = 16  # This gate's specified workload, not a scheduler or engine limit.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def hex_value(value, length):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{%d}" % length, value) is not None


def requests(model):
    require(len(PROMPTS) == COUNT and len(set(PROMPTS)) == COUNT, "need sixteen distinct prompts")
    return [{"model": model, "temperature": 0, "seed": 1, "stream": False,
             "messages": [{"role": "user", "content": prompt}]} for prompt in PROMPTS]


def normalize(response):
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "need exactly one choice")
    choice = choices[0]
    require(choice.get("finish_reason") == "stop", "response incomplete or did not stop normally")
    message = choice.get("message", {})
    content = message.get("content")
    require(all(message.get(k) is None or isinstance(message.get(k), str)
                for k in ("reasoning_content", "reasoning")), "invalid reasoning")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    require(content is None or isinstance(content, str), "invalid content")
    require(bool((content or "").strip() or (reasoning or "").strip()), "empty output")
    tokens = (response.get("usage") or {}).get("completion_tokens")
    require(type(tokens) is int and tokens > 0, "missing positive completion token count")
    return {"content": content, "reasoning": reasoning, "completion_tokens": tokens}


def docker_json(*args):
    return json.loads(subprocess.check_output(["docker", *args], stderr=subprocess.DEVNULL))[0]


def identity(a):
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", a.image) is not None, "need immutable image ID")
    require(hex_value(a.source, 40) and hex_value(a.engine_sha256, 64), "invalid expected source/engine identity")
    expected_env = json.loads(Path(a.settings).read_text())
    require(isinstance(expected_env, dict) and expected_env, "need explicit expected STRATA settings")
    require(all(isinstance(k, str) and k.startswith("STRATA_") and k not in SECRET_ENV_KEYS_DEFAULT and
                isinstance(v, str) for k, v in expected_env.items()), "settings must contain nonsecret STRATA strings")
    contract = json.loads(Path(a.runtime_contract).read_text())
    require(hex_value(a.runtime_contract_sha256, 64) and digest(contract) == a.runtime_contract_sha256,
            "unaccepted runtime contract")
    require(contract["image"] == a.image and contract["engine_sha256"] == a.engine_sha256 and
            contract["engine"] == a.engine,
            "runtime contract engine/image mismatch")
    container = docker_json("inspect", a.container)
    require(container["State"].get("Running") is True and not container["State"].get("Restarting"),
            "container not steadily running")
    require(container["Image"] == a.image, "running container image mismatch")
    image = docker_json("image", "inspect", a.image)
    labels = image["Config"].get("Labels") or {}
    require(image["Id"] == a.image and labels.get("homelab.engine") == a.source and
            labels.get("homelab.strata_sha256") == a.engine_sha256, "immutable image label mismatch")
    actual_env = dict(e.split("=", 1) for e in container["Config"].get("Env", [])
                      if e.startswith("STRATA_") and "=" in e and e.split("=", 1)[0] not in SECRET_ENV_KEYS_DEFAULT)
    require(actual_env == expected_env, "actual STRATA settings differ from accepted settings")
    url = urllib.parse.urlsplit(a.url)
    require(url.scheme == "http" and url.hostname == "127.0.0.1" and not url.username and
            not url.password and url.path in ("", "/") and not url.query and not url.fragment,
            "URL must be a loopback HTTP origin")
    ports = container["NetworkSettings"].get("Ports") or {}
    require(any(binding.get("HostPort") == str(url.port or 80) and
                binding.get("HostIp") in ("127.0.0.1", "0.0.0.0", "::")
                for port, bindings in ports.items() if port.endswith("/tcp")
                for binding in (bindings or [])),
            "URL port not published by verified container")
    target = container_identity(container, image, contract, a.url)
    if getattr(a, "_collector", None) is None:
        a._collector = Collector(container["Id"], contract, target, timeout=a.collector_timeout,
                                 cleanup_timeout=a.collector_cleanup_timeout, lifetime=a.collector_lifetime)
        a._target = target
        process = a._collector.initial
    else:
        require(target == a._target, "container/launch/custody changed during capture")
        process = a._collector.snapshot()
    return {**target, "source": a.source, "engine_sha256": process["engine"]["sha256"],
            "settings": actual_env, "worker": process["worker"], "process": process}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def loopback_open(request, timeout):
    # Credentials and responses must stay on the already verified origin.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(
        request, timeout=timeout)


def capture(a):
    require(math.isfinite(a.timeout) and a.timeout > 0 and hex_value(a.model_config_sha256, 64),
            "invalid timeout/model-config binding")
    # Reserve a new path before live work, so failure cannot leave an old success receipt.
    with Path(a.out).open("x") as out:
        try:
            receipt = collect(a)
        finally:
            collector = getattr(a, "_collector", None)
            if collector is not None:
                collector.close()
                del a._collector
        # No success JSON exists unless the collector also closed cleanly.
        validate(receipt)
        json.dump(receipt, out, sort_keys=True)
        out.write("\n")
    print("KEY served-greedy16 capture rc=0 (comparison still required)")


def collect(a):
    before = identity(a)
    require(isinstance(before.get("worker"), dict), "missing pinned worker identity")
    bodies = requests(a.model)
    barrier = threading.Barrier(COUNT)
    key = os.environ.get("API_KEY")
    require(bool(key), "API_KEY is required in the environment")

    def monitor(path=""):
        request = urllib.request.Request(a.url.rstrip("/") + "/api/requests" + path,
                                         headers={"Authorization": "Bearer " + key})
        with loopback_open(request, timeout=a.timeout) as response:
            return json.load(response)

    boundary = [r["id"] for r in monitor()["requests"]]

    def one(body):
        request = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions",
                                         json.dumps(body).encode(),
                                         {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        barrier.wait(timeout=a.timeout)
        with loopback_open(request, timeout=a.timeout) as response:
            require(response.status == 200, "completion HTTP status is not 200")
            return json.load(response)

    with concurrent.futures.ThreadPoolExecutor(max_workers=COUNT) as pool:
        responses = list(pool.map(one, bodies))
    summaries = monitor()["requests"]
    records = [monitor("?id=" + urllib.parse.quote(r["id"], safe="")) for r in summaries if r["id"] not in boundary]
    if boundary and any(r["id"] == boundary[0] for r in summaries):
        records.append({"id": boundary[0]})  # eviction anchor only; do not fetch unrelated old prompts/responses
    joined = join_records(bodies, responses, records, boundary, before["worker"])
    require(identity(a) == before, "container/engine/settings changed during capture")
    return {"schema": 2, "kind": "served-greedy16", "requests": bodies, "requests_sha256": digest(bodies),
            "model_config_sha256": a.model_config_sha256, "identity": before,
            "results": [normalize(response) for response in responses], "responses": responses,
            "monitor_records": records, "monitor_boundary": boundary, "route_joins": joined}


def validate(receipt):
    bodies, results = receipt["requests"], receipt["results"]
    require(receipt["schema"] == 2 and receipt.get("kind") == "served-greedy16" and
            isinstance(bodies, list) and len(bodies) == COUNT and
            isinstance(results, list) and len(results) == COUNT, "incomplete sixteen-prompt receipt")
    require(bodies == requests(bodies[0]["model"]) and receipt["requests_sha256"] == digest(bodies),
            "request identity differs from the fixed greedy workload")
    require(hex_value(receipt["model_config_sha256"], 64), "missing model/config evidence binding")
    for result in results:
        require(normalize({"choices": [{"finish_reason": "stop", "message": result}],
                           "usage": {"completion_tokens": result["completion_tokens"]}}) == result,
                "invalid normalized result")
    require([normalize(response) for response in receipt["responses"]] == results,
            "responses differ from normalized results")
    ident = receipt["identity"]
    process = ident["process"]
    require(hex_value(ident["contract_sha256"], 64) and hex_value(process["closure_sha256"], 64) and
            hex_value(process["interpreter"]["sha256"], 64) and process["worker"] == ident["worker"] and
            process["engine"]["sha256"] == ident["engine_sha256"] and
            process["listener"]["pid"] == ident["worker"]["wrapper_pid"] and
            process["child"]["pid"] == ident["worker"]["worker_pid"] and
            all(type(process[k]["start"]) is int and process[k]["start"] > 0 for k in ("listener", "child")),
            "invalid process-bound identity")
    require(join_records(bodies, receipt["responses"], receipt["monitor_records"],
                         receipt["monitor_boundary"], ident["worker"]) == receipt["route_joins"],
            "route joins differ from retained evidence")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", ident["image"]) is not None and
            hex_value(ident["source"], 40) and hex_value(ident["engine_sha256"], 64) and
            hex_value(ident["launch_sha256"], 64) and bool(ident["settings"]) and
            bool(ident["container"]) and bool(ident["started"]), "invalid identity receipt")


def compare(ref, candidate, variant_keys):
    validate(ref)
    validate(candidate)
    for key in ("requests", "requests_sha256", "model_config_sha256"):
        require(ref[key] == candidate[key], "control/candidate workload or model/config binding differs")
    # This paired gate isolates runtime switches in the same composed binary.
    for key in ("image", "source", "engine_sha256", "launch_sha256"):
        require(ref["identity"][key] == candidate["identity"][key], "paired binary/launch identity differs")
    require(ref["identity"]["process"]["closure_sha256"] == candidate["identity"]["process"]["closure_sha256"] and
            ref["identity"]["process"]["interpreter"]["sha256"] ==
            candidate["identity"]["process"]["interpreter"]["sha256"], "paired frozen-wrapper identity differs")
    left, right = ref["identity"]["settings"], candidate["identity"]["settings"]
    changed = {k for k in left.keys() | right.keys() if left.get(k) != right.get(k)}
    require(changed == set(variant_keys) and bool(changed), "settings delta must equal declared variant keys")
    mismatches = [i for i in range(COUNT) if ref["results"][i] != candidate["results"][i]]
    require(not mismatches, "served greedy output mismatch at prompt indexes " + repr(mismatches))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture")
    for name in ("container", "image", "source", "engine", "engine-sha256", "settings",
                 "model-config-sha256", "runtime-contract", "runtime-contract-sha256", "out"):
        cap.add_argument("--" + name, required=True)
    cap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8090"))
    cap.add_argument("--model", default="m")
    cap.add_argument("--timeout", type=float, default=900)
    cap.add_argument("--collector-timeout", type=float, default=READ_TIMEOUT_DEFAULT,
                     help="deadline for each collector snapshot, including pipe writes")
    cap.add_argument("--collector-cleanup-timeout", type=float, default=CLEANUP_TIMEOUT_DEFAULT)
    cap.add_argument("--collector-lifetime", type=float, default=CAPTURE_LIFETIME_DEFAULT,
                     help="helper whole-capture lifetime; budget for all POSTs and monitor reads, not just snapshots")
    cmp = sub.add_parser("compare")
    cmp.add_argument("ref")
    cmp.add_argument("candidate")
    cmp.add_argument("--variant-keys", required=True, help="comma-separated exact changed STRATA settings")
    a = parser.parse_args()
    if a.mode == "capture":
        capture(a)
    else:
        compare(json.loads(Path(a.ref).read_text()), json.loads(Path(a.candidate).read_text()),
                a.variant_keys.split(","))
        print("KEY served-greedy16 parity rc=0 prompts=16 exact-content-reasoning-completion_tokens")


if __name__ == "__main__":
    main()
