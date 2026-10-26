# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fail-closed joins for the existing request monitor; no HTTP or process operations.

The caller supplies independently pinned worker/pipe identity. This module does not
attest executable bytes, frozen imports/configuration, or actual engine batch rows.
"""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def join_records(bodies, responses, records, boundary, worker):
    """Join the same POSTs, not another inference pass, to new complete monitor rows.

    boundary is the pre-window monitor ID list, newest first. Its newest retained
    row is the eviction anchor; a lost anchor or unrelated new request refuses.
    worker contains namespace PIDs and pipe device/inode pairs from the collector.
    The per-spawn wrapper generation must additionally be identical across rows.
    """
    fields = ("wrapper_pid", "worker_pid", "stdin_pipe", "stdout_pipe")
    require(isinstance(worker, dict) and all(k in worker for k in fields), "missing pinned worker")
    require(all(type(worker[k]) is int and worker[k] > 0 for k in fields[:2]), "invalid pinned PID")
    require(all(isinstance(worker[k], list) and len(worker[k]) == 2 and
                all(type(x) is int and x >= 0 for x in worker[k]) and worker[k][1] > 0
                for k in fields[2:]), "invalid pinned pipe")
    require(worker["stdin_pipe"] != worker["stdout_pipe"], "ambiguous pinned pipes")
    require(isinstance(boundary, list) and all(isinstance(x, str) and x for x in boundary) and
            len(set(boundary)) == len(boundary), "invalid pre-window boundary")
    require(isinstance(records, list) and all(isinstance(r, dict) for r in records), "missing monitor records")
    ids = [r.get("id") for r in records]
    require(all(isinstance(x, str) and x for x in ids) and len(set(ids)) == len(ids),
            "duplicate or missing monitor ID")
    require(not boundary or boundary[0] in ids, "monitor boundary evicted")
    new = [r for r in records if r["id"] not in boundary]
    require(bool(bodies) and len(bodies) == len(responses) == len(new), "missing or ambiguous new monitor rows")
    require(len({digest(b) for b in bodies}) == len(bodies), "ambiguous repeated requests")
    response_ids = [r.get("id") for r in responses]
    require(all(isinstance(x, str) and x for x in response_ids) and
            len(set(response_ids)) == len(response_ids), "duplicate or missing response ID")
    joined, generations, used = [], set(), set()
    for body, response in zip(bodies, responses):
        matches = [r for r in new if r.get("response") and json.loads(r["response"]).get("id") == response["id"]]
        require(len(matches) == 1, "missing or duplicate response join")
        record = matches[0]
        require(record["id"] not in used, "reused monitor row")
        used.add(record["id"])
        require(record.get("state") == "completed" and record.get("http_status") == 200 and
                record.get("path") == "/v1/chat/completions" and record.get("stream") is False,
                "monitor request did not complete normally")
        require(all(record.get(k + "_truncated") is False for k in ("input", "output", "reasoning", "response")),
                "missing truncation evidence or truncated monitor record")
        require(json.loads(record["input"]) == body, "monitor input mismatch")
        require(digest(json.loads(record["response"])) == digest(response), "monitor response mismatch")
        route = record.get("engine_route") or {}
        consumed = all(type(route.get(k)) is int and route[k] > 0 for k in ("calls", "tokens"))
        require(route.get("stable") is True and consumed, "missing positive stable worker consumption")
        chosen = route.get("worker") or {}
        require(all(chosen.get(k) == worker[k] for k in fields), "chosen worker or pipe mismatch")
        generation = chosen.get("generation")
        require(isinstance(generation, str) and bool(generation), "missing worker generation")
        generations.add(generation)
        joined.append({"monitor_id": record["id"], "response_id": response["id"],
                       "request_sha256": digest(body), "response_sha256": digest(response),
                       "engine_route": route})
    require(len(generations) == 1, "worker generation changed during capture")
    return joined
