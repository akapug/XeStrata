# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Synthetic proc/Docker custody controls; never reads live proc or opens a socket."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from serve.process_evidence import (
    Collector, EvidenceError, ProcessPins, ProcView, container_identity, digest, env_digest)
import serve.process_evidence as evidence


def sha(value):
    return hashlib.sha256(value).hexdigest()


class Processes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proc = self.root / "proc"
        (self.proc / "net").mkdir(parents=True)
        (self.proc / "net/tcp").write_text("header\n0: 00000000:1F9A 00000000:0000 0A 0:0 0:0 0 0 0 200\n")
        (self.proc / "net/tcp6").write_text("header\n")
        self.interpreter, self.engine = self.root / "python", self.root / "strata"
        self.interpreter.write_bytes(b"accepted interpreter")
        self.engine.write_bytes(b"accepted engine")
        self.entry, self.config = self.root / "server.py", self.root / "config.json"
        self.entry.write_bytes(b"reviewed wrapper import closure")
        self.config.write_text(json.dumps({"api_monitor": True}))
        files = {str(p): sha(p.read_bytes()) for p in (self.entry, self.config)}
        self.contract = {"image": "sha256:" + "a" * 64, "files": files, "closure_sha256": digest(files),
                         "entrypoint_file": str(self.entry), "config_file": str(self.config),
                         "interpreter": str(self.interpreter), "interpreter_sha256": sha(self.interpreter.read_bytes()),
                         "engine": str(self.engine), "engine_sha256": sha(self.engine.read_bytes()),
                         "wrapper_argv": [str(self.interpreter), str(self.entry), "--engine", "strata", "--config",
                                          str(self.config), "--api-monitor"],
                         "worker_argv": [str(self.engine), "--serve"], "wrapper_cwd": str(self.root),
                         "worker_cwd": str(self.root), "wrapper_env_sha256": env_digest(["PATH=/bin"]),
                         "worker_env_sha256": env_digest(["PATH=/bin"]), "import_roots": [str(self.root)]}
        self.target = {"target_port": 8090, "target_address": "192.0.2.2"}
        self.make_process(1, 0, 100, ["init"], {})
        self.make_process(10, 1, 101, self.contract["wrapper_argv"],
                          {3: ("socket:[200]", 2), 4: ("pipe:[100]", 1), 5: ("pipe:[101]", 0)})
        self.make_process(11, 10, 102, self.contract["worker_argv"], {0: ("pipe:[100]", 0), 1: ("pipe:[101]", 1)})
        (self.proc / "10/exe").symlink_to(self.interpreter)
        (self.proc / "11/exe").symlink_to(self.engine)
        self.view = ProcView(self.proc)
        self.pins = ProcessPins(self.contract, self.target, self.view)
        self.addCleanup(self.pins.close)
        real_open = os.open
        patcher = mock.patch("os.pidfd_open", side_effect=lambda pid: real_open(os.devnull, os.O_RDONLY), create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch("serve.process_evidence.select.select", return_value=([], [], []))
        self.select = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(self.view, "pipe_identity", side_effect=self.pipe_identity)
        patcher.start()
        self.addCleanup(patcher.stop)

    def pipe_identity(self, pid, fd):
        link = os.readlink(self.proc / str(pid) / "fd" / str(fd))
        return [77, int(link.removeprefix("pipe:[").removesuffix("]"))]

    def make_process(self, pid, parent, start, argv, fds):
        p = self.proc / str(pid)
        (p / "fd").mkdir(parents=True)
        (p / "fdinfo").mkdir()
        (p / "ns").mkdir()
        for ns, link in (("net", "net:[1]"), ("pid", "pid:[2]")):
            (p / "ns" / ns).symlink_to(link)
        self.set_stat(pid, parent, start)
        (p / "cwd").symlink_to(self.root)
        (p / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
        (p / "environ").write_bytes(b"PATH=/bin\0STRATA_API_KEY=fixture-credential\0")
        for fd, (target, mode) in fds.items():
            self.set_fd(pid, fd, target, mode)

    def set_stat(self, pid, parent, start):
        fields = ["S", str(parent)] + ["0"] * 17 + [str(start)]
        (self.proc / str(pid) / "stat").write_text(f"{pid} (test process) " + " ".join(fields))

    def set_fd(self, pid, fd, target, mode):
        p = self.proc / str(pid)
        link = p / "fd" / str(fd)
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        (p / "fdinfo" / str(fd)).write_text(f"flags:\t{mode:08o}\n")

    def test_pins_stay_open_and_same_identity_rechecks(self):
        before = self.pins.snapshot()
        self.assertEqual(self.pins.snapshot(), before)
        self.assertEqual(len(self.pins.pins), 2)
        self.assertEqual(before["worker"], {"wrapper_pid": 10, "worker_pid": 11,
                                            "stdin_pipe": [77, 100], "stdout_pipe": [77, 101]})
        self.assertNotIn("fixture-credential", json.dumps(before))
        self.pins.close()
        self.assertEqual(self.pins.pins, [])

    def test_self_fd_census_keeps_enumeration_descriptor_alive(self):
        # Model proc's own directory descriptor without reading any live proc files. Path.iterdir()
        # closes its scan before exposing entries; the enumerator's own FD then no longer exists.
        self.set_fd(10, 99, "anon_inode:fixture-scandir", os.O_RDONLY)
        directory = self.proc / "10/fd"
        entries = [SimpleNamespace(path=str(p), name=p.name) for p in directory.iterdir()]
        active = [False]
        real_readlink = os.readlink

        @contextlib.contextmanager
        def scan(path):
            self.assertEqual(Path(path), directory)
            active[0] = True

            def iterator():
                try:
                    yield from entries
                finally:
                    active[0] = False
            try:
                yield iterator()
            finally:
                active[0] = False

        def materialized(path):
            with scan(path) as iterator:
                return iter([Path(entry.path) for entry in iterator])

        def readlink(path):
            if Path(path).name == "99" and not active[0]:
                raise FileNotFoundError("enumeration descriptor already closed")
            return real_readlink(path)

        with mock.patch.object(Path, "iterdir", autospec=True, side_effect=materialized), \
                mock.patch.object(evidence.os, "scandir", side_effect=scan), \
                mock.patch.object(evidence.os, "readlink", side_effect=readlink):
            self.assertEqual(self.view.fds(10)[99], ("anon_inode:fixture-scandir", os.O_RDONLY))
        self.assertFalse(active[0])

    def test_unused_matching_rootfs_file_does_not_attest_running_old_executable(self):
        old = self.root / "old"
        old.write_bytes(b"unaccepted old engine")
        (self.proc / "11/exe").unlink()
        (self.proc / "11/exe").symlink_to(old)
        self.assertEqual(sha(self.engine.read_bytes()), self.contract["engine_sha256"])
        with self.assertRaisesRegex(ValueError, "executable mismatch"):
            self.pins.snapshot()

    def test_equal_bytes_replaced_object_and_generation_changes_refuse(self):
        self.pins.snapshot()
        self.engine.rename(self.root / "old-engine")
        self.engine.write_bytes(b"accepted engine")
        with self.assertRaisesRegex(ValueError, "executable mismatch"):
            self.pins.snapshot()

    def test_pid_start_and_pidfd_death_refuse(self):
        self.pins.snapshot()
        self.set_stat(11, 10, 999)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.pins.snapshot()
        self.select.return_value = ([0], [], [])
        with self.assertRaisesRegex(ValueError, "process exited"):
            self.pins.snapshot()

    def test_extra_pipe_writer_or_reader_refuses(self):
        self.make_process(12, 1, 103, ["unrelated"], {6: ("pipe:[100]", 1)})
        with self.assertRaisesRegex(ValueError, "extra or ambiguous"):
            self.pins.snapshot()
        self.set_fd(12, 6, "pipe:[101]", 0)
        with self.assertRaisesRegex(ValueError, "extra or ambiguous"):
            self.pins.snapshot()

    def test_wrong_direction_or_reciprocal_endpoint_refuses(self):
        self.set_fd(11, 0, "pipe:[100]", 1)
        with self.assertRaisesRegex(ValueError, "directions"):
            self.pins.snapshot()
        self.set_fd(11, 0, "pipe:[100]", 0)
        self.set_fd(10, 4, "pipe:[999]", 1)
        with self.assertRaisesRegex(ValueError, "reciprocal"):
            self.pins.snapshot()

    def test_ambiguous_listener_and_ipv6_reachability_refuse(self):
        self.make_process(12, 1, 103, ["unrelated"], {7: ("socket:[200]", 2)})
        with self.assertRaisesRegex(ValueError, "ownership ambiguous"):
            self.pins.snapshot()
        (self.proc / "12/fd/7").unlink()
        (self.proc / "net/tcp6").write_text("header\n0: " + "0" * 32 + ":1F9A 0:0 0A 0:0 0:0 0 0 0 201\n")
        with self.assertRaisesRegex(ValueError, "IPv6"):
            self.pins.snapshot()

    def test_worker_namespace_or_effective_environment_changes_refuse(self):
        p = self.proc / "11/ns/net"
        p.unlink()
        p.symlink_to("net:[wrong]")
        with self.assertRaisesRegex(ValueError, "namespace mismatch"):
            self.pins.snapshot()
        p.unlink()
        p.symlink_to("net:[1]")
        (self.proc / "11/environ").write_bytes(b"PATH=/bin\0PYTHONPATH=/injection\0")
        with self.assertRaisesRegex(ValueError, "interpreter or loader injection"):
            self.pins.snapshot()

    def test_matching_environment_digest_cannot_authorize_injected_import_or_loader(self):
        for key in evidence.INJECTION_ENV_KEYS_DEFAULT:
            values = {"PATH": "/bin", key: "/injection"}
            self.contract["worker_env_sha256"] = digest(values)
            (self.proc / "11/environ").write_bytes(
                b"\0".join((k + "=" + value).encode() for k, value in values.items()) + b"\0")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "interpreter or loader injection"):
                self.pins.snapshot()
        self.assertEqual(env_digest(["PATH=/bin", "PYTHONPATH="]), digest({"PATH": "/bin", "PYTHONPATH": ""}))

    def test_changed_source_or_alternate_configuration_refuses(self):
        self.entry.write_bytes(b"changed wrapper")
        with self.assertRaisesRegex(ValueError, "closure|source/import/config"):
            self.pins.snapshot()
        self.entry.write_bytes(b"reviewed wrapper import closure")
        for cfg in ({"api_monitor": True, "replicas": 2}, {"api_monitor": True, "before_load": "unaccepted hook"},
                    {"api_monitor": True, "api_key": "fakekey"}):
            self.config.write_text(json.dumps(cfg))
            self.contract["files"][str(self.config)] = sha(self.config.read_bytes())
            with self.subTest(cfg=list(cfg)), self.assertRaises(ValueError):
                self.pins.snapshot()

    def test_mock_or_another_live_worker_does_not_match_selected_child(self):
        self.contract["wrapper_argv"][self.contract["wrapper_argv"].index("strata")] = "mock"
        (self.proc / "10/cmdline").write_bytes("\0".join(self.contract["wrapper_argv"]).encode())
        with self.assertRaisesRegex(ValueError, "single StrataEngine"):
            self.pins.snapshot()


class DockerCustody(unittest.TestCase):
    def fixture(self):
        image = {"Id": "sha256:" + "a" * 64}
        config = {"Entrypoint": ["python", "-m", "serve.server"], "Cmd": [], "WorkingDir": "/app", "User": ""}
        files = {"/app/serve/server.py": "a" * 64, "/app/config.json": "b" * 64}
        contract = {"image": image["Id"], "files": files, "closure_sha256": digest(files),
                    "entrypoint_file": "/app/serve/server.py", "config_file": "/app/config.json",
                    "import_roots": ["/app/serve", "/app/tools"], "interpreter": "/usr/bin/python3",
                    "engine": "/engine/strata", "launch_sha256": digest(config),
                    "docker_env_sha256": env_digest(["PATH=/bin"]), "mounts_sha256": digest([]),
                    "tmpfs_sha256": digest({})}
        container = {"Id": "f" * 64, "Image": image["Id"], "State": {"Running": True, "Pid": 123, "StartedAt": "start"},
                     "HostConfig": {"ReadonlyRootfs": True, "NetworkMode": "bridge"}, "Config": {**config, "Env":
                     ["PATH=/bin", "STRATA_API_KEY=fakekey"]}, "Mounts": [], "NetworkSettings": {"Ports":
                     {"80/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8090"}]},
                     "Networks": {"bridge": {"IPAddress": "192.0.2.2"}}}}
        return container, image, contract

    def test_frozen_image_contract_and_exact_tcp_origin(self):
        container, image, contract = self.fixture()
        result = container_identity(container, image, contract, "http://127.0.0.1:8090")
        self.assertEqual(result["target_port"], 80)
        self.assertNotIn("fakekey", json.dumps(result))
        self.assertEqual(env_digest(["PATH=/bin", "STRATA_API_KEY=new-secret"]), contract["docker_env_sha256"])

    def test_unaccepted_launch_environment_rootfs_namespaces_or_mount_refuse(self):
        original, image, contract = self.fixture()
        changes = [("HostConfig", {"ReadonlyRootfs": False}),
                   ("HostConfig", {"ReadonlyRootfs": True, "PidMode": "host"}),
                   ("HostConfig", {"ReadonlyRootfs": True, "NetworkMode": "host"}),
                   ("HostConfig", {"ReadonlyRootfs": True, "CapAdd": ["SYS_PTRACE"]}),
                   ("Config", {**original["Config"], "Cmd": ["changed"]}),
                   ("Config", {**original["Config"], "Env": ["PYTHONPATH=/injection"]}),
                   ("Mounts", [{"RW": True, "Destination": "/app"}])]
        for field, value in changes:
            container = copy.deepcopy(original)
            container[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                container_identity(container, image, contract, "http://127.0.0.1:8090")
        original["Mounts"] = [{"RW": False, "Destination": "/app/serve"}]
        contract["mounts_sha256"] = digest(original["Mounts"])
        with self.assertRaisesRegex(ValueError, "frozen image custody"):
            container_identity(original, image, contract, "http://127.0.0.1:8090")

    def test_udp_duplicate_mapping_and_wrong_network_refuse(self):
        container, image, contract = self.fixture()
        ports = container["NetworkSettings"]["Ports"]
        ports["80/udp"] = ports.pop("80/tcp")
        with self.assertRaisesRegex(ValueError, "published TCP"):
            container_identity(container, image, contract, "http://127.0.0.1:8090")
        ports["80/tcp"] = ports.pop("80/udp") * 2
        with self.assertRaisesRegex(ValueError, "published TCP"):
            container_identity(container, image, contract, "http://127.0.0.1:8090")


TEST_IO_TIMEOUT = 1.0
TEST_CLEANUP_TIMEOUT = 1.0
TEST_LIFETIME = 5.0
TEST_REAP_TIMEOUT = 5.0
TEST_IDLE = 1.25
TEST_PADDING_BYTES = 524288

# Only ProcessPins/protocol failure seams are replaced. The watchdog, framing, helper main and host I/O are real.
# The fixture opens no socket, reads no proc and invokes no Docker, serving process, model or native engine.
FAKE_PINS = r'''
class ProcessPins:
    def __init__(self, contract, target):
        self.contract, self.calls = contract, 0
        self.resource = os.fdopen(contract["proof_fd"], "wb") if "proof_fd" in contract else open(os.devnull, "rb")
        if "marker" in contract:
            Path(contract["marker"]).write_text("opened")
        if FIXTURE_MODE == "init-stuck":
            threading.Event().wait()

    def snapshot(self):
        self.calls += 1
        if FIXTURE_MODE == "error":
            raise OSError("fixture-sensitive-content")
        if FIXTURE_MODE == "snapshot-stuck" and self.calls == self.contract.get("stuck_call", 1):
            threading.Event().wait()
        return {"snapshot": self.calls}

    def close(self):
        if FIXTURE_MODE == "close-stuck":
            threading.Event().wait()
        self.resource.close()
        if "marker" in self.contract:
            Path(self.contract["marker"]).write_text("closed")

real_print = print

def print(value, flush=False):
    frame = json.loads(value)
    if "snapshot" in frame and FIXTURE_MODE == "stdout-stuck":
        threading.Event().wait()
    if "snapshot" in frame and FIXTURE_MODE == "wrong-snapshot-session":
        frame["session"] = "another-helper"
    if frame.get("closed"):
        if FIXTURE_MODE == "missing-ack":
            return
        if FIXTURE_MODE == "wrong-ack":
            frame["session"] = "another-helper"
    real_print(json.dumps(frame), flush=flush)

real_input = sys.stdin

class StuckInput:
    def readline(self):
        if FIXTURE_MODE == "stdin-stuck":
            threading.Event().wait()
        return real_input.readline()

    def __iter__(self):
        if FIXTURE_MODE == "next-stdin-stuck":
            threading.Event().wait()
        return iter(real_input)

sys.stdin = StuckInput()
'''


class CollectorProtocol(unittest.TestCase):
    def setUp(self):
        self.children, self.commands = [], []
        self.real_popen = subprocess.Popen
        self.addCleanup(self.reap)

    def reap(self):
        for child in self.children:
            for stream in (child.stdin, child.stdout):
                if stream is not None:
                    stream.close()
            if child.poll() is None:
                child.kill()  # Only this test's isolated Python fixture; never a serving process.
            child.wait(timeout=TEST_REAP_TIMEOUT)

    def source(self, mode):
        code = Path(evidence.__file__).read_text()
        code = code.replace('if __name__ == "__main__":',
                            "FIXTURE_MODE = " + repr(mode) + "\n" + FAKE_PINS + '\nif __name__ == "__main__":')
        if mode == "nonzero":
            code = code.replace('raise SystemExit(main(*sys.argv[1:]))',
                                'main(*sys.argv[1:]); raise SystemExit(7)')
        return code

    def spawn(self, mode, timeout=TEST_IO_TIMEOUT, cleanup=TEST_CLEANUP_TIMEOUT, lifetime=TEST_LIFETIME,
              session="fixture", proof_fd=None):
        child = self.real_popen([sys.executable, "-I", "-S", "-u", "-c", self.source(mode),
                                 str(timeout), str(cleanup), str(lifetime), session], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                                pass_fds=() if proof_fd is None else (proof_fd,))
        self.children.append(child)
        return child

    def collector(self, mode="normal", contract=None, **limits):
        def transport(argv, **kwargs):
            self.commands.append((argv, kwargs))
            return self.spawn(mode, *map(float, argv[10:13]), session=argv[13])
        with mock.patch.object(evidence.subprocess, "Popen", side_effect=transport):
            return Collector("f" * 64, {"interpreter": "/image/python", **(contract or {})}, {"port": 80},
                             timeout=TEST_IO_TIMEOUT, cleanup_timeout=TEST_CLEANUP_TIMEOUT,
                             lifetime=limits.get("lifetime", TEST_LIFETIME))

    def assert_released(self, collector):
        self.assertTrue(collector.proc.stdin.closed)
        self.assertTrue(collector.proc.stdout.closed)
        self.assertIsNotNone(collector.proc.poll())

    def write(self, child, body):
        child.stdin.write((json.dumps(body) + "\n").encode())

    def read(self, child):
        self.assertTrue(select.select([child.stdout], [], [], TEST_REAP_TIMEOUT)[0])
        return json.loads(child.stdout.readline())

    def test_single_persistent_exec_uses_isolated_python_and_acknowledges_closed_pins(self):
        with tempfile.TemporaryDirectory() as root:
            marker = str(Path(root) / "pins")
            collector = self.collector(contract={"marker": marker})
            self.assertEqual(collector.initial, {"snapshot": 1})
            self.assertEqual(collector.snapshot(), {"snapshot": 2})
            self.assertEqual(Path(marker).read_text(), "opened")
            with mock.patch.object(collector.proc, "kill", wraps=collector.proc.kill) as kill:
                collector.close()
            kill.assert_not_called()
            self.assertEqual(Path(marker).read_text(), "closed")
        self.assertTrue(collector.clean)
        self.assert_released(collector)
        self.assertEqual(len(self.commands), 1)
        argv, kwargs = self.commands[0]
        self.assertEqual(argv[:9], ["docker", "exec", "-i", "f" * 64, "/image/python", "-I", "-S", "-u", "-c"])
        self.assertEqual(kwargs["bufsize"], 0)
        self.assertNotIn("text", kwargs)
        collector.close()  # Successful close is idempotent; failed close must not become successful on retry.

    def test_collector_server_protocol_closes_pins_and_redacts_external_errors(self):
        for error in (None, EvidenceError("reciprocal pipe mismatch"), OSError("fixture-sensitive-content")):
            pins = mock.Mock()
            pins.snapshot.return_value = {"snapshot": 1}
            pins.snapshot.side_effect = error
            source = io.StringIO('{"contract":{},"target":{},"session":"fixture"}\n'
                                 '{"snapshot":true,"session":"fixture"}\n')
            output = io.StringIO()
            pins.close.side_effect = lambda: self.assertNotIn('"closed": true', output.getvalue())
            with self.subTest(error=type(error).__name__), \
                    mock.patch.object(evidence, "ProcessPins", return_value=pins), \
                    mock.patch.object(evidence.sys, "stdin", source), mock.patch.object(evidence.sys, "stdout", output):
                rc = evidence.main(session="fixture")
            pins.close.assert_called_once()
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(rc, int(error is not None))
            self.assertNotIn("fixture-sensitive-content", output.getvalue())
            self.assertEqual(records[-1], {"session": "fixture", "closed": True})
            if error is None:
                self.assertEqual(records[:-1], [{"session": "fixture", "snapshot": {"snapshot": 1}}] * 2)
            if isinstance(error, EvidenceError):
                self.assertEqual(records[0], {"session": "fixture", "error": "reciprocal pipe mismatch"})

    def test_missing_mismatched_ack_or_nonzero_transport_cannot_prove_remote_cleanup(self):
        for mode in ("missing-ack", "wrong-ack", "nonzero"):
            with self.subTest(mode=mode):
                collector = self.collector(mode)
                with self.assertRaises(ValueError):
                    collector.close()
                self.assertFalse(collector.clean)
                self.assert_released(collector)
                with self.assertRaisesRegex(ValueError, "previously failed"):
                    collector.close()

    def test_initial_refusal_or_wrong_session_is_bounded_and_releases_transport(self):
        for mode, reason in (("error", "collector refused: OSError"),
                             ("wrong-snapshot-session", "session mismatch"),
                             ("stdout-stuck", "collector")):
            before = time.monotonic()
            with self.subTest(mode=mode), self.assertRaisesRegex((ValueError, TimeoutError), reason):
                self.collector(mode)
            self.assertLess(time.monotonic() - before, TEST_IO_TIMEOUT + TEST_CLEANUP_TIMEOUT + TEST_REAP_TIMEOUT)
            self.assertIsNotNone(self.children[-1].poll())
            self.assertTrue(self.children[-1].stdin.closed and self.children[-1].stdout.closed)

    def test_initial_and_subsequent_blocked_stdin_share_the_exchange_deadline(self):
        before = time.monotonic()
        with self.assertRaisesRegex((ValueError, TimeoutError), "collector"):
            self.collector("stdin-stuck", {"padding": "x" * TEST_PADDING_BYTES})
        self.assertLess(time.monotonic() - before, TEST_IO_TIMEOUT + TEST_CLEANUP_TIMEOUT + TEST_REAP_TIMEOUT)
        collector = self.collector("next-stdin-stuck")
        before = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "I/O timed out"):
            collector.exchange({"snapshot": True, "session": collector.session, "padding": "x" * TEST_PADDING_BYTES})
        with self.assertRaises((ValueError, TimeoutError, subprocess.TimeoutExpired)):
            collector.close()
        self.assertLess(time.monotonic() - before, TEST_IO_TIMEOUT + TEST_CLEANUP_TIMEOUT + TEST_REAP_TIMEOUT)
        self.assertFalse(collector.clean)
        self.assert_released(collector)

    def test_idle_uses_capture_lifetime_not_snapshot_timeout(self):
        collector = self.collector()
        self.assertFalse(threading.Event().wait(TEST_IDLE))
        self.assertEqual(collector.snapshot(), {"snapshot": 2})
        collector.close()
        self.assert_released(collector)

    def test_failed_snapshot_close_returns_without_buffered_reader_lock(self):
        collector = self.collector("snapshot-stuck", {"stuck_call": 2})
        with self.assertRaises((ValueError, TimeoutError)):
            collector.snapshot()
        before = time.monotonic()
        with self.assertRaises((ValueError, TimeoutError, subprocess.TimeoutExpired)):
            collector.close()
        self.assertLess(time.monotonic() - before, TEST_CLEANUP_TIMEOUT + TEST_REAP_TIMEOUT)
        self.assertFalse(collector.clean)
        self.assert_released(collector)

    def test_watchdog_shutdown_join_is_bounded_and_cannot_disable_helper_exit(self):
        watchdog = evidence.HelperDeadline(TEST_IO_TIMEOUT, TEST_LIFETIME)
        watchdog.close(TEST_CLEANUP_TIMEOUT)
        with mock.patch.object(watchdog.thread, "join") as join, \
                mock.patch.object(watchdog.thread, "is_alive", return_value=True), \
                mock.patch.object(evidence.os, "_exit",
                                  side_effect=RuntimeError("fixture helper exit")) as exit_helper, \
                self.assertRaisesRegex(RuntimeError, "fixture helper exit"):
            watchdog.close(TEST_CLEANUP_TIMEOUT)
        join.assert_called_once_with(timeout=TEST_CLEANUP_TIMEOUT)
        exit_helper.assert_called_once_with(evidence.HELPER_DEADLINE_EXIT_DEFAULT)

    def test_actual_helper_deadline_exits_only_itself_and_kernel_releases_open_handle(self):
        cases = (("stdin-stuck", None), ("init-stuck", None), ("snapshot-stuck", 1),
                 ("snapshot-stuck", 2), ("close-stuck", None), ("stdout-stuck", None))
        with tempfile.TemporaryDirectory() as root:
            for index, (mode, call) in enumerate(cases):
                read_fd, write_fd = os.pipe()
                marker = str(Path(root) / str(index))
                child = self.spawn(mode, proof_fd=write_fd)
                body = {"contract": {"proof_fd": write_fd, "marker": marker, "stuck_call": call},
                        "target": {}, "session": "fixture"}
                self.write(child, body)
                os.close(write_fd)
                try:
                    if call == 2 or mode == "close-stuck":
                        self.assertIn("snapshot", self.read(child))
                    if call == 2:
                        self.write(child, {"snapshot": True, "session": "fixture"})
                    if mode == "close-stuck":
                        child.stdin.close()
                    with self.subTest(mode=mode, call=call):
                        self.assertEqual(child.wait(timeout=TEST_REAP_TIMEOUT), evidence.HELPER_DEADLINE_EXIT_DEFAULT)
                        self.assertTrue(select.select([read_fd], [], [], TEST_REAP_TIMEOUT)[0])
                        self.assertEqual(os.read(read_fd, 1), b"")  # Kernel closed the helper's inherited handle.
                        if mode != "stdin-stuck":
                            self.assertEqual(Path(marker).read_text(), "opened")
                finally:
                    os.close(read_fd)

    def test_eof_transport_loss_and_whole_lifetime_are_independent_of_local_cli_exit(self):
        child = self.spawn("normal")
        self.write(child, {"contract": {}, "target": {}, "session": "fixture"})
        self.assertIn("snapshot", self.read(child))
        child.stdin.close()
        self.assertEqual(self.read(child), {"session": "fixture", "closed": True})
        self.assertEqual(child.wait(timeout=TEST_REAP_TIMEOUT), 0)
        # Loss of the output transport must not strand the isolated helper.
        child = self.spawn("normal")
        self.write(child, {"contract": {}, "target": {}, "session": "fixture"})
        self.assertIn("snapshot", self.read(child))
        child.stdout.close()
        child.stdin.close()
        self.assertNotEqual(child.wait(timeout=TEST_REAP_TIMEOUT), 0)
        # No local kill/wait drives this idle helper's finite lifetime.
        child = self.spawn("normal", lifetime=TEST_IDLE)
        self.write(child, {"contract": {}, "target": {}, "session": "fixture"})
        self.assertIn("snapshot", self.read(child))
        self.assertEqual(child.wait(timeout=TEST_REAP_TIMEOUT), evidence.HELPER_DEADLINE_EXIT_DEFAULT)


if __name__ == "__main__":
    unittest.main()
