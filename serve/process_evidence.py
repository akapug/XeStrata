# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Linux delegated-worker collector, kept alive across the same capture window.

No HTTP, signals to serving processes, native engine protocol or model operations.
The accepted immutable image/frozen Python closure is trusted not to forge routing.
Each snapshot scans proc once; no process starts per PID, file, request or token.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import select
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid

PROC_ROOT_DEFAULT = "/proc"
READ_TIMEOUT_DEFAULT = 30
CLEANUP_TIMEOUT_DEFAULT = 30
CAPTURE_LIFETIME_DEFAULT = 21600  # Whole POST/monitor window; never the per-snapshot timeout.
TRANSPORT_REAP_RESERVE_DEFAULT = 1
PIPE_CHUNK_BYTES_DEFAULT = 65536
MAX_FRAME_BYTES_DEFAULT = 1048576
HELPER_DEADLINE_EXIT_DEFAULT = 124
HASH_CHUNK_BYTES_DEFAULT = 1048576
MAX_PROCESSES_DEFAULT = 4096
SECRET_ENV_KEYS_DEFAULT = frozenset(("API_KEY", "STRATA_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
INJECTION_ENV_KEYS_DEFAULT = frozenset((
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "LD_PRELOAD", "LD_AUDIT"))


class EvidenceError(ValueError):
    """Only fixed, nonsecret boundary diagnostics may cross the collector pipe."""


def require(ok, message):
    if not ok:
        raise EvidenceError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(stream):
    h = hashlib.sha256()
    while True:
        chunk = stream.read(HASH_CHUNK_BYTES_DEFAULT)
        if not chunk:
            return h.hexdigest()
        h.update(chunk)


def env_digest(values):
    """Never retain or hash credential values, even STRATA_API_KEY."""
    env = {}
    for value in values:
        key, sep, item = value.partition("=")
        require(bool(sep) and key not in env, "malformed or duplicate environment")
        require(key not in INJECTION_ENV_KEYS_DEFAULT or not item, "interpreter or loader injection is not accepted")
        if key not in SECRET_ENV_KEYS_DEFAULT:
            env[key] = item
    return digest(env)


def overlaps(a, b):
    return a == b or a.startswith(b.rstrip("/") + "/") or b.startswith(a.rstrip("/") + "/")


def container_identity(container, image, contract, url):
    """Docker-side immutable launch/custody checks, before opening the collector."""
    require(container["State"].get("Running") is True and not container["State"].get("Restarting"),
            "container not steadily running")
    require(container["Image"] == image["Id"] == contract["image"], "immutable image mismatch")
    host = container["HostConfig"]
    require(host.get("ReadonlyRootfs") is True and not host.get("Privileged") and
            not host.get("PidMode") and not host.get("IpcMode", "").startswith("container:") and
            host.get("NetworkMode") != "host" and not host.get("NetworkMode", "").startswith("container:"),
            "unsupported writable or shared-namespace container")
    require(not ({"SYS_PTRACE", "SYS_ADMIN", "ALL"} & set(host.get("CapAdd") or [])),
            "debugger or mount capability not accepted")
    launch = {k: container["Config"].get(k) for k in ("Entrypoint", "Cmd", "WorkingDir", "User")}
    require(digest(launch) == contract["launch_sha256"], "unaccepted launch")
    require(env_digest(container["Config"].get("Env") or []) == contract["docker_env_sha256"],
            "unaccepted container environment")
    mounts = container.get("Mounts") or []
    # Accepted data/log/model mounts may be writable; no mount may supply the frozen code/config closure.
    require(digest(mounts) == contract["mounts_sha256"] and
            digest(host.get("Tmpfs") or {}) == contract["tmpfs_sha256"], "unaccepted mounts")
    critical = [*contract["files"], *contract["import_roots"], contract["interpreter"], contract["engine"]]
    require(all(isinstance(p, str) and p.startswith("/") and ".." not in Path(p).parts for p in critical),
            "custody paths must be absolute")
    destinations = [m["Destination"] for m in mounts] + list(host.get("Tmpfs") or {})
    require(all(not overlaps(p, destination) for destination in destinations for p in critical),
            "mounted source/import/config/engine is not frozen image custody")
    require(contract["entrypoint_file"] in contract["files"] and contract["config_file"] in contract["files"] and
            bool(contract["import_roots"]) and digest(contract["files"]) == contract["closure_sha256"],
            "missing accepted source/import/config closure")
    origin = urllib.parse.urlsplit(url)
    require(origin.scheme == "http" and origin.hostname == "127.0.0.1" and not origin.username and
            not origin.password and origin.path in ("", "/") and not origin.query and not origin.fragment,
            "URL must be a loopback HTTP origin")
    targets = [int(port.split("/")[0]) for port, bindings in (container["NetworkSettings"].get("Ports") or {}).items()
               if port.endswith("/tcp") for b in (bindings or [])
               if b.get("HostPort") == str(origin.port or 80) and b.get("HostIp") in ("127.0.0.1", "0.0.0.0")]
    require(len(targets) == 1, "ambiguous or absent published TCP mapping")
    networks = container["NetworkSettings"].get("Networks") or {}
    require(len(networks) == 1, "ambiguous container network")
    address = next(iter(networks.values())).get("IPAddress")
    require(bool(address), "missing container IPv4 address")
    return {"container": container["Id"], "started": container["State"]["StartedAt"],
            "init_pid": container["State"]["Pid"], "image": image["Id"], "launch_sha256": digest(launch),
            "contract_sha256": digest(contract), "target_port": targets[0], "target_address": address}


class ProcView:
    def __init__(self, root=PROC_ROOT_DEFAULT, max_processes=MAX_PROCESSES_DEFAULT):
        self.root = Path(root)
        self.max_processes = max_processes

    def pids(self):
        pids = sorted(int(p.name) for p in self.root.iterdir() if p.name.isdecimal())
        require(0 < len(pids) <= self.max_processes, "unreadable or oversized process census")
        return pids

    def process(self, pid):
        p = self.root / str(pid)
        fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "ppid": int(fields[1]), "start": int(fields[19]),
                "net_ns": os.readlink(p / "ns/net"), "pid_ns": os.readlink(p / "ns/pid")}

    def args(self, pid):
        return (self.root / str(pid) / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")

    def environment(self, pid):
        raw = (self.root / str(pid) / "environ").read_bytes().rstrip(b"\0")
        return env_digest(raw.decode().split("\0") if raw else [])

    def fds(self, pid):
        p = self.root / str(pid)
        result = {}
        # Our own proc directory includes the scan's descriptor. Keep it open until each
        # entry is inspected; Path.iterdir() closes its scan before exposing those names.
        with os.scandir(p / "fd") as entries:
            for fd in entries:
                target = os.readlink(fd.path)
                flags = next(line.split()[1] for line in (p / "fdinfo" / fd.name).read_text().splitlines()
                             if line.startswith("flags:"))
                result[int(fd.name)] = (target, int(flags, 8) & os.O_ACCMODE)
        return result

    def pipe_identity(self, pid, fd):
        s = (self.root / str(pid) / "fd" / str(fd)).stat()
        require(stat.S_ISFIFO(s.st_mode), "worker transport is not a pipe")
        return [s.st_dev, s.st_ino]

    def listener(self, target):
        matches = []
        for family in ("tcp", "tcp6"):
            for line in (self.root / "net" / family).read_text().splitlines()[1:]:
                fields = line.split()
                address, port = fields[1].split(":")
                if fields[3] != "0A" or int(port, 16) != target["target_port"]:
                    continue
                require(family == "tcp", "ambiguous IPv6 listener reachability")
                ip = socket.inet_ntoa(bytes.fromhex(address)[::-1])
                if ip in ("0.0.0.0", target["target_address"]):
                    matches.append(fields[9])
        require(len(matches) == 1, "ambiguous or missing listening socket")
        return "socket:[" + matches[0] + "]"


class ProcessPins:
    """One lifetime owner: both pidfds and executable handles survive the whole POST window."""
    def __init__(self, contract, target, view=None):
        self.contract, self.target = contract, target
        self.view = view or ProcView(max_processes=contract.get("max_processes", MAX_PROCESSES_DEFAULT))
        self.pins, self.baseline = [], None

    def close(self):
        for _, pidfd, exe in self.pins:
            exe.close()
            os.close(pidfd)
        self.pins.clear()

    def executable(self, pid, expected):
        if self.baseline is None:
            fd = os.pidfd_open(pid)
            try:
                exe = (self.view.root / str(pid) / "exe").open("rb")
            except BaseException:
                os.close(fd)
                raise
            self.pins.append((pid, fd, exe))
        pinned = next(exe for selected, _, exe in self.pins if selected == pid)
        pinned.seek(0)
        pinned_stat, pinned_hash = os.fstat(pinned.fileno()), file_digest(pinned)
        with (self.view.root / str(pid) / "exe").open("rb") as actual:
            s = os.fstat(actual.fileno())
            h = file_digest(actual)
        require(h == pinned_hash == expected and (s.st_dev, s.st_ino) == (pinned_stat.st_dev, pinned_stat.st_ino),
                "actual process executable mismatch")
        return {"sha256": h, "device": s.st_dev, "inode": s.st_ino}

    def snapshot(self):
        c, v = self.contract, self.view
        for _, pidfd, _ in self.pins:
            require(not select.select([pidfd], [], [], 0)[0], "pinned process exited")
        listener = v.listener(self.target)
        processes = {pid: v.process(pid) for pid in v.pids()}
        fdmap = {pid: v.fds(pid) for pid in processes}
        owners = [pid for pid, fds in fdmap.items() if any(link == listener for link, _ in fds.values())]
        require(len(owners) == 1, "listener ownership ambiguous")
        wrapper = owners[0]
        children = [pid for pid, p in processes.items() if p["ppid"] == wrapper and v.args(pid) == c["worker_argv"]]
        require(len(children) == 1, "missing or ambiguous selected worker")
        worker = children[0]
        require(
            self.baseline is None or
            (wrapper == self.baseline["listener"]["pid"] and worker == self.baseline["child"]["pid"]),
            "listener or worker PID changed")
        require(processes[wrapper]["net_ns"] == processes[worker]["net_ns"] == processes[1]["net_ns"] and
                processes[wrapper]["pid_ns"] == processes[worker]["pid_ns"] == processes[1]["pid_ns"],
                "listener/worker namespace mismatch")
        require(v.args(wrapper) == c["wrapper_argv"], "unaccepted Python wrapper command")
        require(os.readlink(v.root / str(wrapper) / "cwd") == c["wrapper_cwd"] and
                os.readlink(v.root / str(worker) / "cwd") == c["worker_cwd"], "unaccepted process working directory")
        entry = c["wrapper_argv"][1:]
        while entry and entry[0] in ("-u", "-B", "-I", "-S"):
            entry = entry[1:]
        module = c.get("wrapper_module", "serve.server")
        require(bool(entry) and (entry[0] == c["entrypoint_file"] or
                (entry[:2] == ["-m", module] and
                 c["entrypoint_file"].endswith("/" + module.replace(".", "/") + ".py"))),
                "command does not load accepted wrapper entrypoint")
        require("--engine" in c["wrapper_argv"] and
                c["wrapper_argv"][c["wrapper_argv"].index("--engine") + 1] == "strata" and
                "--config" in c["wrapper_argv"] and
                c["wrapper_argv"][c["wrapper_argv"].index("--config") + 1] == c["config_file"] and
                not any(a in ("-c", "-", "--script", "--api-key", "--replicas") or
                        a.startswith(("--api-key=", "--script=", "--replicas=")) for a in c["wrapper_argv"]),
                "wrapper does not select the accepted single StrataEngine configuration")
        require(v.environment(wrapper) == c["wrapper_env_sha256"] and v.environment(worker) == c["worker_env_sha256"],
                "effective process environment mismatch")
        require(all(os.path.realpath(path) == path for path in (*c["files"], *c["import_roots"])),
                "source/import/config path escapes frozen image custody through a symlink")
        for path, expected in c["files"].items():
            with open(path, "rb") as stream:
                require(file_digest(stream) == expected, "frozen source/import/config mismatch")
        cfg = json.loads(Path(c["config_file"]).read_text())
        require(cfg.get("replicas") in (None, False, 0, 1) and not cfg.get("before_load") and
                ("--api-monitor" in c["wrapper_argv"] or cfg.get("api_monitor") is True),
                "configuration selects unsupported routing or no monitor")
        require(not any(k in cfg for k in ("api_key", "password", "token")), "secret-bearing config not accepted")
        source = self.executable(wrapper, c["interpreter_sha256"])
        engine = self.executable(worker, c["engine_sha256"])
        require(os.path.realpath(self.view.root / str(wrapper) / "exe") == c["interpreter"] and
                os.path.realpath(self.view.root / str(worker) / "exe") == c["engine"],
                "process executable is outside accepted frozen image paths")
        child = fdmap[worker]
        require(0 in child and 1 in child and child[0][0].startswith("pipe:[") and child[1][0].startswith("pipe:[") and
                child[0][1] == os.O_RDONLY and child[1][1] == os.O_WRONLY and child[0][0] != child[1][0],
                "worker stdin/stdout pipe directions invalid")
        parent = fdmap[wrapper]
        stdin = [fd for fd, (link, mode) in parent.items() if link == child[0][0] and mode == os.O_WRONLY]
        stdout = [fd for fd, (link, mode) in parent.items() if link == child[1][0] and mode == os.O_RDONLY]
        require(len(stdin) == len(stdout) == 1, "missing or ambiguous reciprocal wrapper pipes")
        for pipe, expected_ends in ((child[0][0], {(worker, 0), (wrapper, stdin[0])}),
                                    (child[1][0], {(worker, 1), (wrapper, stdout[0])})):
            ends = {(pid, fd) for pid, fds in fdmap.items() for fd, (link, _) in fds.items() if link == pipe}
            require(ends == expected_ends, "extra or ambiguous worker pipe endpoints")
        in_id, out_id = v.pipe_identity(worker, 0), v.pipe_identity(worker, 1)
        require(v.pipe_identity(wrapper, stdin[0]) == in_id and v.pipe_identity(wrapper, stdout[0]) == out_id,
                "reciprocal pipe object mismatch")
        result = {"listener": processes[wrapper], "child": processes[worker], "socket": listener,
                  "interpreter": source, "engine": engine,
                  "worker": {"wrapper_pid": wrapper, "worker_pid": worker,
                             "stdin_pipe": in_id, "stdout_pipe": out_id},
                  "closure_sha256": c["closure_sha256"]}
        # Close PID-reuse races while opening handles or walking proc, including on the initial snapshot.
        require(v.process(wrapper) == processes[wrapper] and v.process(worker) == processes[worker] and
                not any(select.select([fd], [], [], 0)[0] for _, fd, _ in self.pins), "process generation changed")
        require(v.listener(self.target) == listener and
                all(v.fds(pid).get(fd) == fdmap[pid][fd] for pid, fd in
                    ((worker, 0), (worker, 1), (wrapper, stdin[0]), (wrapper, stdout[0]))) and
                any(link == listener for link, _ in v.fds(wrapper).values()),
                "listener or pipe changed during snapshot")
        if self.baseline is not None:
            require(result == self.baseline, "listener/worker/source/pipe identity changed during capture")
        self.baseline = result
        return result


class HelperDeadline:
    """Only this injected helper may exit. A local docker CLI kill is never remote cleanup evidence."""
    def __init__(self, snapshot_timeout, lifetime):
        self.limit = time.monotonic() + lifetime
        self.deadline = min(self.limit, time.monotonic() + snapshot_timeout)
        self.condition, self.stopped = threading.Condition(), False
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()  # Before reading initial stdin or constructing/inspecting ProcessPins.

    def arm(self, timeout=None):
        with self.condition:
            self.deadline = min(self.limit, time.monotonic() + timeout) if timeout is not None else self.limit
            self.condition.notify()

    def watch(self):
        with self.condition:
            while not self.stopped:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    # sys.exit here would end only this thread. Kernel process exit closes all owned handles,
                    # including when a blocked snapshot/cleanup cannot reach its finally. Never signal a PID.
                    os._exit(HELPER_DEADLINE_EXIT_DEFAULT)
                self.condition.wait(remaining)

    def close(self, timeout):
        with self.condition:
            self.stopped = True
            self.condition.notify()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            os._exit(HELPER_DEADLINE_EXIT_DEFAULT)


class Collector:
    def __init__(self, container_id, contract, target, timeout=READ_TIMEOUT_DEFAULT,
                 cleanup_timeout=CLEANUP_TIMEOUT_DEFAULT, lifetime=CAPTURE_LIFETIME_DEFAULT):
        require(all(math.isfinite(t) and t > 0 for t in (timeout, cleanup_timeout, lifetime)),
                "invalid collector timeout")
        self.timeout, self.cleanup_timeout = timeout, cleanup_timeout
        self.session, self.buffer = uuid.uuid4().hex, bytearray()
        self.closed, self.clean = False, False
        source = Path(__file__).read_text()
        self.proc = subprocess.Popen(["docker", "exec", "-i", container_id, contract["interpreter"],
                                      "-I", "-S", "-u", "-c", source, str(timeout), str(cleanup_timeout),
                                      str(lifetime), self.session], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        try:
            os.set_blocking(self.proc.stdin.fileno(), False)
            os.set_blocking(self.proc.stdout.fileno(), False)
            self.initial = self.exchange({"contract": contract, "target": target, "session": self.session})
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Preserve the primary refusal; close still releases only our local transport boundedly.
            raise

    def ready(self, stream, deadline, write=False):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not any(select.select([] if write else [stream],
                                                   [stream] if write else [], [], remaining)[:2]):
            raise TimeoutError("process collector I/O timed out")

    def send(self, request, deadline):
        data = (json.dumps(request) + "\n").encode()
        require(len(data) <= MAX_FRAME_BYTES_DEFAULT, "collector command oversized")
        offset = 0
        while offset < len(data):
            self.ready(self.proc.stdin, deadline, write=True)
            try:
                sent = os.write(self.proc.stdin.fileno(), data[offset:offset + PIPE_CHUNK_BYTES_DEFAULT])
            except BlockingIOError:
                continue
            require(sent > 0, "collector input closed")
            offset += sent

    def reply(self, deadline):
        while b"\n" not in self.buffer:
            self.ready(self.proc.stdout, deadline)
            try:
                chunk = os.read(self.proc.stdout.fileno(), PIPE_CHUNK_BYTES_DEFAULT)
            except BlockingIOError:
                continue
            require(bool(chunk), "collector ended without evidence")
            self.buffer.extend(chunk)
            require(len(self.buffer) <= MAX_FRAME_BYTES_DEFAULT, "collector response oversized")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = rest
        reply = json.loads(line)
        require(isinstance(reply, dict) and reply.get("session") == self.session, "collector session mismatch")
        require("error" not in reply, "collector refused: " + str(reply.get("error", "")))
        return reply

    def exchange(self, request):
        require(not self.closed, "collector already closed")
        deadline = time.monotonic() + self.timeout
        self.send(request, deadline)  # The same deadline covers blocked stdin as well as blocked stdout.
        reply = self.reply(deadline)
        require(set(reply) == {"session", "snapshot"} and isinstance(reply["snapshot"], dict),
                "missing collector snapshot")
        return reply["snapshot"]

    def snapshot(self):
        return self.exchange({"snapshot": True, "session": self.session})

    def close(self):
        if self.closed:
            require(self.clean, "collector cleanup previously failed")
            return
        self.closed = True
        deadline = time.monotonic() + self.cleanup_timeout
        reserve = min(TRANSPORT_REAP_RESERVE_DEFAULT, self.cleanup_timeout / 2)
        graceful = deadline - reserve
        try:
            self.proc.stdin.close()  # Unbuffered EOF; there is no reader thread or buffered close lock.
            require(self.reply(graceful) == {"session": self.session, "closed": True} and not self.buffer,
                    "missing collector completion acknowledgement")
            require(self.proc.wait(timeout=max(0, graceful - time.monotonic())) == 0,
                    "process collector did not exit cleanly")
            require(os.read(self.proc.stdout.fileno(), PIPE_CHUNK_BYTES_DEFAULT) == b"",
                    "unexpected output after collector completion")
            self.clean = True
        finally:
            # These are only the pipes/local CLI we created. Never claim this fallback killed a remote helper:
            # missing acknowledgement/clean exec status is refusal, and the helper has its own lifetime limit.
            self.proc.stdin.close()
            self.proc.stdout.close()
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=max(0, deadline - time.monotonic()))


def main(snapshot_timeout=READ_TIMEOUT_DEFAULT, cleanup_timeout=CLEANUP_TIMEOUT_DEFAULT,
         lifetime=CAPTURE_LIFETIME_DEFAULT, session=None):
    pins, watchdog, reason = None, None, None
    try:
        snapshot_timeout, cleanup_timeout, lifetime = map(float, (snapshot_timeout, cleanup_timeout, lifetime))
        require(all(math.isfinite(t) and t > 0 for t in (snapshot_timeout, cleanup_timeout, lifetime)) and
                isinstance(session, str) and bool(session), "invalid collector limits or session")
        watchdog = HelperDeadline(snapshot_timeout, lifetime)
        initial = json.loads(sys.stdin.readline())
        require(initial.get("session") == session, "collector session mismatch")
        pins = ProcessPins(initial["contract"], initial["target"])
        print(json.dumps({"session": session, "snapshot": pins.snapshot()}), flush=True)
        watchdog.arm()  # Idle through POSTs/monitor reads under the whole-capture lifetime, not snapshot timeout.
        for line in sys.stdin:
            watchdog.arm(snapshot_timeout)
            require(json.loads(line) == {"snapshot": True, "session": session}, "unsupported collector command")
            print(json.dumps({"session": session, "snapshot": pins.snapshot()}), flush=True)
            watchdog.arm()
    except Exception as error:
        # Errors may contain no inspected file content, argv or environment values.
        reason = str(error) if isinstance(error, EvidenceError) else type(error).__name__
    finally:
        if watchdog is not None:
            watchdog.arm(cleanup_timeout)
        try:
            if pins is not None:
                pins.close()
            if reason is not None:
                print(json.dumps({"session": session, "error": reason}), flush=True)
            print(json.dumps({"session": session, "closed": True}), flush=True)  # Only after all pins close.
        finally:
            if watchdog is not None:
                watchdog.close(cleanup_timeout)
    return int(reason is not None)


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
