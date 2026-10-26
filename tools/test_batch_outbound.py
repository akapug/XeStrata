# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Host-only controls of BatchOutbound and the actual pipelined scheduler; no SYCL, device or model."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


def between(text, start, end):
    if text.count(start) != 1:
        raise AssertionError(f"source boundary not unique: {start}")
    parts = text.split(start, 1)[1].split(end, 1)
    if len(parts) != 2 or not parts[0].strip():
        raise AssertionError(f"missing/empty source boundary: {end}")
    return parts[0]


def harness():
    program = (ROOT / "src/program/generate.cpp").read_text()
    source = (ROOT / "src/core/verify.cpp").read_text()
    group = "struct PGroup {" + between(program, "struct PGroup {", "        std::vector<PGroup> pg")
    pump = "auto pump =" + between(program, "auto pump =", "            if (!pipe_inflight() && !batch_on()")
    pump += "return true; };"
    delivery = "int TestStage::batch_outbound_poll(" + between(
        source, "int Verifier::batch_outbound_poll(", "bool Verifier::capture_batch(")
    release = "if (outbound_.device_rows() && !outbound_.start" + between(
        source, "if (outbound_.device_rows() && !outbound_.start", "void Verifier::BatchTimes::set_handoff(")
    # The production batch_poll tail includes its closing brace. Compile it inside the mock stage, unchanged.
    return textwrap.dedent(r'''
        #define main ownership_main
        #include "@ROOT@/src/core/batch_outbound_test.cpp"
        #undef main
        #include <atomic>
        #include <chrono>
        #include <condition_variable>
        #include <thread>
        namespace strata::kernels { constexpr int kVerifyMaxT = 64; }
        namespace strata::core {
            struct Progress { std::atomic<bool> busy{false}; };
            Progress& progress() { static Progress p; return p; }
        }
        struct OnDevice { explicit OnDevice(int) {} };
        void progress_beat() {}
        struct Receiver {
            std::vector<float> in;
            std::vector<bool> ready;
            explicit Receiver(int n) : in(n, -1), ready(n, false) {}
            void rows(const float* p, int base, int n) {
                for (int t = 0; t < n; ++t) { in[base + t] = p[base + t]; ready[base + t] = true; }
            }
        };
        struct TestStage {
            BatchOutbound outbound_;
            fake::Queue* cs_ = static_cast<fake::Queue*>(gpu::stream_create_copy());
            gpu::Event done;
            std::vector<float> host;
            Receiver* next_ = nullptr;
            Receiver* input = nullptr;
            float* hand_out_;
            int device_ = 0, row_base_ = 0, count = 0;
            bool final, b_running_ = false;
            int windows = 0;
            double ms_pool = 0;
            int32_t output[64] = {};
            TestStage(int n, bool last, bool async) : host(n, -1), hand_out_(host.data()), final(last) {
                if (async && !last) check(outbound_.init(host.data(), n, 1));
            }
            ~TestStage() { gpu::stream_sync(cs_); gpu::stream_destroy(cs_); }
            bool batch_launch(int base, int S, const int32_t* tok, const int64_t*, std::string&) {
                check(!b_running_);
                if (outbound_.device_rows()) check(outbound_.writable(base, S));
                std::vector<int> tokens(tok, tok + S);
                if (final) for (int t = 0; t < S; ++t) {
                    check(input->ready[base + t] && input->in[base + t] == tokens[t] + 1);
                    input->ready[base + t] = false;
                }
                auto compute = fake::submit(cs_, [=, this] {
                    float* p = outbound_.device_rows() ? outbound_.device_rows() : host.data();
                    for (int t = 0; t < S; ++t)
                        if (final) output[t] = tokens[t] + 2; else p[base + t] = (float) tokens[t] + 1;
                });
                // Only stage zero is used to measure same-stage compute/copy overlap.
                compute->compute = !final;
                compute->ticks = 4;
                if (!final && !outbound_.device_rows()) {
                    auto mapped_copy = fake::submit(cs_, [] {});
                    mapped_copy->copy = true;
                    mapped_copy->ticks = fake::copy_ticks;
                }
                check(gpu::event_record(&done, cs_));
                row_base_ = base; count = S; b_running_ = true;
                return true;
            }
            int batch_poll(void*, void*, std::string& err) {
                if (!b_running_) return 1;
                if (gpu::event_status(&done) == gpu::Status::Pending) return 0;
                const int S = count;
                if (!final && !outbound_.device_rows()) next_->rows(hand_out_, row_base_, S);
                @RELEASE@
            int batch_outbound_poll(int base, std::string& err);
            const int32_t* batch_out() const { return output; }
        };
        @DELIVERY@
        using Clock = std::chrono::steady_clock;
        void scheduler_control(bool asynchronous, int groups, bool cancel) {
            const int GS = 2, n_pipe = 2;
            const int n = GS * groups;
            fake::overlap = 0;
            fake::copy_ticks = 6;   // deterministic delayed event, not wall-clock timing or a performance estimate
            std::vector<std::unique_ptr<TestStage>> stages;
            stages.push_back(std::make_unique<TestStage>(n, false, asynchronous));
            stages.push_back(std::make_unique<TestStage>(n, true, false));
            Receiver receiver(n);
            stages[0]->next_ = &receiver;
            stages[1]->input = &receiver;
            auto stage_ver = [&](int k) -> TestStage& { return *stages[(size_t) k]; };
            @GROUP@
            std::vector<PGroup> pg(groups);
            std::vector<int> stage_group(n_pipe, -1);
            struct BSlot {
                bool active = true, stop = false, cached = false, img = false;
                int32_t x = 1;
                int64_t p = 0, produced = 0, max_new = 3;
                std::vector<int> ids;
                Clock::time_point t0 = Clock::now();
            };
            std::vector<BSlot> bs(n);
            struct Options { std::vector<int64_t> eos_ids; int prompt_cache = 0, batch_groups = 0; } o;
            o.batch_groups = groups;
            auto slot_cells = [](const Options&, int) { return 100; };
            auto group_active = [&](int gi) {
                for (int t = 0; t < GS; ++t) if (bs[gi * GS + t].active) return true;
                return false;
            };
            auto pipe_inflight = [&] { for (auto& g : pg) if (g.inflight) return true; return false; };
            int64_t pipe_tick = 0, rr = 0, bt_windows = 0, bt_rows = 0;
            Clock::time_point bt_start;
            std::vector<double> pipe_pool0(n_pipe), st_run(n_pipe), st_launch(n_pipe);
            std::vector<int64_t> st_n(n_pipe), tl_ready(groups);
            std::vector<Clock::time_point> st_t0(n_pipe);
            const int tl_every = std::atoi("0");
            int tl_steps = 0;
            auto tl_ns = [](Clock::time_point) { return 0; };
            auto tl_done = [](int, int) {};
            auto tl_launch = [](int, int, Clock::time_point) {};
            auto tl_report = [] {};
            auto apply_pending = [](bool) {};
            std::string err;
            void* win_pool_fn = nullptr;
            void* win_pool_user = nullptr;
            struct Drive { struct { bool failed = false; const char* fail = nullptr; } d; } drive, drive1;
            struct StagePoll {
                std::atomic<int> result{3};
                std::string err;
                std::mutex mu;
                std::condition_variable cv;
                bool go = false;
            };
            const bool pipe_threads = false;
            std::vector<std::unique_ptr<StagePoll>> spoll;
            @PUMP@
            bool drained = false;
            for (int tick = 0; tick < 2000; ++tick) {
                check(pump(!cancel || tick < 12));
                for (auto* q : fake::queues) fake::step(q);
                if (tick >= 12 && !pipe_inflight()) { drained = true; break; }
                if (cancel && tick == 11) for (auto& sl : bs) sl.stop = true;
            }
            check(drained && err.empty());
            for (const auto& sl : bs) {
                if (cancel) check(sl.produced <= 1); else check(sl.produced == 3 && sl.x == 5);
                for (size_t t = 0; t < sl.ids.size(); ++t) check(sl.ids[t] == 1 + 2 * (int) t);
            }
            check(asynchronous ? fake::overlap > 0 : fake::overlap == 0);
            std::printf("scheduler: async=%d groups=%d cancel=%d overlap-work=%d drain=ok\n",
                        asynchronous, groups, cancel, fake::overlap);
            fake::settle();
        }
        int main() {
            ownership_controls();
            for (bool async : {false, true}) for (int groups : {2, 3}) for (bool cancel : {false, true})
                scheduler_control(async, groups, cancel);
            check(fake::allocations.empty() && fake::queues.empty());
        }
    ''').replace("@ROOT@", str(ROOT)).replace("@GROUP@", group).replace("@PUMP@", pump).replace(
        "@DELIVERY@", delivery).replace("@RELEASE@", release)


class BatchOutboundControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or not shutil.which(compiler[0]):
            raise RuntimeError("host C++ compiler required for outbound ownership controls")
        cls.tmp = tempfile.TemporaryDirectory(prefix="batch-outbound-controls-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.build = Path(cls.tmp.name)
        cls.ownership = cls.build / "ownership"
        cls.scheduler = cls.build / "scheduler"
        common = compiler + ["-std=c++20", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(ROOT / "include")]
        subprocess.run(common + [str(ROOT / "src/core/batch_outbound_test.cpp"), "-o", str(cls.ownership)],
                       check=True)
        source = cls.build / "scheduler.cpp"
        source.write_text(harness())
        subprocess.run(common + [str(source), "-o", str(cls.scheduler)], check=True)

    def test_ownership_and_actual_scheduler(self):
        result = subprocess.run([str(self.scheduler)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("scheduler:"), 8, result.stdout)
        print(result.stdout)

    def test_causal_removed_dependency_drain_and_ack_guard(self):
        original = (ROOT / "include/strata/core/batch_outbound.hpp").read_text()
        mutations = {
            "dependency": ("stream_, t.ready, t.done", "stream_, nullptr, t.done"),
            "drain": ("if (stream_) gpu::stream_sync(stream_);", "/* missing drain */"),
            "ack": ("if (t.state != State::Ready) return false;", "/* premature acknowledgement */"),
        }
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        for name, (old, new) in mutations.items():
            with self.subTest(name=name):
                self.assertEqual(original.count(old), 1)
                root = self.build / name
                header = root / "strata/core/batch_outbound.hpp"
                header.parent.mkdir(parents=True)
                header.write_text(original.replace(old, new))
                binary = root / "control"
                subprocess.run(compiler + ["-std=c++20", "-pthread", "-I", str(root), "-I", str(ROOT / "include"),
                                           str(ROOT / "src/core/batch_outbound_test.cpp"), "-o", str(binary)],
                               check=True)
                result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, f"mutation not detected: {name}")

    def test_stalled_copy_timeout(self):
        for value, expected in ((None, 20), ("2", 2), ("0", 20)):
            env = {k: v for k, v in os.environ.items() if k != "STRATA_BATCH_OUTBOUND_TIMEOUT_S"}
            if value is not None:
                env["STRATA_BATCH_OUTBOUND_TIMEOUT_S"] = value
            result = subprocess.run([str(self.ownership), "timeout"], env=env, capture_output=True, text=True,
                                    timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"timeout={expected} late delivery refused", result.stdout)
            print(result.stdout)

    def test_restart_only_gate(self):
        for value in (None, "0", "1"):
            env = {k: v for k, v in os.environ.items() if k != "STRATA_BATCH_OUTBOUND_ASYNC"}
            if value is not None:
                env["STRATA_BATCH_OUTBOUND_ASYNC"] = value
            result = subprocess.run([str(self.ownership), "gate"], env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "1" if value == "1" else "0")


if __name__ == "__main__":
    unittest.main()
