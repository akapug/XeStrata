# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model-free source harness: execute the actual hand-off block, reset, validator and timeline reporter.

Only a host C++ compiler is needed, not SYCL, an SDK, a GPU or a model. This is not a native engine build.
Mutation controls must fail when either branch loses its end stamp, the per-replay reset is removed, or the
validator accepts stale stamps. Run through the normal test runner:
python3 -m unittest tools.test_batch_handoff_timeline.
"""
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
    """Require unique, non-empty source slices; a moved boundary fails rather than testing an empty fixture."""
    if text.count(start) != 1:
        raise AssertionError(f"source boundary is not unique: {start}")
    part = text.split(start, 1)[1].split(end, 1)
    if len(part) != 2 or not part[0].strip():
        raise AssertionError(f"source boundary is missing or empty: {end}")
    return part[0]


def harness():
    source = (ROOT / "src/core/verify.cpp").read_text()
    header = (ROOT / "include/strata/core/verify.hpp").read_text()
    program = (ROOT / "src/program/generate.cpp").read_text()
    times = "struct BatchTimes {" + between(header, "struct BatchTimes {", "    static int batch_timeline();")
    validator = "void BatchTimes::set_handoff(" + between(
        source, "void Verifier::BatchTimes::set_handoff(", "int Verifier::batch_timeline()")
    copies = between(source, "    // The non-final stage has no head:", "    // ---- the head,")
    copies = copies[copies.index("    const bool handoff_tl"):]
    reset = between(source, "        // Clear both hand-off and calibration pairs",
                    "        bt_cur_.h_submit = ns_now();")
    reset = reset[reset.index("        if (prof_on_"):]
    launch = between(source, "bool Verifier::batch_launch(", "int Verifier::batch_poll(")
    if not launch.index("if (tl) {") < launch.index(reset.strip()) < launch.index("event_record(tl_ev_[0]"):
        raise AssertionError("reset is not inside the timeline gate and before the start tag")
    poll = between(source, "int Verifier::batch_poll(", "void Verifier::BatchTimes::set_handoff(")
    if not poll.index("Status::Pending") < poll.index("const bool copied") < poll.index("bt_cur_.set_handoff"):
        raise AssertionError("stamps are not read only after completion and a checked copy")
    if "if (copied) {" not in poll or "prof_h_[i + 2], prof_h_[i], prof_h_[i + 1], prof_h_[i + 3], ms" not in poll:
        raise AssertionError("the same-window stamp envelope or copy guard is missing")
    readback = "if (prof_on_) { //" + between(
        poll, "if (prof_on_) {   // STRATA_VERIFY_PROFILE=1:", "    if (batch_timeline() > 0) bt_last_ = bt_cur_;")
    if "prof_tag_ok_[end] = strata::gpu::event_record(prof_ev_[end], cs_);" not in source:
        raise AssertionError("calibration record success is not refreshed per replay")
    accum = "enum { kTlPrep" + between(program, "enum { kTlPrep", "        std::vector<TlAcc> tla")
    done = "auto tl_done =" + between(program, "auto tl_done =", "        auto tl_launch =")
    report = "auto tl_report =" + between(program, "auto tl_report =", "        if (tl_every > 0 && piped)")
    return textwrap.dedent(r'''
        #include <algorithm>
        #include <cmath>
        #include <cstdint>
        #include <cstdio>
        #include <cstring>
        #include <limits>
        #include <stdexcept>
        #include <string>
        #include <vector>
        using std::size_t;
        void check(bool ok) { if (!ok) throw std::runtime_error("causal check failed"); }
        @TIMES@
        @VALIDATOR@
        namespace strata::core { struct Verifier { using BatchTimes = ::BatchTimes; }; }
        std::vector<int> trace;
        struct RowsCopy { float* dst; const float* src; int64_t n, dst_stride, src_stride; };
        void copy_from_mapped(float* dst, const float* src, int64_t n, void*) {
            trace.push_back(1);
            std::copy(src, src + n, dst);
        }
        void copy_rows_mapped_multi(const RowsCopy* pc, int pieces, int rows, void*) {
            trace.push_back(1);
            for (int r = 0; r < rows; ++r)
                for (int p = 0; p < pieces; ++p)
                    std::copy(pc[p].src + r * pc[p].src_stride,
                              pc[p].src + r * pc[p].src_stride + pc[p].n,
                              pc[p].dst + r * pc[p].dst_stride);
        }
        bool reset_ok = true, copy_ok = true, elapsed_ok = true;
        int reset_calls = 0, elapsed_calls = 0;
        namespace strata::kernels { const char* gpu_stamp_clock() { return "sub_group"; } }
        namespace strata::gpu {
            bool copy(void* dst, const void* src, size_t bytes) {
                if (!copy_ok) return false;
                std::memcpy(dst, src, bytes);
                return true;
            }
            bool event_elapsed_ms(float* ms, const void*, const void*) {
                ++elapsed_calls;
                *ms = 8;
                return elapsed_ok;
            }
            bool memset_async(void* dst, int value, size_t bytes, void*) {
                ++reset_calls;
                if (!reset_ok) return false;
                std::memset(dst, value, bytes);
                return true;
            }
            const char* last_error() { return "injected reset failure"; }
        }
        bool reset_span(bool tl, bool prof_on_, bool final, unsigned long long* prof_) {
            struct Geometry { int n_layers = 2; } g;
            auto* g_ = &g;
            constexpr int kProfPer = 33;
            void* cs_ = nullptr;
            auto last_stage = [&] { return final; };
            std::string err;
            if (tl) {
                @RESET@
            }
            return true;
        }
        void test_copies() {
            const int N = 8, HC = 4, HB = HC * N + N + HC;
            struct Geometry { int n_layers = 4; } g;
            for (int T : {1, 16}) for (int hrow0 : {0, 3}) for (bool one : {false, true})
            for (bool batch_rec_ : {false, true}) for (bool asynchronous : {false, true})
            for (bool timeline : {false, true})
            for (bool prof_on_ : {false, true}) for (int le_ : {2, 4}) {
                std::vector<float> r(T * HC * N), bo(T * N), inj(T * HC), hand((hrow0 + T + 1) * HB, -9);
                for (size_t i = 0; i < r.size(); ++i) r[i] = 1000 + i;
                for (size_t i = 0; i < bo.size(); ++i) bo[i] = 2000 + i;
                for (size_t i = 0; i < inj.size(); ++i) inj[i] = 3000 + i;
                std::vector<float> staged(hand.size(), -9);
                struct Outbound { float* p; float* device_rows() { return p; } };
                Outbound outbound_{asynchronous ? staged.data() : nullptr};
                float *R_ = r.data(), *bo_ = bo.data(), *inj2_ = inj.data(), *hand_out_ = hand.data();
                void* cs = nullptr;
                auto Rt = [&](int t) { return R_ + t * HC * N; };
                auto batch_timeline = [&] { return timeline ? 50 : 0; };
                auto handoff_one_on = [&] { return one; };
                auto stamp = [&](int64_t layer, int index, int group) {
                    check(layer == g.n_layers && group == 0);
                    if (prof_on_) trace.push_back(index == 0 ? 0 : 2);
                };
                trace.clear();
                auto record = [&]() -> bool {
                    @COPIES@
                    return false;
                };
                const bool early = le_ < g.n_layers;
                check(record() == early);
                const bool tagged = early && batch_rec_ && timeline && prof_on_;
                const size_t launches = early ? (one ? 1 : 3 * T) : 0;
                check(trace.size() == launches + (tagged ? 2 : 0));
                if (tagged) check(trace.front() == 0 && trace.back() == 2);
                check(std::count(trace.begin(), trace.end(), 1) == (int) launches);
                for (int row = 0; row < hrow0 + T + 1; ++row)
                    for (int col = 0; col < HB; ++col) {
                        float expected = -9;
                        const int t = row - hrow0;
                        if (early && t >= 0 && t < T)
                            expected = col < HC * N ? r[t * HC * N + col] : col < HC * N + N
                                     ? bo[t * N + col - HC * N] : inj[t * HC + col - HC * N - N];
                        const bool redirected = batch_rec_ && asynchronous;
                        check((redirected ? staged : hand)[row * HB + col] == expected);
                        check((redirected ? hand : staged)[row * HB + col] == -9);
                    }
            }
        }
        void test_span() {
            BatchTimes b;
            check(b.out_ticks == 0 && b.out_ms == -1 && std::string(b.out_clock) == "none");
            b.set_handoff(100, 200, 300, 500, 8);
            check(b.out_ticks == 100 && b.out_ms == 2);
            for (double ms : {0., -1., std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity()}) {
                b.set_handoff(100, 200, 300, 500, ms);
                check(b.out_ticks == 100 && b.out_ms == -1);
            }
            const uint64_t bad[][4] = {{0, 200, 300, 500}, {100, 0, 300, 500}, {100, 200, 0, 500},
                {100, 200, 300, 0}, {100, 20, 30, 500}, {100, 300, 200, 500}, {100, 200, 200, 500},
                {100, 200, 600, 500}, {500, 200, 300, 100}};
            for (const auto& s : bad) {
                b.set_handoff(100, 200, 300, 500, 8);
                b.set_handoff(s[0], s[1], s[2], s[3], 8);
                check(b.out_ticks == 0 && b.out_ms == -1);
            }
        }
        void test_reset() {
            unsigned long long stamps[72];
            for (bool tl : {false, true}) for (bool profile : {false, true}) for (bool final : {false, true}) {
                std::fill(std::begin(stamps), std::end(stamps), 250);
                reset_calls = 0;
                check(reset_span(tl, profile, final, stamps));
                const bool cleared = tl && profile && !final;
                check(reset_calls == (cleared ? 1 : 0));
                for (size_t i = 0; i < 72; ++i) check(stamps[i] == (cleared && i >= 66 && i < 70 ? 0u : 250u));
            }
            // Fresh calibration, missing out writes: a plausible old pair must not survive the per-replay reset.
            std::fill(std::begin(stamps), std::end(stamps), 250);
            check(reset_span(true, true, false, stamps));
            stamps[68] = 100; stamps[69] = 500;
            BatchTimes b;
            b.set_handoff(stamps[68], stamps[66], stamps[67], stamps[69], 8);
            check(b.out_ticks == 0 && b.out_ms == -1);
            reset_ok = false;
            check(!reset_span(true, true, false, stamps));
            reset_ok = true;
        }
        void test_readback() {
            for (bool prof_on_ : {false, true}) for (bool tl : {false, true}) for (bool final : {false, true})
            for (bool copied : {false, true}) for (bool tag0 : {false, true}) for (bool tag1 : {false, true})
            for (bool elapsed : {false, true}) {
                struct Geometry { int n_layers = 2; } g;
                auto* g_ = &g;
                constexpr int kProfPer = 33;
                std::vector<unsigned long long> device(70), prof_h_(70, 250);
                auto* prof_ = device.data();
                device[66] = 200; device[67] = 300; device[68] = 100; device[69] = 500;
                void* prof_ev_[2] = {nullptr, nullptr};
                bool prof_tag_ok_[2] = {tag0, tag1};
                auto batch_timeline = [&] { return tl ? 50 : 0; };
                auto last_stage = [&] { return final; };
                int accumulations = 0;
                auto accumulate_profile = [&](const unsigned long long*) { ++accumulations; };
                BatchTimes bt_cur_;
                copy_ok = copied; elapsed_ok = elapsed; elapsed_calls = 0;
                @READBACK@
                const bool readable = prof_on_ && tl && !final && copied;
                const bool calibrated = readable && tag0 && tag1 && elapsed;
                check(bt_cur_.out_ticks == (readable ? 100u : 0u));
                check(bt_cur_.out_ms == (calibrated ? 2 : -1));
                check(accumulations == (prof_on_ && copied ? 1 : 0));
                check(elapsed_calls == (readable && tag0 && tag1 ? 1 : 0));
            }
        }
        void test_report() {
            @ACCUM@
            const int n_pipe = 2;
            std::vector<TlAcc> tla(n_pipe);
            std::vector<int64_t> tl_ready(n_pipe);
            struct Stage {
                BatchTimes b;
                bool final = false;
                bool last_stage() const { return final; }
                const BatchTimes& batch_times() const { return b; }
            };
            std::vector<Stage> stages(n_pipe);
            stages[1].final = true;
            auto stage_ver = [&](int k) -> Stage& { return stages[k]; };
            int calibrations = 0;
            auto tl_calibrate = [&] { ++calibrations; };
            @DONE@
            @REPORT@
            for (Stage& stage : stages) {
                stage.b.set_handoff(100, 200, 300, 500, 8);
                stage.b.out_clock = "sub_group";
            }
            tl_done(0, 0); tl_done(1, 0); tl_report();
            stages[0].b = BatchTimes{};
            stages[0].b.out_scope = "device-pack";
            tl_done(0, 0); tl_done(1, 0); tl_report();
            check(calibrations == 2);
        }
        int main(int argc, char** argv) {
            try {
                check(argc == 2);
                const std::string mode = argv[1];
                if (mode == "copies") test_copies();
                if (mode == "span") test_span();
                if (mode == "reset") test_reset();
                if (mode == "report") test_report();
                if (mode == "readback") test_readback();
            } catch (const std::exception& error) {
                std::fprintf(stderr, "%s\\n", error.what());
                return 1;
            }
        }
    ''').replace("@TIMES@", times).replace("@VALIDATOR@", validator).replace("@COPIES@", copies).replace(
        "@RESET@", reset).replace("@ACCUM@", accum).replace("@DONE@", done).replace("@REPORT@", report).replace(
        "@READBACK@", readback)


class HandoffTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="batch-handoff-timeline-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not cls.compiler or not shutil.which(cls.compiler[0]):
            raise AssertionError("a host C++ compiler is required; these tests must not silently skip")
        cls.source = harness()
        cls.binary = cls.compile_source(cls.source, "original")

    @classmethod
    def compile_source(cls, source, name):
        path = Path(cls.tmp.name) / (name + ".cpp")
        binary = path.with_suffix("")
        path.write_text(source)
        result = subprocess.run(
            cls.compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(binary)],
            text=True, capture_output=True, check=False)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return binary

    def execute(self, mode, binary=None, success=True):
        result = subprocess.run([str(binary or self.binary), mode], text=True, capture_output=True, check=False)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, "causal control did not detect the mutation")
        return result

    def test_both_branches_and_opt_in_gates_preserve_exact_rows(self):
        self.execute("copies")

    def test_same_window_calibration_and_invalid_tags(self):
        self.execute("span")

    def test_per_replay_reset_missing_tags_and_failure(self):
        self.execute("reset")

    def test_readback_copy_and_per_replay_calibration_failure(self):
        self.execute("readback")

    def test_raw_ticks_scope_estimate_and_unknown_report(self):
        lines = self.execute("report").stderr.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("out-est 2.000/2.000", lines[0])
        self.assertIn("out-ticks(raw) 100.000/100.000", lines[0])
        self.assertIn("out-clock=sub_group out-duration=calibrated-estimate out-unknown=0", lines[0])
        self.assertIn("out-scope=mapped-host", lines[0])
        self.assertNotIn("out-", lines[1])
        self.assertIn("out-clock=none out-duration=UNKNOWN out-unknown=1", lines[2])
        self.assertIn("out-scope=device-pack", lines[2])
        self.assertNotIn("out-est ", lines[2])
        self.assertNotIn("out-ticks(raw)", lines[2])
        self.assertNotIn("out-", lines[3])

    def test_causal_controls_each_end_stamp_is_required(self):
        needle = "if (handoff_tl) stamp(g.n_layers, 1, 0);"
        self.assertEqual(self.source.count(needle), 2)
        for branch in range(2):
            pieces = self.source.split(needle)
            pieces[branch] += "/* removed end tag */"
            mutated = pieces[0] + ("" if branch == 0 else needle) + pieces[1] + (
                "" if branch == 1 else needle) + pieces[2]
            binary = self.compile_source(mutated, f"no-end-{branch}")
            self.execute("copies", binary, success=False)

    def test_causal_control_reset_is_required(self):
        old = "if (tl) {\n"
        self.assertEqual(self.source.count(old), 1)
        binary = self.compile_source(self.source.replace(old, "if (tl && false) {\n"), "no-reset")
        self.execute("reset", binary, success=False)

    def test_causal_control_failed_record_cannot_use_old_event(self):
        old = "!prof_tag_ok_[0] ||"
        self.assertEqual(self.source.count(old), 1)
        binary = self.compile_source(self.source.replace(old, "false ||"), "old-event")
        self.execute("readback", binary, success=False)

    def test_causal_control_stale_envelope_is_rejected(self):
        old = "out_start < start"
        self.assertEqual(self.source.count(old), 1)
        binary = self.compile_source(self.source.replace(old, "false"), "no-envelope")
        self.execute("span", binary, success=False)
