// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/program/prefill_export.hpp"

#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    using namespace strata::program;
    PrefillRequest r;
    std::string why;
    check(parse_prefill_request("PREFILL 11,0,11,29 /private/session with spaces.bin", r, why) &&
          !r.restore && r.ids == std::vector<int64_t>({11, 0, 11, 29}) &&
          r.path == "/private/session with spaces.bin", "typed producer preserves ordered repeated IDs, zero, final token and path");
    check(parse_prefill_request("RESTORE_PREFILL 0,2147483647 state.bin", r, why) && r.restore &&
          r.ids == std::vector<int64_t>({0, INT32_MAX}), "typed consumer accepts the full int32 token range");
    const auto before = r;
    const std::vector<std::string> invalid = {
        "GEN 0 1,2", "PREFILL", "PREFILL 1 state.bin", "PREFILL 1,2", "PREFILL  1,2 state.bin",
        "PREFILL 1, state.bin", "PREFILL ,2 state.bin", "PREFILL 1,,2 state.bin", "PREFILL -1,2 state.bin",
        "PREFILL +1,2 state.bin", "PREFILL 2147483648,2 state.bin", "PREFILL 1x,2 state.bin",
        "PREFILL 1,2 ", "PREFILL 1,2 state\n.bin", "PREFILL 1,2 state\r.bin",
        std::string("PREFILL 1,2 state\0.bin", 22), "RESTORE_PREFILL 1,2 "
    };
    for (const auto& line : invalid) {
        check(!parse_prefill_request(line, r, why) && !why.empty(), "malformed command refused by real owner parser");
        check(r.ids == before.ids && r.path == before.path && r.restore == before.restore,
              "parser refusal leaves destination request unchanged");
    }
    const PrefillCapabilities supported{true, false, false, true, 32, int64_t(INT32_MAX) + 1};
    check(admit_prefill_request(before, supported, why), "real request admission accepts supported unsplit owner");
    for (int unavailable : {0, 1, 2, 3, 4, 5}) {
        auto capability = supported;
        if (unavailable == 0) capability.mtp = false;
        if (unavailable == 1) capability.split = true;
        if (unavailable == 2) capability.batch = true;
        if (unavailable == 3) capability.prompt_cache = false;
        if (unavailable == 4) capability.prompt_limit = 1;
        if (unavailable == 5) capability.vocabulary = INT32_MAX;
        check(!admit_prefill_request(before, capability, why) && !why.empty(),
              "unsupported split/batch/state/token/context refused by owner pre-mutation gate");
    }
    PrefillBinding binding;
    const std::vector<int64_t> full_prompt{11, 29, 11, 0, 41, 17};
    binding.bind(full_prompt);
    check(binding.active() && !binding.discard_for(full_prompt, false) && binding.active(),
          "mounted prefill remains bound for exactly matching GEN, not an ordinary parked session");
    for (size_t token = 0; token < full_prompt.size(); ++token) {
        binding.bind(full_prompt);
        auto changed = full_prompt; changed[token] ^= 1;
        check(binding.discard_for(changed, false) && !binding.active(),
              "ordinary GEN with any changed prefix/final token requires owner to discard mounted reuse");
    }
    binding.bind(full_prompt);
    check(binding.discard_for(full_prompt, true), "a new export never borrows a mounted draft's last cell");
    binding.bind(full_prompt);
    check(binding.finish(true) && !binding.active(), "cancelled consume requires live/checkpoint discard, not downgrade or parking");
    binding.bind(full_prompt);
    check(!binding.finish(false) && !binding.active(), "successful final-token consume releases binding to ordinary generation");
    check(!binding.finish(true), "ordinary generation has no prefill binding to invalidate");
    RequestPublication gate;
    const uint64_t first = gate.begin();
    gate.cancel();
    bool invoked = false;
    check(!gate.publish(first, [&] { invoked = true; return true; }, why) && !invoked,
          "STOP before publication does not invoke persistence rename");
    const uint64_t second = gate.begin();
    check(!gate.publish(first, [&] { invoked = true; return true; }, why) && !invoked,
          "stale generation cannot publish even after STOP was reset");
    check(gate.publish(second, [&] { invoked = true; return true; }, why) && invoked,
          "current generation can publish after an earlier cancellation");
    gate.cancel();
    check(invoked && gate.stopped.load(), "STOP after committed publication does not retract success");

    const uint64_t concurrent = gate.begin();
    std::promise<void> entered, release, cancelling;
    auto released = release.get_future();
    bool published = false;
    std::thread writer([&] {
        std::string error;
        published = gate.publish(concurrent, [&] {
            entered.set_value();
            released.wait();
            return true;
        }, error);
    });
    entered.get_future().wait();
    std::thread stopper([&] { cancelling.set_value(); gate.cancel(); });
    cancelling.get_future().wait();
    release.set_value();
    writer.join(); stopper.join();
    check(published && gate.stopped.load(), "publication already in its critical section wins over concurrent STOP");
    check(!gate.publish(concurrent, [] { return true; }, why), "later publication in the cancelled generation refused");
    std::printf("prefill_export_test: %d checks passed\n", checks);
}
