// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only consumers of the production opt-in dispatch policy, no SYCL or device.
#include "strata/kernels/dmmq_dispatch.hpp"

#include <cstdio>

namespace d = strata::kernels::dmmq;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    failures += !ok;
    std::printf("  %s: %s\n", what, ok ? "ok" : "FAIL");
}
}  // namespace

int main() {
    check(d::dispatch_min(0, nullptr) == 0 && d::dispatch_min(-1, "1") == 0,
          "the named row policy never enables default-off levers");
    check(d::dispatch_min(0, "invalid") == 0, "disabled paths do not depend on unused policy text");
    check(d::dispatch_min(1, nullptr) == 7 && d::dispatch_min(1, "") == 7,
          "an enabled path defaults to the provisional seven-row floor");
    check(d::dispatch_min(1, "1") == 1, "the floor is explicitly retunable for a different device or experiment");
    check(d::dispatch_min(12, "7") == 12 && d::dispatch_min(17, nullptr) == 17,
          "a stricter requested threshold is preserved, including disabling the entire window");
    for (int rows = 1; rows <= d::kMaxRows; ++rows) {
        const int threshold = d::dispatch_min(1, nullptr);
        check((rows >= threshold) == (rows >= 7), "production default preserves the established path below seven");
    }
    for (const char* text : {"0", "-1", "33", "7x", "99999999999999999999"}) {
        bool rejected = false;
        try { (void) d::dispatch_min(1, text); } catch (const std::exception&) { rejected = true; }
        check(rejected, "invalid enabled policy is rejected rather than silently changing the selected path");
    }
    std::printf("dmmq_dispatch_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
