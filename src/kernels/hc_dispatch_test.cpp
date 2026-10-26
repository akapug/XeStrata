// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Focused causal controls for the exact HC selector used by production and the roll-up. No device or SYCL.
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
    using P = d::HcPath;
    const int floor = d::dispatch_min(1, nullptr);
    check(floor == d::kDispatchMinRowsDefault, "HC opt-in uses the shared named floor");
    for (int rows = 1; rows <= d::kMaxRows; ++rows) {
        check(d::shape_ok(strata::kernels::kNativeQ8Rows, 320, 10240, rows) &&
              d::shape_ok(strata::kernels::kNativeQ8Rows, 10240, 320, rows),
              "both real HC row-split projection geometries satisfy the production decode contract");
        check(d::hc_path(rows, 0, 0, true, true, true, true) == P::fused,
              "capable hardware and valid Q8 weights never enable an absent HC opt-in");
        check(d::hc_path(rows, 0, floor, true, true, true, true) == (rows >= floor ? P::decode : P::fused),
              "changing only the HC opt-in selects decode exactly at its effective floor");
        check(d::hc_path(rows, 0, floor, false, true, true, true) == P::fused,
              "BF16 weights never enter the int8 decoder");
        check(d::hc_path(rows, 0, floor, true, true, true, false) == P::fused,
              "unsupported decode falls back to the original fused read without a separate MMQ opt-in");
        check(d::hc_path(rows, 1, 0, true, true, true, true) == P::mmq &&
              d::hc_path(rows, 1, 0, false, true, true, true) == P::mmq,
              "the established Q8 and BF16 MMQ opt-in is unchanged");
        check(d::hc_path(rows, 1, floor, true, true, true, false) == P::mmq,
              "unsupported decode may use independently enabled established MMQ");
        check(d::hc_path(rows, 1, floor, true, false, true, true) == P::fused,
              "missing or misaligned weights and scratch refuse both matrix paths");
    }
    check(d::hc_path(1, 0, d::dispatch_min(1, "1"), true, true, true, true) == P::decode,
          "an explicit floor of one permits a focused same-binary low-row experiment");
    check(d::hc_path(7, 1, d::dispatch_min(8, nullptr), true, true, true, true) == P::mmq &&
          d::hc_path(8, 1, d::dispatch_min(8, nullptr), true, true, true, true) == P::decode,
          "a stricter HC row threshold keeps MMQ below it and decode at it");
    for (int rows : {-1, 0, d::kMaxRows + 1})
        check(d::hc_path(rows, 0, floor, true, true, true, true) == P::fused,
              "the decode-only opt-in cannot exceed the actual supported row window");
    check(d::hc_path(d::kMaxRows + 1, 1, floor, true, true, true, true) == P::mmq,
          "past the decode window the independent established MMQ path remains available");
    check(d::hc_path(16, 1, 0, true, true, false, false) == P::fused,
          "a smaller device with no matrix engine keeps the original fused path");
    check(!d::shape_ok(strata::kernels::kNativeQ8Rows, 320, 10208, 7) &&
          !d::shape_ok(strata::kernels::kNativeQ8Rows, 10240, 288, 7),
          "odd native block counts in either HC projection are not decode-compatible");
    std::printf("hc_dispatch_test: %d checks, %d failures\n", checks, failures);
    return !d::checks_ok(checks, failures);
}
