// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only regression consumers of the production decode entrypoint and parity contracts.
#include "strata/kernels/dmmq_contract.hpp"

#include <cstdio>
#include <limits>

namespace d = strata::kernels::dmmq;
namespace k = strata::kernels;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    failures += !ok;
    std::printf("  %s: %s\n", what, ok ? "ok" : "FAIL");
}
template<class F> bool rejects(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
}  // namespace

int main() {
    check(k::verify_window_rows(nullptr) == 8, "default window remains eight rows");
    for (const char* text : {"8", "16", "24", "32"})
        check(k::verify_window_rows(text) == std::atoi(text), "explicit supported window selected");
    for (const char* text : {"", "0", "1", "17", "33", "32x", " 32"})
        check(rejects([&] { (void) k::verify_window_rows(text); }), "invalid window rejected before allocation");
    check(d::kMaxRows == k::kVerifyMaxT, "decode operands and window share the same allocation capacity");
    check(d::parse_rows("17,24,31,32") == std::vector<int>({17, 24, 31, 32}), "wide decode rows preserved");
    for (int type : {11, 12, 13, 14, 16, 17, 18, 21, 22, 23, 29}) {
        check(!d::shape_ok(type, 16, 32, 1), "K-quant/IQ cannot read a partial 256-value block");
        check(d::shape_ok(type, 2560, 2560, 16), "complete 256-value blocks accepted");
    }
    for (int type : {6, 8, 20}) check(d::shape_ok(type, 16, 32, 1), "32-value block accepted");
    check(!d::shape_ok(42, 16, 32, 1) && d::shape_ok(42, 16, 64, 1), "Q2_0 requires 64 values");
    check(!d::shape_ok(k::kNativeQ6KRows, 16, 256, 1) && d::shape_ok(k::kNativeQ6KRows, 16, 512, 1),
          "row-split Q6_K requires aligned pairs of blocks");
    check(!d::shape_ok(k::kNativeQ8Rows, 16, 32, 1) && d::shape_ok(k::kNativeQ8Rows, 16, 64, 1),
          "row-split Q8_0 requires aligned pairs of blocks");
    for (int rows : {0, -1, d::kMaxRows + 1}) check(!d::shape_ok(8, 16, 32, rows), "row count outside allocated window rejected");
    for (int rows = 1; rows <= d::kMaxRows; ++rows) check(d::shape_ok(8, 16, 32, rows), "all window rows accepted");
    check(!d::shape_ok(8, 0, 32, 1) && !d::shape_ok(8, 15, 32, 1), "incomplete output tile rejected");
    check(!d::shape_ok(8, 16, 0, 1) && !d::shape_ok(8, 16, -32, 1), "nonpositive K rejected");
    for (int type : {-1, 0, 10, 43}) check(!d::shape_ok(type, 16, 256, 1), "unsupported/duplicated type rejected");
    check(d::parse_rows("1,7,16") == std::vector<int>({1, 7, 16}), "explicit row list preserved");
    for (const char* text : {"", "0", "-1", "33", "1,", ",1", "1,,7", "1x", "1,33", "99999999999999999999"})
        check(rejects([&] { (void) d::parse_rows(text); }), "invalid/empty/out-of-range row list rejected before device access");
    for (const char* text : {"", "0", "-1", "25x", "99999999999999999999"})
        check(rejects([&] { (void) d::positive_int(text, std::numeric_limits<int>::max()); }), "invalid timing repetition count rejected");
    check(d::positive_int("25", std::numeric_limits<int>::max()) == 25, "valid repetition count accepted");
    check(!d::Error{}.ok(1e-4), "empty oracle fails");
    check(!d::checks_ok(0, 0) && !d::checks_ok(1, 1) && d::checks_ok(1, 0), "zero checked cells cannot claim parity PASS");
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        d::Error a, b;
        a.add(bad, 1);
        b.add(1, bad);
        check(!a.ok(1e-4) && !b.ok(1e-4), "nonfinite result or oracle fails (not discarded by std::max)");
    }
    d::Error exact, near, outside, whole;
    exact.add(0, 0);
    near.add(1.00001, 1);
    outside.add(1.001, 1);
    whole.add(1.005, 1);
    check(exact.ok(1e-4) && near.ok(1e-4) && !outside.ok(1e-4), "dense FP64 relative bound enforced");
    check(whole.ok(1e-2) && !whole.ok(1e-4), "shared whole bound remains distinct");
    std::printf("dmmq_contract_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
