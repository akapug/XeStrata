// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/kernels/dmmq_contract.hpp"

namespace strata::kernels::dmmq {

// The opt-in decode paths keep GEMV/the established MMQ below this row count.
// STRATA_DMMQ_MIN_ROWS may retune it; it never enables either default-off lever.
// A provisional seven-row floor was requested for the dense/shared decode shapes
// after the original dense-only card timings; shared-chain crossover is unverified.
constexpr int kDispatchMinRowsDefault = 7;
inline int dispatch_min(int requested, const char* policy) {
    if (requested <= 0) return 0;
    const int floor = policy && *policy ? positive_int(policy, kMaxRows) : kDispatchMinRowsDefault;
    return std::max(requested, floor);
}

// Both HC consumers use this selector; dense/shared switches cannot select HC.
// Capability and storage facts come from the real HC kernel boundary.
enum class HcPath { fused, mmq, decode };
inline HcPath hc_path(int rows, int mmq_min, int decode_min, bool q8, bool weights,
                      bool matrix_engine, bool decode_engine) {
    if (rows < 1 || !weights) return HcPath::fused;
    if (q8 && decode_min > 0 && rows >= decode_min && rows <= kMaxRows && decode_engine)
        return HcPath::decode;
    if (mmq_min > 0 && rows >= mmq_min && matrix_engine) return HcPath::mmq;
    return HcPath::fused;
}

}  // namespace strata::kernels::dmmq
