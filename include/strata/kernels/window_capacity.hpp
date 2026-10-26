// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {

// Allocation capacity is not the default decode window. Arithmetic kernels keep
// their existing <=16-row paths; wider windows are explicit process settings.
inline constexpr int kVerifyDefaultT = 8;
inline constexpr int kVerifyMaxT = 32;
inline int verify_window_rows(const char* value) {
    if (value == nullptr || std::strcmp(value, "8") == 0) return kVerifyDefaultT;
    if (std::strcmp(value, "16") == 0) return 16;
    if (std::strcmp(value, "24") == 0) return 24;
    if (std::strcmp(value, "32") == 0) return kVerifyMaxT;
    throw std::invalid_argument("STRATA_BATCH_MAX_T must be 8, 16, 24 or 32");
}
inline int verify_max_t() {
    static const int rows = verify_window_rows(std::getenv("STRATA_BATCH_MAX_T"));
    return rows;
}

}  // namespace strata::kernels
