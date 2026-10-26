// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>

namespace strata::prefill {

// xn16 in the shared HC region aliases the routing weights. A fused launch cannot write it while other
// work-groups still combine experts; the ordinary two launches provide the required whole-queue boundary.
inline constexpr bool combine_hc_norm_eligible(bool enabled, bool hc_in_region, int half, int64_t layer, int64_t end,
                                     bool ple, bool steered, bool debug) {
    return enabled && !hc_in_region && half == 1 && layer + 1 < end &&
           !(layer + 1 == 1 && ple) && !steered && !debug;
}

}  // namespace strata::prefill
