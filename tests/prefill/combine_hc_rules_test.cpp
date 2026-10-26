// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// All eligibility combinations, including the shared-scratch routing-weight alias: no GPU or model.
#include "strata/prefill/combine_hc.hpp"

#include <cstdio>
#include <initializer_list>

using strata::prefill::combine_hc_norm_eligible;

int main() {
    int cases = 0, failed = 0;
    for (unsigned mask = 0; mask < 32; ++mask) for (int half = 0; half < 2; ++half)
        for (int layer = 0; layer < 4; ++layer) {
            const bool expected = half == 1 && layer < 2 && (mask == 1 || (mask == 5 && layer != 0));
            const bool actual = combine_hc_norm_eligible(mask & 1, mask & 2, half, layer, 3,
                                                         mask & 4, mask & 8, mask & 16);
            ++cases;
            if (actual != expected) {
                std::fprintf(stderr, "FAIL mask=%u half=%d layer=%d actual=%d expected=%d\n",
                             mask, half, layer, (int) actual, (int) expected);
                ++failed;
            }
        }
    // The final layer of a nonzero-start stage must not read the next stage's norm weights.
    for (int layer : {23, 24, 25}) {
        ++cases;
        if (combine_hc_norm_eligible(true, false, 1, layer, 25, false, false, false) != (layer == 23))
            ++failed;
    }
    static_assert(combine_hc_norm_eligible(true, false, 1, 1, 3, true, false, false));
    static_assert(!combine_hc_norm_eligible(true, true, 1, 1, 3, false, false, false));
    std::printf("combine_hc_rules_test: %d cases, %d failures\n", cases, failed);
    return failed ? 1 : 0;
}
