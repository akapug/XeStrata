// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/native_router_body.hpp - one token of the native router (ggml-cuda's topk-moe at llama.cpp 3cf03257,
// 512 experts, top 10, softmax, normalized; MIT License, Copyright (c) 2023-2026 The ggml authors) for one 32-lane
// sub-group.  router.cpp's launch and the router GEMV's fused tail (bf16_gemv.cpp, STRATA_ROUTER_FUSE) both run this
// body, so a token's ids and weights are the same bits from either.
#pragma once

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace strata::kernels::xe {

// The argmax order: the larger probability wins, a tie keeps the lower expert index.  It is a total order, so the xor
// butterflies reach the same winner as CUDA's shuffle-down trees.
inline void router_better(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

inline void native_router_token(const sycl::sub_group& sg, int lane, const float* logits, int32_t* ids, float* weights) {
    float values[16];
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
    for (int i = 0; i < 16; ++i) maximum = sycl::max(maximum, values[i]);
    for (int mask = 16; mask; mask >>= 1) maximum = sycl::fmax(maximum, sycl::permute_group_by_xor(sg, maximum, mask));
    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        values[i] = sycl::exp(values[i] - maximum);
        sum += values[i];
    }
    for (int mask = 16; mask; mask >>= 1) sum += sycl::permute_group_by_xor(sg, sum, mask);
    const float reciprocal = 1.0f / sum;
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
        for (int mask = 16; mask; mask >>= 1)
            router_better(best, expert, sycl::permute_group_by_xor(sg, best, mask),
                          sycl::permute_group_by_xor(sg, expert, mask));
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    for (int mask = 16; mask; mask >>= 1) selected_sum += sycl::permute_group_by_xor(sg, selected_sum, mask);
    selected_sum = sycl::max(selected_sum, 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}

}  // namespace strata::kernels::xe
