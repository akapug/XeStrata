// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace strata::prefill {

inline constexpr double kStreamCpuShareDefault = 0.0;

inline double stream_cpu_share_request(const char* value) {
    if (value == nullptr || *value == '\0') return kStreamCpuShareDefault;
    char* end = nullptr;
    const double share = std::strtod(value, &end);
    return end != value && *end == '\0' && std::isfinite(share) && share >= 0.0 && share <= 1.0
               ? share : kStreamCpuShareDefault;
}

inline double stream_cpu_share_setting() {
    static const double share = stream_cpu_share_request(std::getenv("STRATA_PREFILL_STREAM_CPU_SHARE"));
    return share;
}

inline size_t stream_cpu_take(size_t eligible, double share) {
    return std::isfinite(share) && share > 0.0 && share <= 1.0
               ? std::min(eligible, (size_t) std::floor((double) eligible * share)) : 0;
}

// Only a stable non-null source pointer can leave the GPU stream. Keep a fixed fraction
// of those pointers in expert-id order; null/transient/resident entries stay unassigned.
inline size_t stream_cpu_partition(std::vector<const uint8_t*>& blobs, double share) {
    const size_t eligible = (size_t) std::count_if(blobs.begin(), blobs.end(), [](const uint8_t* b) { return b != nullptr; });
    const size_t take = stream_cpu_take(eligible, share);
    size_t kept = 0;
    for (const uint8_t*& b : blobs) {
        if (b == nullptr) continue;
        if (kept < take) ++kept;
        else b = nullptr;
    }
    return kept;
}

// Exactly one split stage borrows the decode pool. Its run joins before decode resumes;
// earlier stages may still run other chunks, so they must never borrow the same pool.
inline constexpr bool stream_cpu_owner(bool enabled, bool split, bool last, bool no_pool) {
    return enabled && split && last && !no_pool;
}

// A large expert's rows are split into the pool's existing bounded multi-token jobs.
// The same partition is used for activation and result pointers, including the tail.
inline size_t stream_cpu_rows(size_t remaining, size_t max_rows) {
    return std::min(remaining, max_rows);
}

inline void stream_cpu_jobs(std::vector<kernels::cpu::ExpertJobMulti>& jobs, const uint8_t* blob,
                            size_t rows, float* out, size_t width) {
    for (size_t r0 = 0; r0 < rows;) {
        kernels::cpu::ExpertJobMulti j;
        j.blob = blob;
        j.nt = (int) stream_cpu_rows(rows - r0, kernels::cpu::MAXT);
        for (int r = 0; r < j.nt; ++r) j.out[r] = out + (r0 + (size_t) r) * width;
        jobs.push_back(j);
        r0 += (size_t) j.nt;
    }
}

}  // namespace strata::prefill
