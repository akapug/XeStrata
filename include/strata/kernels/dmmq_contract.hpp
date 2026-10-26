// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/window_capacity.hpp"

namespace strata::kernels::dmmq {

constexpr int kMaxRows = kVerifyMaxT;

// STRATA_DMMQ_NATIVE_Q8_1 changes only dense/shared decode operands, never prompt MMQ or HC.
constexpr bool kNativeQ81Default = false;
inline bool native_q8_1_requested(const char* value) {
    return value && *value ? std::strtol(value, nullptr, 10) != 0 : kNativeQ81Default;
}

inline bool native_q8_1_storage_ok(const void* native, const void* planar, int64_t cols, int64_t rows) {
    if (!native || !planar || reinterpret_cast<uintptr_t>(planar) % alignof(float) != 0 ||
        rows < 1 || rows > kMaxRows || cols <= 0 || cols % 32 != 0 ||
        cols > std::numeric_limits<int>::max() / rows) return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(native), b = reinterpret_cast<uintptr_t>(planar);
    const size_t blocks = (size_t) (rows * (cols / 32));
    // 36 bytes per native block; 32 codes and two FP32 planes per planar block. No in-place expansion.
    return a < b ? b - a >= blocks * 36 : a - b >= blocks * 40;
}

// The canonical native quantizer owns finite/clamp/rounding and FP16 storage. This is ONLY its layout bridge:
// native [half d, half raw_sum, int8 qs[32]] -> planar codes | FP32 d | FP32 d*sum(qs).
// Q4_K/Q5_K use the INTEGER-code correction, not native's rounded raw sum. Q5_0 stays centered in Dec<6>.
// Half is sycl::half on the device; the host contract consumer supplies IEEE binary16.
// The caller owns nonoverlapping storage and submits one work-item per 32-value block.
template<class Half>
inline void native_q8_1_planar_block(const void* native, void* planar, int64_t blocks, int64_t block, bool sums) {
    static_assert(sizeof(Half) == 2);
    const auto* src = static_cast<const int8_t*>(native) + block * 36;
    auto* codes = static_cast<int8_t*>(planar);
    auto* scales = reinterpret_cast<float*>(codes + blocks * 32);
    Half stored;
    std::memcpy(&stored, src, sizeof(stored));
    const float d = (float) stored;
    int sum = 0;
    for (int j = 0; j < 32; ++j) {
        const int8_t v = src[4 + j];
        codes[block * 32 + j] = v;
        sum += v;
    }
    scales[block] = d;
    if (sums) scales[blocks + block] = d * (float) sum;
}

// STRATA_DMMQ_LEAN (homelab densefuse; 0 = off, the default): bits.  kLeanKernel: iq_mmq_decode takes LeanKernel (the
// same blocks, slices and sum order in fewer messages a block: the same bits).  kLeanPair: the shared expert's gate and
// up products in one launch (iq_mmq_decode_pair; each its own launch's bits).  kLeanReuse: a batch window's GDN z
// product reads the int8 rows its qkv product quantized from the same input (a quantize launch fewer; the same
// bytes).
constexpr int kLeanDefault = 0;
constexpr int kLeanKernel = 1, kLeanPair = 2, kLeanReuse = 4;
inline int lean_bits(const char* value) {
    constexpr long kAll = kLeanKernel | kLeanPair | kLeanReuse;
    return value && *value ? (int) (std::strtol(value, nullptr, 10) & kAll) : kLeanDefault;
}

// The decoder walks complete weight blocks, not just its 32-value XMX steps.
// The row-split native layouts also require an even block count (their 4-byte alignment).
inline int block_cols(int type) {
    switch (type) {
        case 6: case 8: case 20: return 32;
        case 42: case kNativeQ8Rows: return 64;
        case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 21: case 22: case 23: case 29: return 256;
        case kNativeQ6KRows: return 512;
        default: return 0;   // Q2_K's duplicated activation format is not supported.
    }
}
inline bool shape_ok(int type, int64_t outputs, int64_t cols, int64_t rows) {
    const int block = block_cols(type);
    return rows >= 1 && rows <= kMaxRows && outputs > 0 && outputs % 16 == 0 && cols > 0 &&
           block > 0 && cols % block == 0;
}

inline int positive_int(const std::string& text, int max) {
    size_t used = 0;
    const int n = std::stoi(text, &used);
    if (used != text.size() || n < 1 || n > max) throw std::invalid_argument("invalid positive integer: " + text);
    return n;
}

inline std::vector<int> parse_rows(const std::string& text) {
    std::vector<int> rows;
    size_t begin = 0;
    do {
        const size_t end = text.find(',', begin);
        rows.push_back(positive_int(text.substr(begin, end - begin), kMaxRows));
        if (end == std::string::npos) return rows;
        begin = end + 1;
    } while (true);
}

// Reject empty and nonfinite oracles as well as nonfinite device outputs. std::max
// alone silently discards NaNs, which otherwise lets a broken reference pass.
struct Error {
    size_t count = 0;
    bool finite = true;
    double worst = 0, scale = 0;

    void add(double actual, double reference) {
        ++count;
        finite = finite && std::isfinite(actual) && std::isfinite(reference);
        worst = std::max(worst, std::fabs(actual - reference));
        scale = std::max(scale, std::fabs(reference));
    }
    double relative() const {
        return count && finite ? worst / std::max(scale, 1e-30) : std::numeric_limits<double>::infinity();
    }
    bool ok(double tolerance) const { return relative() < tolerance; }
};

inline bool checks_ok(int checks, int failures) { return checks > 0 && failures == 0; }

}  // namespace strata::kernels::dmmq
