// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/l2_prefetch.cpp - homelab: see l2_prefetch.hpp.
#include "strata/kernels/l2_prefetch.hpp"

#include "strata/core/runtime.hpp"

#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>

namespace strata::kernels {
namespace {
namespace sx = sycl::ext::oneapi::experimental;

constexpr size_t kLine = 64;   // bytes a prefetch hint covers
constexpr size_t kGroup = 256;
int g_form = -1;               // -1: not yet read from STRATA_HC_PREFETCH

}  // namespace

void hc_prefetch_set(int form) { g_form = form == 1 ? 1 : kHcPrefetchDefault; }
int hc_prefetch() {
    if (g_form < 0) {
        const char* v = std::getenv("STRATA_HC_PREFETCH");
        hc_prefetch_set(v != nullptr ? (int) std::strtol(v, nullptr, 10) : kHcPrefetchDefault);
    }
    return g_form;
}

void l2_prefetch(const void* a, size_t na, const void* b, size_t nb, void* stream) {
    const size_t la = (na + kLine - 1) / kLine, lb = b != nullptr ? (nb + kLine - 1) / kLine : 0;
    if (la + lb == 0) return;
    auto& q = core::Runtime::get().stream(stream);
    const auto* pa = static_cast<const char*>(a);
    const auto* pb = static_cast<const char*>(b);
    const size_t n = la + lb, groups = (n + kGroup - 1) / kGroup;
    q.parallel_for(sycl::nd_range<1>(groups * kGroup, kGroup), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_id(0);
        if (i >= n) return;
        const char* p = i < la ? pa + i * kLine : pb + (i - la) * kLine;
        sx::prefetch(const_cast<char*>(p), sx::properties{sx::prefetch_hint_L2});
    });
}

}  // namespace strata::kernels
