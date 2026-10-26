// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/fuse_counter.cpp - homelab: the arrival counters of fuse_counter.hpp, a queue's own.
#include "fuse_counter.hpp"

#include "strata/core/runtime.hpp"

#include <map>
#include <mutex>

namespace strata::kernels::xe {

uint32_t* fuse_counters(sycl::queue& q) {
    static std::mutex mutex;
    static std::map<const sycl::queue*, uint32_t*> counters;   // kept until the process exits, as the queues are
    std::lock_guard<std::mutex> lock(mutex);
    uint32_t*& p = counters[&q];
    if (p == nullptr) {
        auto* d = sycl::malloc_device<uint32_t>(kFuseCounters, q.get_device(), q.get_context());
        if (d == nullptr) throw core::DeviceError("fuse_counters: device allocation failed");
        sycl::queue fill(q.get_context(), q.get_device());   // not the queue: it may be recording a graph
        fill.memset(d, 0, sizeof(uint32_t) * kFuseCounters).wait();
        p = d;
    }
    return p;
}

}  // namespace strata::kernels::xe
