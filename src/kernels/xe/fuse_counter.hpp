// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/fuse_counter.hpp - homelab: arrival counters for kernels whose last work-group runs the next step (the
// router GEMV's top-10, the hyper-connection read's lo sums), so the step takes no launch of its own.
//
// The pattern (CUDA's threadFenceReduction sample; cub's single-pass reductions): every work-group stores its outputs,
// passes a work-group barrier with device-scope fences, and one work-item adds 1 to the counter (acq_rel, device
// scope); the work-group that reads count - 1 knows every other one's outputs are visible to it after a second such
// barrier, runs the step, and puts the counter back to 0, so the next launch (or the next replay of a captured graph)
// starts from 0.  A queue's launches run in order, so the counters are a queue's own: two queues never share one.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels::xe {

constexpr int kFuseCounters = 64;
constexpr int kFuseRouter = 0;    ///< bf16_gemv.cpp's router top-10 tail
constexpr int kFuseHcLo = 16;     ///< fused_gr_win.cpp's lo tails, one a row group (up to 16)

/// `q`'s kFuseCounters counters in device memory, zero (allocated and zeroed on first use outside any graph capture:
/// a queue of its own does the fill).
uint32_t* fuse_counters(sycl::queue& q);

/// one work-group's arrival; true in every work-item of the last one to arrive (its view of the others' stores is then
/// complete).  `flag` is one int of local memory.  Every work-item of the work-group must call it.
template <int D>
inline bool fuse_arrive_last(const sycl::nd_item<D>& it, uint32_t* counter, uint32_t groups,
                             const sycl::local_accessor<int, 1>& flag) {
    sycl::group_barrier(it.get_group(), sycl::memory_scope::device);
    if (it.get_local_linear_id() == 0) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> c(*counter);
        const bool last = c.fetch_add(1u) == groups - 1;
        if (last) c.store(0u, sycl::memory_order::relaxed);
        flag[0] = last ? 1 : 0;
    }
    sycl::group_barrier(it.get_group(), sycl::memory_scope::device);
    const bool last = flag[0] != 0;
    if (last) sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
    return last;
}

}  // namespace strata::kernels::xe
