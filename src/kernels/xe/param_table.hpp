// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/param_table.hpp - a kernel's per-row table in device memory, handed to the kernel by pointer.
//
// A batch window's `_slots` kernels take a table with one entry a row.  By value, a table is a kernel argument, and the
// device caps the bytes of a kernel's arguments (max_parameter_size: 2048 on the Arc B70).  At 32 rows KvAppendSlots
// (2304 bytes) and the resolve's tables (Maps 2304, RowRuns 2840) are past it, so those kernels take a pointer to a
// device copy of the table instead.
//
// The copy is keyed by the table's bytes: the same table (the same rows' slots, whose buffers keep their addresses
// for life) always gets the same device copy, and a copy is never written again or freed.  So a captured graph that
// baked a copy's pointer replays it as long as it lives, and an eager launch reads exactly the bytes it was given.  A
// new table is copied once, on a private in-order queue that never records a graph, and the copy is waited for before
// the kernel is submitted: outside any capture, and complete before any launch (eager or replayed) can read it.
//
// The store grows by one copy per distinct table (a few KiB each) and keeps them until the process ends: as many as the
// distinct row sets the windows have run, the same keys the captured graphs are cached under.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::kernels::xe {

/// The smallest max_parameter_size of a device the engine runs on (the Arc B70: 2048 bytes, measured by
/// qsa_slots_parity --parameter-probe).  Every table still passed BY VALUE stays under it at kQsaSlotsMax rows, checked
/// at compile time (param_table.cpp, qsa_slots_parity.cpp) so a cap raise cannot pass a table the B70 refuses.
inline constexpr std::size_t kParamBytesFloor = 2048;

/// Notes a kernel's argument bytes (its functor's size) under `name`, once per kernel type: kernel_arg_bytes lists them.
void note_kernel_bytes(const char* name, std::size_t bytes);
/// Every noted kernel and the largest argument bytes seen under its name, in name order.
std::vector<std::pair<std::string, std::size_t>> kernel_arg_bytes();

/// A kernel functor whose captures (its kernel arguments: every captured table, pointer and scalar) stay under
/// kParamBytesFloor, checked at compile time; wrap each kernel whose captures scale with the window's rows.
template <class F> F&& fits(const char* name, F&& kernel) {
    static_assert(sizeof(std::remove_reference_t<F>) < kParamBytesFloor,
                  "a kernel's captures are past the B70's 2048 bytes of kernel arguments");
    static const bool noted = (note_kernel_bytes(name, sizeof(std::remove_reference_t<F>)), true);
    (void) noted;
    return std::forward<F>(kernel);
}

/// The device copy of `bytes` bytes at `host` on `q`'s device (made on the first call with those bytes).
const void* device_table_bytes(sycl::queue& q, const void* host, std::size_t bytes);
/// Distinct tables the store holds, over every device.
std::size_t device_table_count();

/// The device copy of `host`, for a kernel to take by pointer.  The table must have no padding: its bytes are its key.
template <class T> const T* device_table(sycl::queue& q, const T& host) {
    static_assert(std::is_trivially_copyable_v<T> && std::has_unique_object_representations_v<T>,
                  "a device table's bytes are its key: no padding, no indirection");
    return static_cast<const T*>(device_table_bytes(q, &host, sizeof(T)));
}

}  // namespace strata::kernels::xe
