// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/param_table.cpp - the device-table store (param_table.hpp).
#include "param_table.hpp"

#include "strata/core/device.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "fused_gr_win.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::kernels::xe {

// The tables still passed BY VALUE, at kQsaSlotsMax rows (the pointer ones: KvAppendSlots, the resolve's Maps and
// RowRuns, QsaIndexerSlots).  The rest of a kernel's arguments need room too, so this is the least that must hold.
static_assert(sizeof(QsaAttnSlots) < kParamBytesFloor, "QsaAttnSlots passed by value is past the B70's kernel arguments");
static_assert(sizeof(QsaScoreSlots) < kParamBytesFloor, "QsaScoreSlots passed by value is past the B70's kernel arguments");
static_assert(sizeof(GdnSlotTable) < kParamBytesFloor, "GdnSlotTable passed by value is past the B70's kernel arguments");
static_assert(sizeof(grw::GrWin) < kParamBytesFloor, "GrWin passed by value is past the B70's kernel arguments");

namespace {

// One device's copies, and the queue that fills them (never recording: a capture is on the engine's queues).
struct Store {
    sycl::context ctx;
    sycl::device dev;
    sycl::queue fill;
    std::unordered_map<std::string, void*> tabs;
};

std::mutex g_mutex;
// Never destroyed: the copies outlive every graph, and no SYCL object is torn down after the runtime at exit.
std::vector<std::unique_ptr<Store>>& stores() {
    static auto* s = new std::vector<std::unique_ptr<Store>>();
    return *s;
}

std::map<std::string, std::size_t>& kernels() {
    static auto* k = new std::map<std::string, std::size_t>();
    return *k;
}

}  // namespace

const void* device_table_bytes(sycl::queue& q, const void* host, std::size_t bytes) {
    const std::lock_guard<std::mutex> lock(g_mutex);
    const sycl::context ctx = q.get_context();
    const sycl::device dev = q.get_device();
    auto& all = stores();
    const auto at = std::find_if(all.begin(), all.end(), [&](const auto& s) { return s->ctx == ctx && s->dev == dev; });
    Store& s = at != all.end() ? **at
                               : *all.emplace_back(new Store{ctx, dev, sycl::queue(ctx, dev, sycl::property::queue::in_order{}), {}});
    std::string key(static_cast<const char*>(host), bytes);
    const auto hit = s.tabs.find(key);
    if (hit != s.tabs.end()) return hit->second;
    void* d = sycl::malloc_device(bytes, dev, ctx);
    if (d == nullptr) throw core::DeviceError("device_table: no device memory for a " + std::to_string(bytes) + "-byte table");
    s.fill.memcpy(d, host, bytes).wait_and_throw();
    s.tabs.emplace(std::move(key), d);
    return d;
}

void note_kernel_bytes(const char* name, std::size_t bytes) {
    const std::lock_guard<std::mutex> lock(g_mutex);
    std::size_t& m = kernels()[name];
    m = std::max(m, bytes);
}

std::vector<std::pair<std::string, std::size_t>> kernel_arg_bytes() {
    const std::lock_guard<std::mutex> lock(g_mutex);
    return {kernels().begin(), kernels().end()};
}

std::size_t device_table_count() {
    const std::lock_guard<std::mutex> lock(g_mutex);
    std::size_t n = 0;
    for (const auto& s : stores()) n += s->tabs.size();
    return n;
}

}  // namespace strata::kernels::xe
