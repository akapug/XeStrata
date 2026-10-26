// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/slot_fill.cpp - see include/strata/core/slot_fill.hpp.
#include "strata/core/slot_fill.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>

namespace strata::core {

namespace {

// a whole decimal in [lo, hi], else false
bool parse_int(const char* v, int lo, int hi, int& out) {
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || x < lo || x > hi) return false;
    out = (int) x;
    return true;
}

}  // namespace

bool parse_slot_fill_settings(const char* on, const char* buffers, const char* threads, SlotFillSettings& s,
                              std::string& why) {
    SlotFillSettings x;
    if (on != nullptr && on[0] != '\0') {
        if (std::string(on) != "0" && std::string(on) != "1") {
            why = std::string("STRATA_REFILL_STAGED=") + on + " is not 0 or 1";
            s = SlotFillSettings{};
            return false;
        }
        x.on = on[0] == '1';
    }
    if (buffers != nullptr && buffers[0] != '\0' && !parse_int(buffers, 2, SlotFillSettings::kMax, x.buffers)) {
        why = std::string("STRATA_REFILL_BUFFERS=") + buffers + " is not 2.." + std::to_string(SlotFillSettings::kMax);
        s = SlotFillSettings{};
        return false;
    }
    if (threads != nullptr && threads[0] != '\0' && !parse_int(threads, 1, SlotFillSettings::kMax, x.threads)) {
        why = std::string("STRATA_REFILL_THREADS=") + threads + " is not 1.." + std::to_string(SlotFillSettings::kMax);
        s = SlotFillSettings{};
        return false;
    }
    s = x;
    return true;
}

SlotFiller::~SlotFiller() {
    if (device_ < 0) return;
    const OnDevice on(device_);
    if (copy_ != nullptr) {
        gpu::stream_sync(copy_);
        gpu::stream_destroy(copy_);
    }
    for (gpu::Event* e : done_) gpu::event_destroy(e);
    for (size_t i = 0; i < buf_.size(); ++i)
        if (pinned_[i]) gpu::free(buf_[i]);
}

bool SlotFiller::init(int buffers, std::string& err) {
    const size_t bytes = (size_t) strata::kernels::cpu::expert_layout().max_blob;
    device_ = current_device();
    copy_ = gpu::stream_create_copy();
    if (copy_ == nullptr) {
        err = std::string("slot fill: no copy queue: ") + gpu::last_error();
        return false;
    }
    buf_.assign((size_t) buffers, nullptr);
    pinned_.assign((size_t) buffers, 0);
    pageable_.resize((size_t) buffers);
    done_.assign((size_t) buffers, nullptr);
    for (size_t i = 0; i < buf_.size(); ++i) {
        pinned_[i] = (char) gpu::alloc_host(&buf_[i], bytes, "refill-bounce");
        if (pinned_[i]) {
            pinned_bytes_ += bytes;
        } else {   // pageable: still correct, the copy only waits for the runtime's staging
            pageable_[i].resize(bytes);
            buf_[i] = pageable_[i].data();
        }
        if (!gpu::event_create(&done_[i])) {
            err = std::string("slot fill: event: ") + gpu::last_error();
            return false;
        }
    }
    return true;
}

bool SlotFiller::fill(ExpertSource& src, ExpertCache& cache, const std::vector<Fill>& fills, int buffers, int threads,
                      std::string& err) {
    if (fills.empty()) return true;
    if (device_ < 0 && !init(std::clamp(buffers, 2, SlotFillSettings::kMax), err)) return false;
    if (current_device() != device_) {
        err = "slot fill: called on device " + std::to_string(current_device()) + ", made on " + std::to_string(device_);
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const size_t n = fills.size(), K = buf_.size();
    const int dev = device_;
    // job j: its blob into buffer j % K, once job j - K's DMA from that buffer is queued (`issued` > j - K) and done
    std::unique_ptr<std::atomic<uint8_t>[]> ready(new std::atomic<uint8_t>[n]);
    for (size_t j = 0; j < n; ++j) ready[j].store(0, std::memory_order_relaxed);
    std::atomic<size_t> next{0}, issued{0};
    std::atomic<bool> failed{false}, stop{false};
    auto work = [&] {
        const OnDevice on(dev);
        for (;;) {
            const size_t j = next.fetch_add(1, std::memory_order_acq_rel);
            if (j >= n || stop.load(std::memory_order_acquire)) return;
            const size_t b = j % K;
            if (j >= K) {
                for (size_t v = issued.load(std::memory_order_acquire); v <= j - K;
                     v = issued.load(std::memory_order_acquire)) {
                    if (stop.load(std::memory_order_acquire)) return;
                    issued.wait(v, std::memory_order_acquire);
                }
                if (stop.load(std::memory_order_acquire)) return;
                if (!gpu::event_sync(done_[b])) failed.store(true, std::memory_order_release);
            }
            const Fill& f = fills[j];
            if (!failed.load(std::memory_order_acquire) && !src.copy_blob(f.layer, f.expert, buf_[b]))
                failed.store(true, std::memory_order_release);
            ready[j].store(1, std::memory_order_release);
            ready[j].notify_one();
        }
    };
    const int nt = (int) std::min<size_t>((size_t) std::clamp(threads, 1, SlotFillSettings::kMax), n);
    std::vector<std::thread> pool;
    pool.reserve((size_t) nt);
    for (int t = 0; t < nt; ++t) pool.emplace_back(work);
    bool ok = true;
    for (size_t j = 0; j < n; ++j) {
        ready[j].wait(0, std::memory_order_acquire);
        if (failed.load(std::memory_order_acquire)) {
            err = "slot fill: the source could not copy expert (" + std::to_string(fills[j].layer) + ", " +
                  std::to_string(fills[j].expert) + ") or a buffer's DMA failed: " + gpu::last_error();
            ok = false;
            break;
        }
        const Fill& f = fills[j];
        const size_t b = j % K;
        if (!cache.fill_slot(f.slot, buf_[b], copy_, err, (int64_t) lay.blob_bytes(f.layer)) ||
            !gpu::event_record(done_[b], copy_)) {
            if (err.empty()) err = std::string("slot fill: event record: ") + gpu::last_error();
            ok = false;
            break;
        }
        issued.store(j + 1, std::memory_order_release);
        issued.notify_all();
    }
    if (!ok) stop.store(true, std::memory_order_release);
    issued.store(std::numeric_limits<size_t>::max(), std::memory_order_release);   // no worker waits past here
    issued.notify_all();
    for (std::thread& t : pool) t.join();
    if (!gpu::stream_sync(copy_) && ok) {
        err = std::string("slot fill: the copy queue failed: ") + gpu::last_error();
        ok = false;
    }
    return ok;
}

}  // namespace strata::core
