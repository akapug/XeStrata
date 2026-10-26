// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/core/gpu.hpp"

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata::core {

// A batch group's existing hand-off rows, staged on the device before a copy-engine transfer to host USM.  The
// producer may reuse its compute scratch after packing, but neither row range may be reused until copy completion
// and the receiver's synchronous rows() acknowledgement.  No cross-context event or device pointer crosses StageLink.
class BatchOutbound {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr bool kEnabledDefault = false;
    static constexpr int kTimeoutSecondsDefault = 20;   // the batch commit's existing watchdog, per pending group
    static std::chrono::seconds timeout() {
        static const int seconds = [] {
            const char* e = std::getenv("STRATA_BATCH_OUTBOUND_TIMEOUT_S");
            const int n = e != nullptr ? std::atoi(e) : 0;
            return n > 0 ? n : kTimeoutSecondsDefault;
        }();
        return std::chrono::seconds(seconds);
    }
    static bool enabled() {
        static const bool on = [] {
            const char* e = std::getenv("STRATA_BATCH_OUTBOUND_ASYNC");
            return e != nullptr ? std::atoi(e) != 0 : kEnabledDefault;
        }();
        return on;
    }
    BatchOutbound() = default;
    BatchOutbound(const BatchOutbound&) = delete;
    BatchOutbound& operator=(const BatchOutbound&) = delete;
    ~BatchOutbound() {
        // Also drain submissions whose completion could not be recorded/queried.  A failed copy is never delivered.
        if (stream_) gpu::stream_sync(stream_);
        for (auto& t : transfers_) {
            if (t.ready) gpu::event_destroy(t.ready);
            if (t.done) gpu::event_destroy(t.done);
        }
        if (device_) gpu::free(device_);
        if (stream_) gpu::stream_destroy(stream_);
    }
    bool init(float* host, size_t rows, size_t stride) {
        if (stream_ || !host || rows == 0 || stride == 0) return false;
        host_ = host;
        stride_ = stride;
        transfers_.resize(rows);
        stream_ = gpu::stream_create_copy();
        if (!stream_ || !gpu::alloc_device(&device_, rows * stride * sizeof(float))) return false;
        for (auto& t : transfers_)
            if (!gpu::event_create(&t.ready) || !gpu::event_create(&t.done)) return false;
        return true;
    }
    float* device_rows() const { return device_; }
    bool writable(int base, int n) const {
        std::lock_guard<std::mutex> lock(mu_);
        return writable_locked(base, n);
    }
private:
    bool writable_locked(int base, int n) const {
        if (base < 0 || n <= 0 || (size_t) base >= transfers_.size() ||
            (size_t) n > transfers_.size() - (size_t) base) return false;
        for (size_t i = 0; i < transfers_.size(); ++i) {
            const auto& t = transfers_[i];
            if (t.n > 0 && i < (size_t) base + (size_t) n && (size_t) base < i + (size_t) t.n) return false;
        }
        return true;
    }
public:
    bool start(int base, int n, gpu::Stream producer) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!writable_locked(base, n)) return false;
        auto& t = transfers_[(size_t) base];
        t.n = n;   // even a partial submission owns both ranges until drain
        t.state = State::Failed;
        const size_t offset = (size_t) base * stride_;
        if (!gpu::event_record(t.ready, producer) ||
            !gpu::copy_async_after(host_ + offset, device_ + offset, (size_t) n * stride_ * sizeof(float),
                                   stream_, t.ready, t.done)) return false;
        t.state = State::Pending;
        t.since = Clock::now();
        return true;
    }
    // A successful submission is Pending, never delivery.  event_sync on the completed event also surfaces async
    // runtime errors (a query alone need not throw them).  A failed range stays owned through teardown.
    int poll(int base, Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mu_);
        if (base < 0 || (size_t) base >= transfers_.size()) return -1;
        auto& t = transfers_[(size_t) base];
        if (t.state == State::TimedOut) return -2;
        if (t.n == 0 || t.state == State::Failed) return -1;
        if (t.state == State::Ready) return 1;
        const auto s = gpu::event_status(t.done);
        if (s == gpu::Status::Pending) {
            if (now - t.since <= timeout()) return 0;
            t.state = State::TimedOut;
            return -2;   // no other group's progress may reset this group's watchdog
        }
        if (s == gpu::Status::Failed || !gpu::event_sync(t.done)) { t.state = State::Failed; return -1; }
        t.state = State::Ready;
        return 1;
    }
    bool wait(int base) {
        std::lock_guard<std::mutex> lock(mu_);
        if (base < 0 || (size_t) base >= transfers_.size()) return false;
        auto& t = transfers_[(size_t) base];
        if (t.state == State::TimedOut) return false;
        if (t.n == 0 || t.state == State::Failed || !gpu::event_sync(t.done)) {
            t.state = State::Failed;
            return false;
        }
        t.state = State::Ready;
        return true;
    }
    bool acknowledge(int base) {
        std::lock_guard<std::mutex> lock(mu_);
        if (base < 0 || (size_t) base >= transfers_.size()) return false;
        auto& t = transfers_[(size_t) base];
        if (t.state != State::Ready) return false;
        t.n = 0;
        t.state = State::Idle;
        return true;
    }
    int count(int base) const {
        std::lock_guard<std::mutex> lock(mu_);
        return base >= 0 && (size_t) base < transfers_.size() ? transfers_[(size_t) base].n : 0;
    }

private:
    enum class State { Idle, Pending, Ready, Failed, TimedOut };
    struct Transfer {
        gpu::Event* ready = nullptr;
        gpu::Event* done = nullptr;
        int n = 0;
        State state = State::Idle;
        Clock::time_point since;
    };
    mutable std::mutex mu_;   // CPU-split: the stage's poll thread submits while the scheduler acknowledges old groups
    gpu::Stream stream_ = nullptr;
    float* device_ = nullptr;
    float* host_ = nullptr;
    size_t stride_ = 0;
    std::vector<Transfer> transfers_;
};

}  // namespace strata::core
