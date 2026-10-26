// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/conversation_park_queue.hpp"

#include <algorithm>
#include <chrono>

namespace strata::core {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
} // namespace

ConversationParkQueue::ConversationParkQueue(ConversationDiskOptions options, std::shared_ptr<ConversationFolder> folder,
                                             size_t max_jobs, size_t max_bytes)
    : options_(std::move(options)), folder_(std::move(folder)), max_jobs_(std::max<size_t>(1, max_jobs)),
      max_bytes_(max_bytes) {
    writer_ = std::thread([this] { loop(); });
}

ConversationParkQueue::~ConversationParkQueue() {
    drain();
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
    }
    cv_.notify_all();
    writer_.join();
}

uint64_t ConversationParkQueue::push(SavedConversation&& image, double& waited_ms) {
    const auto t0 = Clock::now();
    const size_t n = image.bytes();
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [&] { return jobs_.empty() || (jobs_.size() < max_jobs_ && bytes_ + n <= max_bytes_); });
    waited_ms = ms_since(t0);
    auto j = std::make_unique<Job>();
    j->id = next_id_++;
    j->image = std::make_unique<SavedConversation>(std::move(image));
    j->bytes = n;
    j->queued = Clock::now();
    const uint64_t id = j->id;
    bytes_ += n;
    jobs_.push_back(std::move(j));
    cv_.notify_all();
    return id;
}

void ConversationParkQueue::loop() {
    conversation_io_widen_thread();
    std::unique_lock<std::mutex> lock(m_);
    for (;;) {
        cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
        if (jobs_.empty()) return;
        Job& j = *jobs_.front();
        j.writing = true;
        Done d;
        d.id = j.id;
        d.tokens = j.image->live.ids.size();
        d.image_bytes = j.bytes;
        d.queued_ms = ms_since(j.queued);
        lock.unlock();
        // the image is only read here, and only read by best() and the claimer's wait: no lock while it is written
        const auto t0 = Clock::now();
        d.ok = ConversationDisk::write_file(options_, *folder_, *j.image, d.file, d.error, &j.cancel);
        d.write_ms = ms_since(t0);
        lock.lock();
        if (j.claimed) {
            handed_.emplace(std::move(*j.image));
            handed_id_ = j.id;
        }
        bytes_ -= j.bytes;
        jobs_.pop_front();   // frees the image unless it was handed back
        if (d.ok || d.error != "cancelled") done_.push_back(std::move(d));
        cv_.notify_all();
    }
}

std::optional<SavedConversation> ConversationParkQueue::claim(uint64_t id) {
    std::unique_lock<std::mutex> lock(m_);
    for (auto it = jobs_.begin(); it != jobs_.end(); ++it) {
        Job& j = **it;
        if (j.id != id) continue;
        if (!j.writing) {   // still waiting: it leaves the queue unwritten
            std::optional<SavedConversation> out(std::move(*j.image));
            bytes_ -= j.bytes;
            jobs_.erase(it);
            cv_.notify_all();
            return out;
        }
        j.claimed = true;
        j.cancel = true;
        cv_.wait(lock, [&] { return handed_id_ == id && handed_.has_value(); });
        std::optional<SavedConversation> out(std::move(handed_));
        handed_.reset();
        handed_id_ = 0;
        return out;
    }
    return std::nullopt;
}

std::vector<ConversationParkQueue::Done> ConversationParkQueue::done() {
    std::lock_guard<std::mutex> lock(m_);
    std::vector<Done> out;
    out.swap(done_);
    return out;
}

void ConversationParkQueue::drain() {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [&] { return jobs_.empty(); });
}

size_t ConversationParkQueue::jobs() const {
    std::lock_guard<std::mutex> lock(m_);
    return jobs_.size();
}

size_t ConversationParkQueue::bytes() const {
    std::lock_guard<std::mutex> lock(m_);
    return bytes_;
}

} // namespace strata::core
