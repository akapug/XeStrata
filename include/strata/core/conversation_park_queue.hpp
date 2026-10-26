// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// homelab background park (STRATA_SLOT_PARK_ASYNC): the serve thread captures a slot's conversation into host memory
// (the only part that touches the GPU) and hands the image here; a writer thread of its own serialises, checksums and
// writes it (ConversationDisk::write_file, through its own folder connection), and the serve thread indexes each
// finished file (done() -> ConversationDisk::adopt).
//
// - A bounded queue: at most `max_jobs` images (the one being written counts) and `max_bytes` of them, the writer
//   connection's staging buffers (ConversationFolder::staging_bytes) included.  push() waits
//   while it is full; an image alone over the byte bound is still taken when the queue is empty.
// - The conversation's next turn may arrive while its image waits or is being written.  best() matches a prompt against
//   the queued images as ConversationDisk::best does against files, and claim() hands the image itself back: a queued
//   one leaves the queue, one being written is cancelled and waited for.  The restore then uses the very bytes it would
//   have read back, never a partial file (a write is committed under its final name only once complete).
// - A failed write is reported by done() and the image freed: that conversation's next turn finds no newer file and
//   reads its history again.
#pragma once

#include "strata/core/conversation_disk.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class ConversationParkQueue {
public:
    struct Done {
        uint64_t id = 0;
        bool ok = false;
        ConversationDisk::Written file;   // ok: to adopt
        std::string error;
        size_t tokens = 0, image_bytes = 0;
        double queued_ms = 0, write_ms = 0;
    };
    struct Match {
        uint64_t id = 0;
        int64_t tokens = 0;
        bool live = false;      // the whole conversation
        bool deepest = false;   // its deepest checkpoint (the same chat, the last reply rendered again)
    };

    ConversationParkQueue(ConversationDiskOptions options, std::shared_ptr<ConversationFolder> folder, size_t max_jobs,
                          size_t max_bytes);
    /// Finishes every queued write, then stops the writer.
    ~ConversationParkQueue();
    ConversationParkQueue(const ConversationParkQueue&) = delete;
    ConversationParkQueue& operator=(const ConversationParkQueue&) = delete;

    /// Queues `image` for writing; waits while the queue is full.  Returns the job's id; `waited_ms` the wait.
    uint64_t push(SavedConversation&& image, double& waited_ms);

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::lock_guard<std::mutex> lock(m_);
        Match best;
        for (const auto& j : jobs_) {   // the newest wins a tie
            if (!j->image || j->image->cvec != cvec) continue;
            const SavedConversation& s = *j->image;
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : s.checkpoints)
                if (deepest == nullptr || c.ids.size() > deepest->ids.size()) deepest = &c;
            const int64_t n = conversation_prefix(s.live, prompt, images);
            if (n > 0 && n >= best.tokens) best = {j->id, n, true, false};
            for (const auto& c : s.checkpoints) {
                const int64_t k = conversation_prefix(c, prompt, images);
                if (k > 0 && k > best.tokens) best = {j->id, k, false, &c == deepest};
            }
        }
        return best;
    }
    /// Job `id`'s image, taken back: a queued job is dropped, one being written is cancelled and waited for.  nullopt
    /// when it has already finished (its file, if written, comes from done()).
    std::optional<SavedConversation> claim(uint64_t id);
    /// The writes finished since the last call, oldest first.
    std::vector<Done> done();
    /// Waits until every queued write has finished.
    void drain();
    size_t jobs() const;
    size_t bytes() const;

private:
    struct Job {
        uint64_t id = 0;
        std::unique_ptr<SavedConversation> image;
        size_t bytes = 0;
        std::atomic<bool> cancel{false};
        bool claimed = false, writing = false;
        std::chrono::steady_clock::time_point queued;
    };
    void loop();
    ConversationDiskOptions options_;
    std::shared_ptr<ConversationFolder> folder_;
    size_t max_jobs_, max_bytes_;
    mutable std::mutex m_;
    std::condition_variable cv_;   // a job queued, finished or claimed
    std::deque<std::unique_ptr<Job>> jobs_;   // the first one is being written
    std::vector<Done> done_;
    std::optional<SavedConversation> handed_;   // a claimed image the writer gives back
    uint64_t handed_id_ = 0, next_id_ = 1;
    size_t bytes_ = 0;
    bool stop_ = false;
    std::thread writer_;
};

} // namespace strata::core
