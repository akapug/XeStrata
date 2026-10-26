// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// homelab restore prefetch (STRATA_SLOT_PREFETCH): a parked conversation's file is read into host memory on a thread of
// its own, through its own folder connection, before (or while) its next turn is admitted, so the serve thread keeps
// only the part of a restore that touches the GPU.
//
// - hint(): serve/server.py sends `PREFETCH [cvec=0|1] <id,id,...>` when a request starts waiting.  The resolver thread
//   parses it, skips a conversation whose image still waits in (or is being written by) the background park (the
//   request path takes that one back whole: a file of it would be stale), and asks `locate` for the file the next turn
//   would read.  A hint over the count or byte bound is dropped; so is one that is not taken within the timeout.
// - load(): the serve thread's own request for a file it is about to restore (an admission it holds back while the
//   other slots decode): ahead of every hint, past the bounds (the RAM admission still applies).
// - take(): the request path's restore asks for the file it chose, by name AND index serial: a job for an older write
//   of that name is dropped, never used.  A job still reading is waited for; a failed or missing one leaves the
//   ordinary read on the serve thread.
// - The loader reads with ConversationDisk::read_file: the same bytes, checksum and checks as ConversationDisk::load,
//   so the image restored is the one the ordinary path would have read.
#pragma once

#include "strata/core/conversation_disk.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class ConversationPrefetch {
public:
    struct Hint;
    /// `[key=value ...] <id,id,...>` (a PREFETCH line after its word): the token ids and cvec= (1 when absent).  A hint
    /// with pictures (img_keys=) is refused: a file's pictures cannot be matched without them.
    static bool parse(const std::string& text, Hint& out, std::string& why);
    using Target = ConversationDisk::Target;
    struct Hint {
        std::vector<int32_t> ids;
        bool cvec = true;   // the request's cvec= key (1 when absent)
    };
    struct Options {
        ConversationDiskOptions disk;
        std::shared_ptr<ConversationFolder> folder;    // the loader's own connection
        size_t max_jobs = 2;                           // queued + reading + read and not taken (hints only)
        size_t max_bytes = size_t{16384} << 20;        // their payload bytes (hints only)
        std::chrono::milliseconds timeout{30000};      // a hint or a read image not taken by then is dropped
        // resolver thread: the file the hint's next turn would read (nullopt: none)
        std::function<std::optional<Target>(const Hint&)> locate;
        // resolver thread: the background park holds an image of this conversation at least `tokens` deep
        std::function<bool(const Hint&, int64_t tokens)> parked;
        // loader thread: the RAM admission for `bytes` more (unset: admitted)
        std::function<bool(size_t bytes)> admit;
    };
    struct Stats {
        uint64_t hints = 0, resolved = 0, bad = 0, none = 0, parked = 0, capped = 0, duplicate = 0, stale = 0;
        uint64_t loads = 0, failed = 0, refused = 0, taken = 0, waited = 0, expired = 0, dropped = 0;
        size_t jobs = 0, bytes = 0;
        double load_ms = 0;   // summed over finished reads
    };

    explicit ConversationPrefetch(Options options);
    ~ConversationPrefetch();
    ConversationPrefetch(const ConversationPrefetch&) = delete;
    ConversationPrefetch& operator=(const ConversationPrefetch&) = delete;
    /// Stops both threads and frees every image; later calls do nothing.  (Before the objects the callbacks use go.)
    void stop();

    /// Any thread: the text of a PREFETCH line after the word (keys, then the token ids).
    void hint(std::string text);
    /// Serve thread: read `t` now.  Returns the job's ticket (a job already on this file and serial is reused).
    uint64_t load(const Target& t);
    /// The job has finished reading, failed, or is gone (taken, dropped).
    bool settled(uint64_t ticket) const;
    /// Waits until settled(ticket).
    void wait(uint64_t ticket);
    /// Serve thread: the image of `t` (name and serial), waiting for a job that is reading it.  nullopt: no such job,
    /// one not started yet (the caller reads it itself), a failed one or one of another serial (`note` says which).
    std::optional<SavedConversation> take(const Target& t, double& waited_ms, std::string& note);
    /// Serve thread: drops a job on file `name` (its conversation was served another way).  True when one was held.
    bool drop(const std::string& name);
    Stats stats() const;

private:
    enum class State { queued, reading, ready, failed };
    struct Job {
        uint64_t id = 0;
        Target target;
        State state = State::queued;
        bool urgent = false;
        std::unique_ptr<SavedConversation> image;
        std::string error;
        std::chrono::steady_clock::time_point created, finished;
        std::atomic<bool> cancel{false};
        double load_ms = 0;
    };
    void resolve_loop();
    void load_loop();
    // under m_: the job on file `name`, or jobs_.end()
    std::deque<std::shared_ptr<Job>>::iterator find_locked(const std::string& name);
    // under m_: removes `it`; a reading job is cancelled and its image freed by the loader, a read image is freed there too
    void erase_locked(std::deque<std::shared_ptr<Job>>::iterator it);
    size_t bytes_locked() const;

    Options o_;
    mutable std::mutex m_;
    std::condition_variable cv_;        // a hint, a job, a finished read, a drop, stop
    std::condition_variable done_cv_;   // a job settled
    std::deque<std::pair<std::chrono::steady_clock::time_point, std::string>> inbox_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::vector<std::unique_ptr<SavedConversation>> trash_;   // freed on the loader thread, off the serve thread
    uint64_t next_id_ = 1;
    bool stop_ = false;
    Stats stats_;
    std::thread resolver_, loader_;
};

} // namespace strata::core
