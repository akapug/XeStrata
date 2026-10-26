// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/conversation_prefetch.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::core {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
} // namespace

bool ConversationPrefetch::parse(const std::string& text, Hint& out, std::string& why) {
    out = Hint{};
    const char* p = text.c_str();
    for (;;) {   // key=value tokens until the first one without '='
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        if (p == start) break;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            p = start;
            break;
        }
        const std::string key = tok.substr(0, eq);
        if (key == "cvec") out.cvec = std::strtol(tok.c_str() + eq + 1, nullptr, 10) != 0;
        if (key == "img_keys") {
            why = "a prompt with pictures";
            return false;
        }
    }
    while (*p == ' ') ++p;
    char* e = nullptr;
    while (*p != '\0') {
        const long long t = std::strtoll(p, &e, 10);
        if (e == p || t < 0 || t > INT32_MAX) {
            why = "a bad token list";
            return false;
        }
        out.ids.push_back((int32_t) t);
        p = e;
        if (*p == ',') ++p;
        else if (*p != '\0' && *p != ' ' && *p != '\n' && *p != '\r') {
            why = "a bad token list";
            return false;
        }
        while (*p == ' ' || *p == '\n' || *p == '\r') ++p;
    }
    if (out.ids.empty()) {
        why = "no prompt tokens";
        return false;
    }
    return true;
}

ConversationPrefetch::ConversationPrefetch(Options options) : o_(std::move(options)) {
    o_.max_jobs = std::max<size_t>(1, o_.max_jobs);
    // the loader connection's staging buffers count under the byte bound (homelab review): images get the rest
    o_.max_bytes -= std::min(o_.max_bytes, o_.folder->staging_bytes());
    resolver_ = std::thread([this] { resolve_loop(); });
    loader_ = std::thread([this] { load_loop(); });
}

ConversationPrefetch::~ConversationPrefetch() { stop(); }

void ConversationPrefetch::stop() {
    {
        std::lock_guard<std::mutex> lock(m_);
        if (stop_ && !resolver_.joinable()) return;
        stop_ = true;
        for (auto& j : jobs_) j->cancel = true;
    }
    cv_.notify_all();
    done_cv_.notify_all();
    if (resolver_.joinable()) resolver_.join();
    if (loader_.joinable()) loader_.join();
    std::lock_guard<std::mutex> lock(m_);
    jobs_.clear();
    inbox_.clear();
    trash_.clear();
    done_cv_.notify_all();
}

void ConversationPrefetch::hint(std::string text) {
    {
        std::lock_guard<std::mutex> lock(m_);
        if (stop_) return;
        ++stats_.hints;
        inbox_.emplace_back(Clock::now(), std::move(text));
    }
    cv_.notify_all();
}

std::deque<std::shared_ptr<ConversationPrefetch::Job>>::iterator ConversationPrefetch::find_locked(
    const std::string& name) {
    return std::find_if(jobs_.begin(), jobs_.end(), [&](const auto& j) { return j->target.name == name; });
}

void ConversationPrefetch::erase_locked(std::deque<std::shared_ptr<Job>>::iterator it) {
    Job& j = **it;
    j.cancel = true;   // a reading job stops at its next piece; the loader frees what it read
    if (j.image) trash_.push_back(std::move(j.image));
    jobs_.erase(it);
    cv_.notify_all();
    done_cv_.notify_all();
}

size_t ConversationPrefetch::bytes_locked() const {
    size_t n = 0;
    for (const auto& j : jobs_) n += j->target.payload;
    return n;
}

void ConversationPrefetch::resolve_loop() {
    conversation_io_widen_thread();
    std::unique_lock<std::mutex> lock(m_);
    for (;;) {
        cv_.wait(lock, [&] { return stop_ || !inbox_.empty(); });
        if (stop_) return;
        auto [at, text] = std::move(inbox_.front());
        inbox_.pop_front();
        if (ms_since(at) > (double) o_.timeout.count()) {   // waited longer than its own turn may: stale
            ++stats_.expired;
            ++stats_.resolved;
            continue;
        }
        lock.unlock();
        Hint h;
        std::string why;
        const bool ok = parse(text, h, why);
        text.clear();
        text.shrink_to_fit();
        std::optional<Target> t;
        bool in_park = false;
        if (ok && o_.locate) t = o_.locate(h);
        if (t && o_.parked) in_park = o_.parked(h, t->tokens);
        lock.lock();
        ++stats_.resolved;
        if (!ok) { ++stats_.bad; continue; }
        if (!t) { ++stats_.none; continue; }
        if (in_park) { ++stats_.parked; continue; }   // the request path takes that image back whole
        auto it = find_locked(t->name);
        if (it != jobs_.end() && (*it)->target.serial == t->serial) { ++stats_.duplicate; continue; }
        if (it != jobs_.end()) {   // an older write of the same name: never used
            ++stats_.stale;
            erase_locked(it);
        }
        if (jobs_.size() >= o_.max_jobs || bytes_locked() + t->payload > o_.max_bytes) { ++stats_.capped; continue; }
        auto j = std::make_shared<Job>();
        j->id = next_id_++;
        j->target = std::move(*t);
        j->created = Clock::now();
        jobs_.push_back(std::move(j));
        cv_.notify_all();
    }
}

uint64_t ConversationPrefetch::load(const Target& t) {
    std::lock_guard<std::mutex> lock(m_);
    if (stop_) return 0;
    auto it = find_locked(t.name);
    if (it != jobs_.end() && (*it)->target.serial == t.serial && (*it)->state != State::failed) {
        (*it)->urgent = true;
        return (*it)->id;
    }
    if (it != jobs_.end()) {
        if ((*it)->target.serial != t.serial) ++stats_.stale;
        erase_locked(it);
    }
    auto j = std::make_shared<Job>();
    j->id = next_id_++;
    j->target = t;
    j->urgent = true;
    j->created = Clock::now();
    const uint64_t id = j->id;
    jobs_.push_back(std::move(j));
    cv_.notify_all();
    return id;
}

bool ConversationPrefetch::settled(uint64_t ticket) const {
    std::lock_guard<std::mutex> lock(m_);
    for (const auto& j : jobs_)
        if (j->id == ticket) return j->state == State::ready || j->state == State::failed;
    return true;
}

void ConversationPrefetch::wait(uint64_t ticket) {
    std::unique_lock<std::mutex> lock(m_);
    done_cv_.wait(lock, [&] {
        if (stop_) return true;
        for (const auto& j : jobs_)
            if (j->id == ticket) return j->state == State::ready || j->state == State::failed;
        return true;
    });
}

void ConversationPrefetch::load_loop() {
    conversation_io_widen_thread();
    std::unique_lock<std::mutex> lock(m_);
    for (;;) {
        // what was dropped is freed here, never on the serve thread
        while (!trash_.empty()) {
            auto junk = std::move(trash_);
            lock.unlock();
            junk.clear();
            lock.lock();
        }
        if (stop_) return;
        // expiry: a hint's job not started, or a read image, not taken within the timeout
        const auto now = Clock::now();
        auto next = now + o_.timeout;
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            Job& j = **it;
            const bool waiting = j.state == State::queued && !j.urgent;
            const bool held = j.state == State::ready || j.state == State::failed;
            const auto from = held ? j.finished : j.created;
            if ((waiting || held) && now - from >= o_.timeout) {
                ++stats_.expired;
                erase_locked(it);
                it = jobs_.begin();
                continue;
            }
            if (waiting || held) next = std::min(next, from + o_.timeout);
            ++it;
        }
        if (!trash_.empty()) continue;
        // the next job: the serve thread's own first, then the oldest hint
        std::shared_ptr<Job> pick;
        for (const auto& j : jobs_)
            if (j->state == State::queued && (!pick || (j->urgent && !pick->urgent))) pick = j;
        if (!pick) {
            cv_.wait_until(lock, next);
            continue;
        }
        pick->state = State::reading;
        lock.unlock();
        const auto t0 = Clock::now();
        auto image = std::make_unique<SavedConversation>();
        std::string error;
        bool ok = false;
        if (o_.admit && !o_.admit(pick->target.payload)) {
            error = "physical RAM admission";
        } else {
            try {
                ok = ConversationDisk::read_file(o_.disk, *o_.folder, pick->target.name, *image, error, &pick->cancel);
            } catch (const std::bad_alloc&) {
                error = "allocation failed";
            }
        }
        const double ms = ms_since(t0);
        if (!ok) image.reset();   // a failed read's memory goes here, not under the lock
        lock.lock();
        pick->load_ms = ms;
        pick->finished = Clock::now();
        if (ok) {
            pick->state = State::ready;
            ++stats_.loads;
            stats_.load_ms += ms;
            if (pick->cancel) trash_.push_back(std::move(image));   // dropped while it was read
            else pick->image = std::move(image);
        } else {
            pick->state = State::failed;
            pick->error = error;
            if (error == "physical RAM admission") ++stats_.refused;
            else if (!pick->cancel) ++stats_.failed;
        }
        done_cv_.notify_all();
    }
}

std::optional<SavedConversation> ConversationPrefetch::take(const Target& t, double& waited_ms, std::string& note) {
    waited_ms = 0;
    note.clear();
    std::unique_lock<std::mutex> lock(m_);
    auto it = find_locked(t.name);
    if (it == jobs_.end()) return std::nullopt;
    if ((*it)->target.serial != t.serial) {
        ++stats_.stale;
        note = "read before a newer write of it; dropped";
        erase_locked(it);
        return std::nullopt;
    }
    if ((*it)->state == State::queued) {   // not started: the caller reads it as before
        note = "not started yet";
        erase_locked(it);
        return std::nullopt;
    }
    const std::shared_ptr<Job> j = *it;
    if (j->state == State::reading) {
        const auto t0 = Clock::now();
        ++stats_.waited;
        done_cv_.wait(lock, [&] { return stop_ || j->state != State::reading; });
        waited_ms = ms_since(t0);
        it = std::find(jobs_.begin(), jobs_.end(), j);
        if (it == jobs_.end()) {
            note = "dropped";
            return std::nullopt;
        }
    }
    if (j->state == State::failed || !j->image) {
        note = j->error.empty() ? "no image" : j->error;
        erase_locked(it);
        return std::nullopt;
    }
    std::optional<SavedConversation> out(std::move(*j->image));
    j->image.reset();
    ++stats_.taken;
    jobs_.erase(it);
    cv_.notify_all();
    return out;
}

bool ConversationPrefetch::drop(const std::string& name) {
    std::lock_guard<std::mutex> lock(m_);
    auto it = find_locked(name);
    if (it == jobs_.end()) return false;
    ++stats_.dropped;
    erase_locked(it);
    return true;
}

ConversationPrefetch::Stats ConversationPrefetch::stats() const {
    std::lock_guard<std::mutex> lock(m_);
    Stats s = stats_;
    s.jobs = jobs_.size();
    s.bytes = bytes_locked();
    return s;
}

} // namespace strata::core
