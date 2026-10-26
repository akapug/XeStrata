// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/conversation_folder.hpp"
#include "strata/core/kvib.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#if defined(_WIN32)
#include <io.h>
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace strata::core {
namespace {
namespace fs = std::filesystem;

#if !defined(_WIN32)
// The CPUs the process may use, taken before main (before the serve loop pins its thread to one core).
struct StartAffinity {
    cpu_set_t set;
    bool ok;
    StartAffinity() noexcept {
        CPU_ZERO(&set);
        ok = sched_getaffinity(0, sizeof set, &set) == 0 && CPU_COUNT(&set) > 0;
    }
};
const StartAffinity g_start_affinity;
#endif

// A fork-join over a few threads, made when first needed.  One job at a time: a caller that finds it busy (the
// background park writer and the serve thread at once) runs its job alone, on its own thread.
class IoPool {
public:
    ~IoPool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    void set(int threads) { threads_.store(std::clamp(threads, 1, 64)); }
    int threads() const { return threads_.load(); }
    void run(size_t n, const std::function<void(size_t)>& fn) {
        std::unique_lock<std::mutex> job(job_m_, std::defer_lock);
        if (n < 2 || threads_.load() < 2 || !job.try_lock()) {
            for (size_t i = 0; i < n; ++i) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            while ((int) workers_.size() < threads_.load() - 1) workers_.emplace_back([this] { loop(); });
            fn_ = &fn;
            n_ = n;
            next_.store(0);
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> lock(m_);
        done_.wait(lock, [&] { return active_ == 0 && next_.load() >= n_; });
        fn_ = nullptr;
    }

private:
    void work() {
        for (size_t i; (i = next_.fetch_add(1)) < n_;) (*fn_)(i);
    }
    void loop() {
        conversation_io_widen_thread();
        std::unique_lock<std::mutex> lock(m_);
        uint64_t seen = gen_;
        for (;;) {
            cv_.wait(lock, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            if (fn_ == nullptr) continue;
            ++active_;
            lock.unlock();
            work();
            lock.lock();
            --active_;
            done_.notify_all();
        }
    }
    std::atomic<int> threads_{ConversationIoSettings::kThreadsDefault};
    std::mutex job_m_, m_;
    std::condition_variable cv_, done_;
    std::vector<std::thread> workers_;
    const std::function<void(size_t)>* fn_ = nullptr;
    size_t n_ = 0;
    std::atomic<size_t> next_{0};
    uint64_t gen_ = 0;
    int active_ = 0;
    bool stop_ = false;
};

IoPool& io_pool() {
    static IoPool pool;
    return pool;
}

// a copy smaller than this is one memcpy; a larger one is split in pieces of at least this
constexpr size_t kCopyPiece = size_t{1} << 20;

class LocalStream final : public FolderStream {
public:
    explicit LocalStream(std::FILE* f) : f_(f) {}
    ~LocalStream() override { if (f_ != nullptr) std::fclose(f_); }
    size_t read(void* p, size_t n) override {
#if !defined(_WIN32)
        // a large read: preads of its pieces over the copy threads, straight into the caller's memory, then the
        // stream's position past them
        if (n >= 2 * kCopyPiece && io_pool().threads() > 1) {
            const off_t at = ::ftello(f_);
            if (at >= 0) {
                const int fd = ::fileno(f_);
                const size_t pieces = std::min<size_t>((size_t) io_pool().threads() * 2, n / kCopyPiece);
                // every piece but the last a whole number of pages; the last one ends at n
                const size_t each = ((n + pieces - 1) / pieces + 4095) / 4096 * 4096;
                std::vector<size_t> got(pieces, 0);
                conversation_io_parallel(pieces, [&](size_t i) {
                    const size_t from = std::min(n, i * each), len = std::min(n, from + each) - from;
                    auto* out = static_cast<uint8_t*>(p) + from;
                    while (got[i] < len) {
                        const ssize_t k = ::pread(fd, out + got[i], len - got[i], at + (off_t) (from + got[i]));
                        if (k <= 0) break;
                        got[i] += (size_t) k;
                    }
                });
                size_t total = 0;   // the bytes read without a gap
                for (size_t i = 0; i < pieces; ++i) {
                    total += got[i];
                    if (got[i] < std::min(n, (i + 1) * each) - std::min(n, i * each)) break;
                }
                if (::fseeko(f_, at + (off_t) total, SEEK_SET) == 0) return total;
                return 0;
            }
        }
#endif
        return std::fread(p, 1, n, f_);
    }
    bool write(const void* p, size_t n) override { return std::fwrite(p, 1, n, f_) == n; }
    bool seek(uint64_t offset) override {
#if defined(_WIN32)
        return ::_fseeki64(f_, (long long) offset, SEEK_SET) == 0;
#else
        return ::fseeko(f_, (off_t) offset, SEEK_SET) == 0;
#endif
    }
    int64_t tell() override {
#if defined(_WIN32)
        return ::_ftelli64(f_);   // ftell's long is 32 bits here
#else
        return (int64_t) ::ftello(f_);
#endif
    }
    bool failed() const override { return std::ferror(f_) != 0; }
    bool finish() override {
        bool ok = std::fflush(f_) == 0;
#if defined(_WIN32)
        ok = ::_commit(::_fileno(f_)) == 0 && ok;
#else
        ok = ::fsync(::fileno(f_)) == 0 && ok;
#endif
        ok = std::fclose(f_) == 0 && ok;
        f_ = nullptr;
        return ok;
    }

private:
    std::FILE* f_;
};

class LocalFolder final : public ConversationFolder {
public:
    explicit LocalFolder(fs::path dir) : dir_(std::move(dir)) {}
    bool prepare(std::string& error) override {
        std::error_code ec;
        if (!fs::exists(dir_, ec)) {
            if (!fs::create_directories(dir_, ec) || ec) {
                error = "cannot create " + dir_.string() + ": " + ec.message();
                return false;
            }
            // the files hold what was computed from the conversations: the owner's only
            fs::permissions(dir_, fs::perms::owner_all, fs::perm_options::replace, ec);
        }
        if (!fs::is_directory(dir_, ec)) {
            error = dir_.string() + " is not a folder";
            return false;
        }
        return true;
    }
    std::vector<FolderFile> list() override {
        std::vector<FolderFile> out;
        std::error_code ec;
        for (fs::directory_iterator it(dir_, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code e;
            if (!it->is_regular_file(e)) continue;
            const auto time = it->last_write_time(e);
            const auto bytes = e ? 0 : it->file_size(e);
            if (!e) out.push_back({it->path().filename().string(), time, bytes});
        }
        std::sort(out.begin(), out.end(), [](const FolderFile& a, const FolderFile& b) { return a.time < b.time; });
        return out;
    }
    // fs::path's native string is UTF-16 on Windows, where fopen would take the ANSI code page
    std::unique_ptr<FolderStream> open_read(const std::string& name) override {
        const fs::path p = dir_ / name;
#if defined(_WIN32)
        std::FILE* f = ::_wfopen(p.c_str(), L"rb");
#else
        std::FILE* f = std::fopen(p.c_str(), "rb");
#endif
        return f == nullptr ? nullptr : std::make_unique<LocalStream>(f);
    }
    std::unique_ptr<FolderStream> open_write(const std::string& name) override {
        const fs::path p = dir_ / name;
#if defined(_WIN32)
        const int fd = ::_wopen(p.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY | _O_NOINHERIT,
                                _S_IREAD | _S_IWRITE);
        std::FILE* f = fd < 0 ? nullptr : ::_fdopen(fd, "wb");
        if (f == nullptr && fd >= 0) ::_close(fd);
#else
        const int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        std::FILE* f = fd < 0 ? nullptr : ::fdopen(fd, "wb");
        if (f == nullptr && fd >= 0) ::close(fd);
#endif
        return f == nullptr ? nullptr : std::make_unique<LocalStream>(f);
    }
    bool rename(const std::string& from, const std::string& to) override {
        std::error_code ec;
        fs::rename(dir_ / from, dir_ / to, ec);
        return !ec;
    }
    bool remove(const std::string& name) override {
        std::error_code ec;
        return fs::remove(dir_ / name, ec);
    }
    void touch(const std::string& name) override {
        std::error_code ec;
        fs::last_write_time(dir_ / name, fs::file_time_type::clock::now(), ec);
    }
    std::string where() const override { return dir_.string(); }

private:
    fs::path dir_;
};
} // namespace

bool conversation_io_settings_from_env(ConversationIoSettings& s, std::string& error) {
    auto one = [&](const char* name, long lo, long hi, int& out) {
        const char* e = std::getenv(name);
        if (e == nullptr || e[0] == '\0') return true;
        char* end = nullptr;
        const long x = std::strtol(e, &end, 10);
        if (end == e || *end != '\0' || x < lo || x > hi) {
            error = std::string(name) + "=" + e + ": expected an integer in [" + std::to_string(lo) + ", " +
                    std::to_string(hi) + "]";
            return false;
        }
        out = (int) x;
        return true;
    };
    return one("STRATA_CONV_IO_THREADS", 1, 64, s.threads) && one("STRATA_CONV_HASH_QUEUE_MIB", 1, 4096, s.hash_queue_mib);
}

namespace {
std::atomic<size_t> g_hash_queue{size_t{ConversationIoSettings::kHashQueueMibDefault} << 20};
} // namespace
void conversation_io_set_hash_queue(size_t bytes) { g_hash_queue.store(std::max<size_t>(1, bytes)); }
size_t conversation_io_hash_queue() { return g_hash_queue.load(); }

void conversation_io_set_threads(int threads) { io_pool().set(threads); }
int conversation_io_threads() { return io_pool().threads(); }

void conversation_io_parallel(size_t n, const std::function<void(size_t)>& fn) { io_pool().run(n, fn); }

void conversation_io_copy(void* dst, const void* src, size_t n) {
    const int t = io_pool().threads();
    if (t < 2 || n < 2 * kCopyPiece) {
        std::memcpy(dst, src, n);
        return;
    }
    const size_t pieces = std::min<size_t>((size_t) t, n / kCopyPiece);
    const size_t each = ((n + pieces - 1) / pieces + 63) / 64 * 64;   // pieces x each >= n: the last ends at n
    io_pool().run(pieces, [&](size_t i) {
        const size_t from = std::min(n, i * each), to = std::min(n, from + each);
        std::memcpy(static_cast<uint8_t*>(dst) + from, static_cast<const uint8_t*>(src) + from, to - from);
    });
}

void conversation_io_widen_thread() {
#if !defined(_WIN32)
    if (g_start_affinity.ok) pthread_setaffinity_np(pthread_self(), sizeof g_start_affinity.set, &g_start_affinity.set);
#endif
}

std::unique_ptr<ConversationFolder> conversation_folder_local(const fs::path& dir) {
    return std::make_unique<LocalFolder>(dir);
}

bool conversation_folder_is_remote(const std::string& spec) { return spec.rfind(kvib::kScheme, 0) == 0; }

std::unique_ptr<ConversationFolder> conversation_folder_open(const std::string& spec) {
    if (conversation_folder_is_remote(spec)) return kvib::folder(spec.substr(std::string(kvib::kScheme).size()));
    return conversation_folder_local(spec);
}

} // namespace strata::core
