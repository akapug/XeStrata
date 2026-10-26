// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// kvib (kvib.hpp): the wire, the daemon's store, and the client folder with its transports.
#include "strata/core/kvib.hpp"
#include "kvib_rdma.hpp"

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace strata::core::kvib {

// ---- settings
namespace {
bool env_int(const char* key, long long lo, long long hi, long long& v, std::string& error) {
    const char* e = std::getenv(key);
    if (e == nullptr || e[0] == '\0') return true;
    char* end = nullptr;
    const long long x = std::strtoll(e, &end, 10);
    if (end == e || *end != '\0' || x < lo || x > hi) {
        error = std::string(key) + "=" + e + ": expected an integer in [" + std::to_string(lo) + ", " +
                std::to_string(hi) + "]";
        return false;
    }
    v = x;
    return true;
}
} // namespace

bool settings_from_env(Settings& s, std::string& error) {
    if (const char* t = std::getenv("STRATA_KVIB_TRANSPORT"); t != nullptr && t[0] != '\0') {
        const std::string v = t;
        if (v != "auto" && v != "rdma" && v != "tcp") {
            error = "STRATA_KVIB_TRANSPORT=" + v + ": expected auto, rdma or tcp";
            return false;
        }
        s.transport = v;
    }
    long long off = s.rdma_port_offset, mib = (long long) s.stage_mib, depth = (long long) s.stage_depth,
              streams = s.tcp_streams, ms = s.timeout_ms, re = s.reconnect_ms, re_max = s.reconnect_max_ms,
              cms = s.connect_timeout_ms;
    if (!env_int("STRATA_KVIB_RDMA_PORT_OFFSET", 1, 1000, off, error) ||
        !env_int("STRATA_KVIB_STAGE_MIB", 1, 1024, mib, error) ||
        !env_int("STRATA_KVIB_STAGE_DEPTH", 2, 64, depth, error) ||
        !env_int("STRATA_KVIB_TCP_STREAMS", 1, 64, streams, error) ||
        !env_int("STRATA_KVIB_TIMEOUT_MS", 100, 3600000, ms, error) ||
        !env_int("STRATA_KVIB_RECONNECT_MS", 0, 3600000, re, error) ||
        !env_int("STRATA_KVIB_RECONNECT_MAX_MS", 0, 86400000, re_max, error) ||
        !env_int("STRATA_KVIB_CONNECT_TIMEOUT_MS", 100, 3600000, cms, error))
        return false;
    if (re_max < re) {
        error = "STRATA_KVIB_RECONNECT_MAX_MS=" + std::to_string(re_max) + ": below STRATA_KVIB_RECONNECT_MS (" +
                std::to_string(re) + ")";
        return false;
    }
    s.reconnect_ms = (int) re;
    s.reconnect_max_ms = (int) re_max;
    s.connect_timeout_ms = (int) cms;
    s.rdma_port_offset = (int) off;
    s.stage_mib = (size_t) mib;
    s.stage_depth = (size_t) depth;
    s.tcp_streams = (int) streams;
    s.timeout_ms = (int) ms;
    if (const char* v = std::getenv("STRATA_KVIB_LIBIBVERBS"); v != nullptr && v[0] != '\0') s.libibverbs = v;
    if (const char* v = std::getenv("STRATA_KVIB_LIBRDMACM"); v != nullptr && v[0] != '\0') s.librdmacm = v;
    return true;
}

// ---- the wire: a request is its header, its name and its data; a response its header, a text and its data
namespace {
constexpr uint32_t kMagic = 0x4b564942;   // "KVIB"
enum Op : uint16_t {
    kHello = 1, kList, kStat, kCreate, kExtend, kWrite, kSetSize, kCommit, kAbort, kOpen, kRead, kClose, kRemove, kTouch
};
struct Req {
    uint32_t magic = kMagic;
    uint16_t op = 0, flags = 0;
    uint64_t a = 0, b = 0, c = 0;
    uint32_t name_len = 0, data_len = 0;
};
struct Resp {
    uint32_t magic = kMagic;
    int32_t status = 0;   // 0, or an errno value
    uint64_t a = 0, b = 0, c = 0;
    uint32_t str_len = 0, data_len = 0;
};
static_assert(sizeof(Req) == 40 && std::has_unique_object_representations_v<Req>, "Req is its bytes");
static_assert(sizeof(Resp) == 40 && std::has_unique_object_representations_v<Resp>, "Resp is its bytes");
constexpr uint32_t kMaxName = 4096;
constexpr uint32_t kMaxData = uint32_t{1} << 30;
// an extent as OPEN and EXTEND describe it (rkey 0: the session has no RDMA)
struct WireExtent {
    uint64_t addr = 0, len = 0;
    uint32_t rkey = 0, pad = 0;
};
static_assert(sizeof(WireExtent) == 24 && std::has_unique_object_representations_v<WireExtent>, "WireExtent");

// where a request's data comes from / a response's data goes: a socket or memory (the in-process channel)
using In = std::function<bool(void*, size_t)>;
struct Piece {
    const uint8_t* p;
    size_t n;
};
} // namespace

// ---- the daemon's store
struct Server::Impl {
    ServerOptions opt;
    struct Extent {
        uint8_t* p = nullptr;
        size_t len = 0;
        std::vector<void*> mrs;
        std::vector<uint32_t> rkeys;
    };
    struct Obj {
        std::mutex m;
        std::vector<std::shared_ptr<Extent>> ext;
        uint64_t size = 0;
        std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
    };
    // A control connection.  The handles it creates or opens are released when it ends, so a client that dies
    // mid-write or mid-read leaves nothing pinned (a data connection only uses handles, it owns none).
    struct Session {
        uint64_t token = 0;
        std::atomic<int> dev{-1};
        std::vector<uint64_t> owned;   // under Impl::m
    };
    mutable std::mutex m;
    std::map<std::string, std::shared_ptr<Obj>> objs;
    std::unordered_map<uint64_t, std::shared_ptr<Obj>> handles;
    std::unordered_map<uint64_t, Session*> sessions;
    // `mapped`: every extent's bytes, against the arena; `live`: those in objects and open writes.  A freed extent goes
    // back to `pool` still registered (keep_pool), and the next object takes it first.
    std::atomic<uint64_t> mapped{0}, live{0}, next_handle{1};
    std::mutex pool_m;
    std::vector<Extent*> pool;
    std::atomic<bool> keep_pool{true};
    std::mt19937_64 rng{std::random_device{}()};
    rdma::ServerSide rd;
    bool rdma_on = false;
    // sockets
    std::vector<int> listen_fds;
    std::vector<std::thread> threads;
    std::mutex conn_m;
    std::vector<int> conn_fds;
    std::atomic<bool> stopping{false};

    void destroy(Extent* x) {
        if (!x->mrs.empty()) rd.dereg(x->mrs);
#if !defined(_WIN32)
        ::munmap(x->p, x->len);
#else
        std::free(x->p);
#endif
        mapped.fetch_sub(x->len);
        delete x;
    }
    void release(Extent* x) {
        live.fetch_sub(x->len);
        if (keep_pool) {
            std::lock_guard<std::mutex> lock(pool_m);
            pool.push_back(x);
            return;
        }
        destroy(x);
    }
    std::shared_ptr<Extent> adopt(Extent* x) {
        live.fetch_add(x->len);
        return std::shared_ptr<Extent>(x, [this](Extent* e) { release(e); });
    }
    std::shared_ptr<Extent> new_extent(std::string& error) {
        {
            std::lock_guard<std::mutex> lock(pool_m);
            if (!pool.empty()) {
                Extent* x = pool.back();
                pool.pop_back();
                return adopt(x);
            }
        }
        Extent* x = map_extent(error);
        return x == nullptr ? nullptr : adopt(x);
    }
    // a fresh extent, mapped and registered (not yet anyone's)
    Extent* map_extent(std::string& error) {
        if (mapped.fetch_add(opt.extent_bytes) + opt.extent_bytes > opt.arena_bytes) {
            mapped.fetch_sub(opt.extent_bytes);
            error = "arena full";
            return nullptr;
        }
#if !defined(_WIN32)
        void* p = ::mmap(nullptr, opt.extent_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            mapped.fetch_sub(opt.extent_bytes);
            error = std::string("mmap: ") + std::strerror(errno);
            return nullptr;
        }
        ::madvise(p, opt.extent_bytes, MADV_HUGEPAGE);
#else
        void* p = std::malloc(opt.extent_bytes);
#endif
        auto* e = new Extent;
        e->p = static_cast<uint8_t*>(p);
        e->len = opt.extent_bytes;
        if (rdma_on && !rd.reg(p, opt.extent_bytes, e->mrs, e->rkeys, error)) {
            destroy(e);
            return nullptr;
        }
        return e;
    }
    // the object's extents cover [0, upto) (o.m held)
    bool grow(Obj& o, uint64_t upto, std::string& error) {
        while ((uint64_t) o.ext.size() * opt.extent_bytes < upto) {
            auto e = new_extent(error);
            if (!e) return false;
            o.ext.push_back(std::move(e));
        }
        return true;
    }
    // the spans of [off, off + n) over the extents (o.m held; covered)
    std::vector<Piece> spans(Obj& o, uint64_t off, uint64_t n) {
        std::vector<Piece> out;
        while (n > 0) {
            const size_t i = (size_t) (off / opt.extent_bytes), at = (size_t) (off % opt.extent_bytes);
            const size_t k = (size_t) std::min<uint64_t>(n, opt.extent_bytes - at);
            out.push_back({o.ext[i]->p + at, k});
            off += k;
            n -= k;
        }
        return out;
    }
    std::vector<uint8_t> describe(Obj& o, const Session& s) {
        std::vector<uint8_t> out(o.ext.size() * sizeof(WireExtent));
        const int dev = s.dev.load();
        for (size_t i = 0; i < o.ext.size(); ++i) {
            WireExtent w;
            w.addr = (uint64_t) (uintptr_t) o.ext[i]->p;
            w.len = o.ext[i]->len;
            w.rkey = dev >= 0 && (size_t) dev < o.ext[i]->rkeys.size() ? o.ext[i]->rkeys[(size_t) dev] : 0;
            std::memcpy(out.data() + i * sizeof w, &w, sizeof w);
        }
        return out;
    }
    std::shared_ptr<Obj> handle(uint64_t h) {
        std::lock_guard<std::mutex> lock(m);
        auto it = handles.find(h);
        return it == handles.end() ? nullptr : it->second;
    }
    uint64_t age_ns(const Obj& o) const {
        return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                               o.time).count();
    }

    // One request.  `in` yields the request's data (every byte of it is consumed, also on a refusal); the response's
    // data is `data` followed by `pieces` (kept alive by `keep`).
    void dispatch(Session& s, const Req& q, const std::string& name, const In& in, Resp& r, std::string& str,
                  std::vector<uint8_t>& data, std::vector<Piece>& pieces, std::shared_ptr<Obj>& keep) {
        uint64_t left = q.data_len;
        auto fail = [&](int status, const std::string& why) {
            r.status = status;
            str = why;
            std::vector<uint8_t> sink(std::min<uint64_t>(left, uint64_t{1} << 20));
            while (left > 0) {
                const size_t k = (size_t) std::min<uint64_t>(left, sink.size());
                if (!in(sink.data(), k)) break;
                left -= k;
            }
        };
        switch (q.op) {
        case kHello: {
            // once a session: a second token would leave the first one naming this session after it ends
            if (s.token != 0) return fail(EPROTO, "hello twice on one connection");
            if (q.a != kProtocol) return fail(EPROTO, "protocol " + std::to_string(q.a) + ", this daemon speaks " +
                                                          std::to_string(kProtocol));
            std::lock_guard<std::mutex> lock(m);
            do s.token = rng(); while (s.token == 0 || sessions.count(s.token));
            sessions[s.token] = &s;
            r.a = s.token;
            r.b = opt.extent_bytes;
            r.c = opt.arena_bytes;
            str = rdma_on ? "rdma" : "tcp";
            return;
        }
        case kList: {
            std::lock_guard<std::mutex> lock(m);
            for (const auto& [n, o] : objs) {
                const uint64_t size = o->size, age = age_ns(*o);
                const uint16_t len = (uint16_t) n.size();
                const size_t at = data.size();
                data.resize(at + 18 + len);
                std::memcpy(data.data() + at, &size, 8);
                std::memcpy(data.data() + at + 8, &age, 8);
                std::memcpy(data.data() + at + 16, &len, 2);
                std::memcpy(data.data() + at + 18, n.data(), len);
            }
            return;
        }
        case kStat: {
            std::lock_guard<std::mutex> lock(m);
            r.a = live.load();
            r.b = opt.arena_bytes;
            r.c = objs.size();
            return;
        }
        case kCreate: {
            auto o = std::make_shared<Obj>();
            std::lock_guard<std::mutex> lock(m);
            r.a = next_handle++;
            handles[r.a] = o;
            s.owned.push_back(r.a);
            return;
        }
        case kExtend: {
            auto o = handle(q.a);
            if (!o) return fail(EBADF, "no such handle");
            std::lock_guard<std::mutex> lock(o->m);
            std::string why;
            if (!grow(*o, q.b, why)) return fail(ENOSPC, why);
            data = describe(*o, s);
            r.c = opt.extent_bytes;
            return;
        }
        case kWrite: {
            auto o = handle(q.a);
            if (!o) return fail(EBADF, "no such handle");
            // an offset whose end wraps would index past the extents
            if (q.b > UINT64_MAX - q.data_len) return fail(EINVAL, "write past the end of the address space");
            std::vector<Piece> sp;
            {
                std::lock_guard<std::mutex> lock(o->m);
                std::string why;
                if (!grow(*o, q.b + q.data_len, why)) return fail(ENOSPC, why);
                sp = spans(*o, q.b, q.data_len);
                o->size = std::max<uint64_t>(o->size, q.b + q.data_len);
            }
            for (const Piece& p : sp) {
                if (!in(const_cast<uint8_t*>(p.p), p.n)) return fail(EIO, "short write");
                left -= p.n;
            }
            return;
        }
        case kSetSize: {
            auto o = handle(q.a);
            if (!o) return fail(EBADF, "no such handle");
            std::lock_guard<std::mutex> lock(o->m);
            if (q.b > (uint64_t) o->ext.size() * opt.extent_bytes) return fail(EINVAL, "size past the extents");
            o->size = q.b;
            return;
        }
        case kCommit: {
            std::lock_guard<std::mutex> lock(m);
            auto it = handles.find(q.a);
            if (it == handles.end()) return fail(EBADF, "no such handle");
            it->second->time = std::chrono::steady_clock::now();
            objs[name] = it->second;   // replaces the old object; its readers keep it until they close
            handles.erase(it);
            return;
        }
        case kAbort:
        case kClose: {
            std::lock_guard<std::mutex> lock(m);
            if (handles.erase(q.a) == 0) return fail(EBADF, "no such handle");
            return;
        }
        case kOpen: {
            std::lock_guard<std::mutex> lock(m);
            auto it = objs.find(name);
            if (it == objs.end()) return fail(ENOENT, "no such object");
            r.a = next_handle++;
            handles[r.a] = it->second;
            s.owned.push_back(r.a);
            std::lock_guard<std::mutex> ol(it->second->m);
            r.b = it->second->size;
            r.c = opt.extent_bytes;
            data = describe(*it->second, s);
            return;
        }
        case kRead: {
            auto o = handle(q.a);
            if (!o) return fail(EBADF, "no such handle");
            std::lock_guard<std::mutex> lock(o->m);
            if (q.b > o->size || q.c > o->size - q.b || q.c > kMaxData) return fail(EINVAL, "read past the end");
            pieces = spans(*o, q.b, q.c);
            keep = o;
            return;
        }
        case kRemove: {
            std::lock_guard<std::mutex> lock(m);
            if (objs.erase(name) == 0) return fail(ENOENT, "no such object");
            return;
        }
        case kTouch: {
            std::lock_guard<std::mutex> lock(m);
            auto it = objs.find(name);
            if (it == objs.end()) return fail(ENOENT, "no such object");
            it->second->time = std::chrono::steady_clock::now();
            return;
        }
        default:
            return fail(ENOSYS, "unknown operation");
        }
    }
    void end_session(Session& s) {
        std::lock_guard<std::mutex> lock(m);
        if (s.token != 0) sessions.erase(s.token);
        for (const uint64_t h : s.owned) handles.erase(h);   // handle ids are never reused
        s.owned.clear();
    }
    bool rdma_attach(uint64_t token, int dev) {
        std::lock_guard<std::mutex> lock(m);
        auto it = sessions.find(token);
        if (it == sessions.end()) return false;
        it->second->dev = dev;
        return true;
    }
#if !defined(_WIN32)
    static bool recv_all(int fd, void* p, size_t n) {
        auto* b = static_cast<uint8_t*>(p);
        while (n > 0) {
            const ssize_t k = ::recv(fd, b, n, 0);
            if (k <= 0) {
                if (k < 0 && errno == EINTR) continue;
                return false;
            }
            b += k;
            n -= (size_t) k;
        }
        return true;
    }
    static bool send_all(int fd, const void* p, size_t n) {
        auto* b = static_cast<const uint8_t*>(p);
        while (n > 0) {
            const ssize_t k = ::send(fd, b, n, MSG_NOSIGNAL);
            if (k <= 0) {
                if (k < 0 && errno == EINTR) continue;
                return false;
            }
            b += k;
            n -= (size_t) k;
        }
        return true;
    }
    void serve(int fd) {
        Session s;
        for (;;) {
            Req q;
            if (!recv_all(fd, &q, sizeof q) || q.magic != kMagic || q.name_len > kMaxName || q.data_len > kMaxData) break;
            std::string name(q.name_len, '\0');
            if (!recv_all(fd, name.data(), name.size())) break;
            Resp r;
            std::string str;
            std::vector<uint8_t> data;
            std::vector<Piece> pieces;
            std::shared_ptr<Obj> keep;
            bool alive = true;
            const In in = [&](void* p, size_t n) { return alive = alive && recv_all(fd, p, n); };
            dispatch(s, q, name, in, r, str, data, pieces, keep);
            if (!alive) break;
            uint64_t total = data.size();
            for (const Piece& p : pieces) total += p.n;
            r.str_len = (uint32_t) str.size();
            r.data_len = (uint32_t) total;
            if (!send_all(fd, &r, sizeof r) || !send_all(fd, str.data(), str.size()) ||
                !send_all(fd, data.data(), data.size()))
                break;
            bool sent = true;
            for (const Piece& p : pieces) sent = sent && send_all(fd, p.p, p.n);
            if (!sent) break;
        }
        end_session(s);
        ::close(fd);
        std::lock_guard<std::mutex> lock(conn_m);
        conn_fds.erase(std::remove(conn_fds.begin(), conn_fds.end(), fd), conn_fds.end());
    }
    void accept_loop(int lfd) {
        while (!stopping) {
            pollfd p{lfd, POLLIN, 0};
            if (::poll(&p, 1, 200) <= 0) continue;
            const int fd = ::accept(lfd, nullptr, nullptr);
            if (fd < 0) continue;
            const int one = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            {
                std::lock_guard<std::mutex> lock(conn_m);
                conn_fds.push_back(fd);
            }
            std::thread([this, fd] { serve(fd); }).detach();
        }
    }
#endif
};

namespace {
bool split_endpoint(const std::string& ep, std::string& host, int& port, std::string& error) {
    const size_t colon = ep.rfind(':');
    char* end = nullptr;
    const long p = colon == std::string::npos ? 0 : std::strtol(ep.c_str() + colon + 1, &end, 10);
    if (colon == std::string::npos || colon == 0 || end == nullptr || *end != '\0' || p <= 0 || p > 65535) {
        error = "'" + ep + "': expected HOST:PORT";
        return false;
    }
    host = ep.substr(0, colon);
    port = (int) p;
    return true;
}
} // namespace

Server::Server(ServerOptions options) : impl_(std::make_unique<Impl>()) { impl_->opt = std::move(options); }
Server::~Server() { stop(); }

bool Server::start(bool listen, std::string& error) {
    Impl& s = *impl_;
    if (s.opt.extent_bytes == 0 || s.opt.arena_bytes < s.opt.extent_bytes) {
        error = "the arena must hold at least one extent";
        return false;
    }
    if (s.opt.transport != "tcp") {
        std::string why;
        const rdma::Api* api = rdma::load(s.opt.libibverbs, s.opt.librdmacm, why);
        s.rdma_on = api != nullptr && s.rd.open(api, why);
        if (!s.rdma_on && s.opt.transport == "rdma") {
            error = "rdma: " + why;
            return false;
        }
        if (!s.rdma_on && s.opt.verbose) std::fprintf(stderr, "kvib-store: no RDMA (%s): TCP only\n", why.c_str());
    }
    for (uint64_t at = 0; at + s.opt.extent_bytes <= s.opt.prealloc_bytes; at += s.opt.extent_bytes) {
        Server::Impl::Extent* x = s.map_extent(error);
        if (x == nullptr) return false;
        std::lock_guard<std::mutex> lock(s.pool_m);
        s.pool.push_back(x);
    }
    if (!listen) return true;
#if defined(_WIN32)
    error = "the kvib daemon runs on Linux";
    return false;
#else
    std::vector<std::pair<std::string, int>> rdma_eps;
    for (const std::string& ep : s.opt.listen) {
        std::string host;
        int port = 0;
        if (!split_endpoint(ep, host, port, error)) return false;
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;
        if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || res == nullptr) {
            error = "cannot resolve " + host;
            return false;
        }
        const int fd = ::socket(res->ai_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
        const int one = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        const bool ok = fd >= 0 && ::bind(fd, res->ai_addr, res->ai_addrlen) == 0 && ::listen(fd, 64) == 0;
        ::freeaddrinfo(res);
        if (!ok) {
            error = "cannot listen on " + ep + ": " + std::strerror(errno);
            if (fd >= 0) ::close(fd);
            return false;
        }
        s.listen_fds.push_back(fd);
        rdma_eps.emplace_back(host, port + s.opt.rdma_port_offset);
    }
    if (s.rdma_on &&
        !s.rd.listen(rdma_eps, [&s](uint64_t token, int dev) { return s.rdma_attach(token, dev); }, s.opt.verbose,
                     error))
        return false;
    for (const int fd : s.listen_fds) s.threads.emplace_back([&s, fd] { s.accept_loop(fd); });
    return true;
#endif
}

void Server::stop() {
    Impl& s = *impl_;
    if (s.stopping.exchange(true)) return;
    for (auto& t : s.threads) t.join();
    s.threads.clear();
#if !defined(_WIN32)
    for (const int fd : s.listen_fds) ::close(fd);
    s.listen_fds.clear();
    {
        std::lock_guard<std::mutex> lock(s.conn_m);
        for (const int fd : s.conn_fds) ::shutdown(fd, SHUT_RDWR);
    }
    // the connection threads end on the shut-down sockets
    for (int i = 0; i < 100; ++i) {
        {
            std::lock_guard<std::mutex> lock(s.conn_m);
            if (s.conn_fds.empty()) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#endif
    {   // the extents are deregistered before the devices close
        s.keep_pool = false;
        std::lock_guard<std::mutex> lock(s.m);
        s.handles.clear();
        s.objs.clear();
        std::lock_guard<std::mutex> pl(s.pool_m);
        for (auto* x : s.pool) s.destroy(x);
        s.pool.clear();
    }
    s.rd.stop();
}

ServerStats Server::stats() const {
    const Impl& s = *impl_;
    std::lock_guard<std::mutex> lock(s.m);
    ServerStats st;
    st.used = s.live.load();
    st.mapped = s.mapped.load();
    st.cap = s.opt.arena_bytes;
    st.objects = s.objs.size();
    st.handles = s.handles.size();
    st.rdma_devices = s.rdma_on ? (uint64_t) s.rd.devices() : 0;
    return st;
}

// ---- the client
namespace {

// One request and its response.  The response's data goes to `vec`, or to `dst` (at most `cap` bytes).
class Channel {
public:
    virtual ~Channel() = default;
    /// false once the connection broke (the daemon went away)
    virtual bool alive() const { return true; }
    /// ends the connection (the daemon then releases the session's handles); alive() is false after
    virtual void close() {}
    virtual bool call(const Req& q, const std::string& name, const void* in, size_t n_in, Resp& r, std::string& str,
                      std::vector<uint8_t>* vec, void* dst, size_t cap, std::string& error) = 0;
    bool simple(uint16_t op, uint64_t a, uint64_t b, uint64_t c, const std::string& name, Resp& r,
                std::vector<uint8_t>* vec, std::string& error) {
        Req q;
        q.op = op;
        q.a = a;
        q.b = b;
        q.c = c;
        std::string str;
        if (!call(q, name, nullptr, 0, r, str, vec, nullptr, 0, error)) return false;
        if (r.status != 0) {
            error = str.empty() ? std::strerror(r.status) : str;
            return false;
        }
        return true;
    }
};

class LocalChannel final : public Channel {
public:
    explicit LocalChannel(Server::Impl& s) : s_(s) {}
    ~LocalChannel() override { s_.end_session(sess_); }
    bool call(const Req& q0, const std::string& name, const void* in, size_t n_in, Resp& r, std::string& str,
              std::vector<uint8_t>* vec, void* dst, size_t cap, std::string& error) override {
        Req q = q0;
        q.name_len = (uint32_t) name.size();
        q.data_len = (uint32_t) n_in;
        size_t at = 0;
        const In feed = [&](void* p, size_t n) {
            if (n > n_in - at) return false;
            std::memcpy(p, static_cast<const uint8_t*>(in) + at, n);
            at += n;
            return true;
        };
        std::vector<uint8_t> data;
        std::vector<Piece> pieces;
        std::shared_ptr<Server::Impl::Obj> keep;
        r = Resp{};
        s_.dispatch(sess_, q, name, feed, r, str, data, pieces, keep);
        size_t total = data.size();
        for (const Piece& p : pieces) total += p.n;
        r.data_len = (uint32_t) total;
        if (vec != nullptr) {
            *vec = std::move(data);
            for (const Piece& p : pieces) vec->insert(vec->end(), p.p, p.p + p.n);
            return true;
        }
        if (total > cap) {
            error = "response larger than its buffer";
            return false;
        }
        auto* d = static_cast<uint8_t*>(dst);
        if (!data.empty()) std::memcpy(d, data.data(), data.size());
        d += data.size();
        for (const Piece& p : pieces) {
            std::memcpy(d, p.p, p.n);
            d += p.n;
        }
        return true;
    }

private:
    Server::Impl& s_;
    Server::Impl::Session sess_;
};

#if !defined(_WIN32)
class TcpChannel final : public Channel {
public:
    ~TcpChannel() override { close(); }
    bool alive() const override { return fd_ >= 0; }
    void close() override {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }
    // `connect_ms` bounds the connect itself (a host that is down), `timeout_ms` every transfer after it
    bool connect(const std::string& host, int port, int connect_ms, int timeout_ms, std::string& error) {
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || res == nullptr) {
            error = "cannot resolve " + host;
            return false;
        }
        fd_ = ::socket(res->ai_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
        const timeval tc{connect_ms / 1000, (connect_ms % 1000) * 1000}, tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        const int one = 1;
        ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tc, sizeof tc);   // a blocking connect waits at most this
        ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        const bool ok = fd_ >= 0 && ::connect(fd_, res->ai_addr, res->ai_addrlen) == 0;
        ::freeaddrinfo(res);
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        if (!ok) error = "cannot connect to " + host + ":" + std::to_string(port) + ": " + std::strerror(errno);
        return ok;
    }
    bool call(const Req& q0, const std::string& name, const void* in, size_t n_in, Resp& r, std::string& str,
              std::vector<uint8_t>* vec, void* dst, size_t cap, std::string& error) override {
        Req q = q0;
        q.name_len = (uint32_t) name.size();
        q.data_len = (uint32_t) n_in;
        iovec iov[3] = {{&q, sizeof q}, {const_cast<char*>(name.data()), name.size()}, {const_cast<void*>(in), n_in}};
        if (!sendv(iov, 3) || !Server::Impl::recv_all(fd_, &r, sizeof r) || r.magic != kMagic) return broken(error);
        str.assign(r.str_len, '\0');
        if (!Server::Impl::recv_all(fd_, str.data(), str.size())) return broken(error);
        if (vec != nullptr) {
            vec->resize(r.data_len);
            return Server::Impl::recv_all(fd_, vec->data(), vec->size()) || broken(error);
        }
        if (r.data_len > cap) {
            error = "response larger than its buffer";
            return broken(error), false;
        }
        return Server::Impl::recv_all(fd_, dst, r.data_len) || broken(error);
    }

private:
    // sendmsg with MSG_NOSIGNAL, never writev: a daemon that went away mid-transfer must fail this call with EPIPE,
    // not raise SIGPIPE, whose default action ends the engine
    bool sendv(iovec* iov, int n) {
        while (n > 0) {
            msghdr mh{};
            mh.msg_iov = iov;
            mh.msg_iovlen = (size_t) n;
            const ssize_t k = ::sendmsg(fd_, &mh, MSG_NOSIGNAL);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) return false;
            size_t left = (size_t) k;
            while (n > 0 && left >= iov->iov_len) {
                left -= iov->iov_len;
                ++iov;
                --n;
            }
            if (n > 0) {
                iov->iov_base = static_cast<uint8_t*>(iov->iov_base) + left;
                iov->iov_len -= left;
            }
        }
        return true;
    }
    bool broken(std::string& error) {
        if (error.empty()) error = std::string("connection lost: ") + std::strerror(errno);
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    int fd_ = -1;
};
#endif

struct Extents {
    std::vector<WireExtent> v;
    uint64_t ext_bytes = 0;
    uint64_t covered() const { return (uint64_t) v.size() * ext_bytes; }
    void parse(const std::vector<uint8_t>& d, uint64_t bytes) {
        v.resize(d.size() / sizeof(WireExtent));
        if (!v.empty()) std::memcpy(v.data(), d.data(), v.size() * sizeof(WireExtent));
        ext_bytes = bytes;
    }
};

// The data path of one daemon: `depth` staging buffers of `bytes` each; post() moves one buffer's bytes to or from
// [off, off + len) of a handle's object, wait() ends that buffer's transfer.
class Io {
public:
    virtual ~Io() = default;
    size_t depth = 0, bytes = 0;
    virtual uint8_t* stage(size_t i) = 0;
    virtual bool needs_extents() const = 0;
    virtual bool post(size_t i, bool write, uint64_t h, uint64_t off, size_t len, const Extents& ex,
                      std::string& error) = 0;
    virtual bool wait(size_t i, std::string& error) = 0;
    virtual std::string name() const = 0;
    /// A transfer failed or timed out in the transport itself (not a refusal by the daemon): requests may still be in
    /// flight and the bookkeeping of what completed is lost, so this data path is never used again (FolderImpl::drop).
    virtual bool broken() const = 0;
    bool wait_all(std::string& error) {
        bool ok = true;
        for (size_t i = 0; i < depth; ++i) ok = wait(i, error) && ok;
        return ok;
    }
};

struct AlignedBuf {
    uint8_t* p = nullptr;
    size_t n = 0;
    explicit AlignedBuf(size_t bytes) : n(bytes) {
        p = static_cast<uint8_t*>(std::aligned_alloc(4096, (bytes + 4095) / 4096 * 4096));
    }
    ~AlignedBuf() { std::free(p); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
};

// WRITE / READ requests over channels: one channel runs in the caller's thread (the in-process store); several
// (tcp streams) each get a thread, and buffer i goes to channel i % n.
class ChannelIo final : public Io {
public:
    ChannelIo(std::vector<std::unique_ptr<Channel>> ch, size_t d, size_t b, bool threaded, std::string label)
        : ch_(std::move(ch)), mem_(d * b), st_(d), label_(std::move(label)) {
        depth = d;
        bytes = b;
        if (threaded)
            for (size_t k = 0; k < ch_.size(); ++k) workers_.emplace_back([this, k] { work(k); });
    }
    ~ChannelIo() override {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    uint8_t* stage(size_t i) override { return mem_.p + i * bytes; }
    bool needs_extents() const override { return false; }
    bool post(size_t i, bool write, uint64_t h, uint64_t off, size_t len, const Extents&, std::string& error) override {
        Job j{i, write, h, off, len};
        if (workers_.empty()) return run(*ch_[0], j, error);
        std::lock_guard<std::mutex> lock(m_);
        st_[i].pending = true;
        st_[i].ok = true;
        q_[i % ch_.size()].push_back(j);
        cv_.notify_all();
        return true;
    }
    bool wait(size_t i, std::string& error) override {
        if (workers_.empty()) return true;
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return !st_[i].pending; });
        if (!st_[i].ok) {
            error = st_[i].err;
            st_[i].ok = true;
            return false;
        }
        return true;
    }
    std::string name() const override { return label_; }
    bool broken() const override {
        for (const auto& c : ch_)
            if (!c->alive()) return true;
        return false;
    }

private:
    struct Job {
        size_t i;
        bool write;
        uint64_t h, off;
        size_t len;
    };
    struct State {
        bool pending = false, ok = true;
        std::string err;
    };
    bool run(Channel& c, const Job& j, std::string& error) {
        Req q;
        q.op = j.write ? kWrite : kRead;
        q.a = j.h;
        q.b = j.off;
        q.c = j.len;
        Resp r;
        std::string str;
        const bool ok = j.write ? c.call(q, {}, stage(j.i), j.len, r, str, nullptr, nullptr, 0, error)
                                : c.call(q, {}, nullptr, 0, r, str, nullptr, stage(j.i), j.len, error);
        if (ok && r.status != 0) error = str.empty() ? std::strerror(r.status) : str;
        if (ok && r.status == 0 && !j.write && r.data_len != j.len) error = "short read";
        return ok && r.status == 0 && (j.write || r.data_len == j.len);
    }
    void work(size_t k) {
        std::unique_lock<std::mutex> lock(m_);
        while (!stop_) {
            if (q_[k].empty()) {
                cv_.wait(lock);
                continue;
            }
            const Job j = q_[k].front();
            q_[k].pop_front();
            lock.unlock();
            std::string e;
            const bool ok = run(*ch_[k], j, e);
            lock.lock();
            st_[j.i].pending = false;
            st_[j.i].ok = ok;
            st_[j.i].err = e;
            cv_.notify_all();
        }
    }
    std::vector<std::unique_ptr<Channel>> ch_;
    AlignedBuf mem_;
    std::vector<State> st_;
    std::string label_;
    std::mutex m_;
    std::condition_variable cv_;
    std::map<size_t, std::deque<Job>> q_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
};

// RDMA WRITE / READ of the daemon's extents through one queue pair; a buffer's transfer is one work request per
// extent it touches, and wait() polls until that buffer's requests have all completed.
class RdmaIo final : public Io {
public:
    RdmaIo(size_t d, size_t b, int timeout_ms) : mem_(d * b), pending_(d, 0), timeout_ms_(timeout_ms) {
        depth = d;
        bytes = b;
    }
    // a buffer's transfer is one work request per extent it touches: at most bytes / ext_bytes + 1 (unaligned)
    bool connect(const rdma::Api* api, const std::string& host, int port, uint64_t token, uint64_t ext_bytes,
                 int connect_ms, std::string& error) {
        const size_t per = ext_bytes == 0 ? 2 : (size_t) ((bytes + ext_bytes - 1) / ext_bytes) + 1;
        return qp_.connect(api, host, port, token, mem_.p, mem_.n, depth * per, connect_ms, error);
    }
    uint8_t* stage(size_t i) override { return mem_.p + i * bytes; }
    bool needs_extents() const override { return true; }
    bool post(size_t i, bool write, uint64_t, uint64_t off, size_t len, const Extents& ex,
              std::string& error) override {
        uint8_t* local = stage(i);
        while (len > 0) {
            const size_t e = (size_t) (off / ex.ext_bytes), at = (size_t) (off % ex.ext_bytes);
            if (e >= ex.v.size() || ex.v[e].rkey == 0) {
                error = "rdma: no registered extent for offset " + std::to_string(off);
                return false;
            }
            const size_t k = (size_t) std::min<uint64_t>(len, ex.ext_bytes - at);
            if (!qp_.post(i, write, local, k, ex.v[e].addr + at, ex.v[e].rkey, error)) return broken_ = true, false;
            ++pending_[i];
            local += k;
            off += k;
            len -= k;
        }
        return true;
    }
    bool wait(size_t i, std::string& error) override {
        std::vector<uint64_t> done;
        while (pending_[i] > 0) {
            done.clear();
            if (!qp_.poll(done, timeout_ms_, error)) {
                std::fill(pending_.begin(), pending_.end(), 0);
                return broken_ = true, false;
            }
            for (const uint64_t id : done) --pending_[(size_t) id];
        }
        return true;
    }
    std::string name() const override { return "rdma"; }
    bool broken() const override { return broken_; }
    std::string device() const { return qp_.device(); }

private:
    AlignedBuf mem_;
    std::vector<int> pending_;
    int timeout_ms_;
    bool broken_ = false;
    rdma::ClientQp qp_;
};

struct Daemon {
    std::string ep, host;
    int port = 0;
    std::unique_ptr<Channel> ctl;
    uint64_t token = 0, ext_bytes = 0, cap = 0;
    std::unique_ptr<Io> io;
    std::string label;
    // revive()'s back-off while this daemon cannot be dialled
    int fails = 0;
    std::chrono::steady_clock::time_point next_dial{};
};

class Stream;

class FolderImpl final : public Folder {
public:
    FolderImpl(std::string hosts, Settings s, Server* local) : hosts_(std::move(hosts)), s_(std::move(s)), local_(local) {}
    bool prepare(std::string& error) override;
    std::vector<FolderFile> list() override;
    std::unique_ptr<FolderStream> open_read(const std::string& name) override;
    std::unique_ptr<FolderStream> open_write(const std::string& name) override;
    bool rename(const std::string& from, const std::string& to) override;
    bool remove(const std::string& name) override;
    void touch(const std::string& name) override;
    std::string where() const override { return local_ != nullptr ? "kvib (in process)" : kScheme + hosts_; }
    std::string transport() const override { return ds_.empty() || !ds_[0].io ? "" : ds_[0].io->name(); }
    size_t staging_bytes() const override { return ds_.size() * (s_.stage_mib << 20) * s_.stage_depth; }
    bool read_at(const std::string& name, uint64_t offset, void* dst, size_t n, std::string& error) override;
    bool usage(uint64_t& used, uint64_t& cap, std::string& error) override;
    /// Daemon `d`'s data path broke (Io::broken): its queue pair or data connections go first (destroying them ends
    /// any request still in flight), then its control connection, whose end makes the daemon release this session's
    /// handles - so no late transfer lands in an extent the daemon has given to another object.  revive() dials again.
    void drop(size_t d) {
        ds_[d].io.reset();
        ds_[d].ctl->close();
    }

    // the streams' side
    struct Pending {
        size_t d;
        uint64_t h;
    };
    std::vector<Daemon> ds_;
    std::map<std::string, Pending> pending_;   // finished writes awaiting rename (commit)
    std::map<std::string, size_t> holder_;     // committed name -> daemon
    bool busy_ = false;                        // a stream is open (one at a time)

private:
    bool dial(const std::string& ep, Daemon& d, std::string& error);
    void revive();
    int holder_of(const std::string& name);
    std::string hosts_;
    Settings s_;
    Server* local_;
    struct Opened {
        size_t d;
        uint64_t h, size;
        Extents ex;
    };
    std::map<std::string, Opened> read_cache_;   // read_at's open handles
};

// A stream over one object on daemon `d`: writes fill buffer after buffer (each posted when full), reads fetch the
// buffers ahead of the position (doubling how far ahead while the reads stay sequential).
class Stream final : public FolderStream {
public:
    Stream(FolderImpl& f, size_t d, uint64_t h, bool writing, uint64_t size, Extents ex, std::string name)
        : f_(f), d_(f.ds_[d]), di_(d), h_(h), writing_(writing), size_(size), ex_(std::move(ex)), name_(std::move(name)),
          chunk_(d_.io->depth, -1), posted_(d_.io->depth, false) {
        f_.busy_ = true;
    }
    ~Stream() override {
        std::string e;
        d_.io->wait_all(e);
        if (d_.io->broken()) {
            f_.drop(di_);
        } else if (!done_) {   // an unfinished write is dropped; a reader just lets its handle go
            Resp r;
            d_.ctl->simple(writing_ ? kAbort : kClose, h_, 0, 0, {}, r, nullptr, e);
        }
        f_.busy_ = false;
    }
    size_t read(void* p, size_t n) override {
        auto* out = static_cast<uint8_t*>(p);
        size_t got = 0;
        const uint64_t S = d_.io->bytes, nchunks = (size_ + S - 1) / S;
        while (got < n && pos_ < size_ && !failed_) {
            const uint64_t c = pos_ / S;
            if (!post_chunk(c)) break;
            for (uint64_t k = 1; k <= ahead_ && c + k < nchunks; ++k)
                if (!post_chunk(c + k)) break;
            const size_t i = (size_t) (c % d_.io->depth);
            if (posted_[i] && !d_.io->wait(i, err_)) {
                failed_ = true;
                break;
            }
            posted_[i] = false;
            const uint64_t end = std::min<uint64_t>((c + 1) * S, size_);
            const size_t k = (size_t) std::min<uint64_t>(n - got, end - pos_);
            conversation_io_copy(out + got, d_.io->stage(i) + (pos_ - c * S), k);   // split over the copy threads
            got += k;
            pos_ += k;
            if (pos_ == end) ahead_ = std::min<uint64_t>(d_.io->depth - 1, std::max<uint64_t>(1, ahead_ * 2));
        }
        return got;
    }
    bool write(const void* p, size_t n) override {
        if (failed_) return false;
        if (cur_len_ > 0 && pos_ != cur_off_ + cur_len_ && !flush(true)) return false;
        const auto* in = static_cast<const uint8_t*>(p);
        while (n > 0) {
            if (cur_len_ == 0) cur_off_ = pos_;
            const size_t k = std::min(n, d_.io->bytes - cur_len_);
            conversation_io_copy(d_.io->stage(cur_) + cur_len_, in, k);
            cur_len_ += k;
            in += k;
            n -= k;
            pos_ += k;
            size_ = std::max(size_, pos_);
            if (cur_len_ == d_.io->bytes && !flush(false)) return false;
        }
        return true;
    }
    bool seek(uint64_t offset) override {
        // a write that goes back (the header) must not race the buffers still in flight over the same bytes
        if (writing_ && !flush(true)) return false;
        pos_ = offset;
        return true;
    }
    int64_t tell() override { return failed_ ? -1 : (int64_t) pos_; }
    bool failed() const override { return failed_; }
    bool finish() override {
        if (!writing_ || done_) return !failed_;
        bool ok = flush(true);
        Resp r;
        ok = ok && d_.ctl->simple(kSetSize, h_, size_, 0, {}, r, nullptr, err_);
        if (!ok) {
            failed_ = true;
            return false;
        }
        done_ = true;
        f_.pending_[name_] = {di_, h_};
        return true;
    }
    const std::string& error() const { return err_; }

private:
    bool post_chunk(uint64_t c) {
        const size_t i = (size_t) (c % d_.io->depth);
        if (chunk_[i] == (int64_t) c) return true;
        if (posted_[i] && !d_.io->wait(i, err_)) return failed_ = true, false;
        const uint64_t S = d_.io->bytes;
        const size_t len = (size_t) std::min<uint64_t>(S, size_ - c * S);
        chunk_[i] = (int64_t) c;
        posted_[i] = true;
        if (!d_.io->post(i, false, h_, c * S, len, ex_, err_)) {
            chunk_[i] = -1;
            posted_[i] = false;
            return failed_ = true, false;
        }
        return true;
    }
    bool extend(uint64_t upto) {
        if (!d_.io->needs_extents() || ex_.covered() >= upto) return true;
        const uint64_t want = (upto + d_.ext_bytes - 1) / d_.ext_bytes * d_.ext_bytes;
        Resp r;
        std::vector<uint8_t> d;
        if (!d_.ctl->simple(kExtend, h_, want, 0, {}, r, &d, err_)) return false;
        ex_.parse(d, r.c);
        return true;
    }
    // posts the current buffer and moves to the next (waiting for it); `all`: then waits for every buffer
    bool flush(bool all) {
        if (cur_len_ > 0) {
            if (!extend(cur_off_ + cur_len_) || !d_.io->post(cur_, true, h_, cur_off_, cur_len_, ex_, err_))
                return failed_ = true, false;
            posted_[cur_] = true;
            cur_ = (cur_ + 1) % d_.io->depth;
            cur_len_ = 0;
            if (posted_[cur_]) {
                posted_[cur_] = false;
                if (!d_.io->wait(cur_, err_)) return failed_ = true, false;
            }
        }
        if (all) {
            if (!d_.io->wait_all(err_)) return failed_ = true, false;
            std::fill(posted_.begin(), posted_.end(), false);
        }
        return true;
    }
    FolderImpl& f_;
    Daemon& d_;
    size_t di_;
    uint64_t h_;
    bool writing_;
    uint64_t size_, pos_ = 0;
    Extents ex_;
    std::string name_, err_;
    bool failed_ = false, done_ = false;
    // reads
    std::vector<int64_t> chunk_;
    std::vector<bool> posted_;
    uint64_t ahead_ = 0;
    // writes
    size_t cur_ = 0, cur_len_ = 0;
    uint64_t cur_off_ = 0;
};

bool FolderImpl::dial(const std::string& ep, Daemon& d, std::string& error) {
    d.ep = ep;
    if (local_ != nullptr) {
        d.ctl = std::make_unique<LocalChannel>(local_->impl());
    } else {
#if defined(_WIN32)
        error = "kvib needs Linux";
        return false;
#else
        if (!split_endpoint(ep, d.host, d.port, error)) return false;
        auto t = std::make_unique<TcpChannel>();
        if (!t->connect(d.host, d.port, s_.connect_timeout_ms, s_.timeout_ms, error)) return false;
        d.ctl = std::move(t);
#endif
    }
    Resp r;
    Req q;
    q.op = kHello;
    q.a = kProtocol;
    std::string str;
    if (!d.ctl->call(q, {}, nullptr, 0, r, str, nullptr, nullptr, 0, error)) return false;
    if (r.status != 0) {
        error = ep + ": " + str;
        return false;
    }
    d.token = r.a;
    d.ext_bytes = r.b;
    d.cap = r.c;
    const size_t S = s_.stage_mib << 20, D = s_.stage_depth;
    if (local_ != nullptr) {
        std::vector<std::unique_ptr<Channel>> ch;
        ch.push_back(std::make_unique<LocalChannel>(local_->impl()));
        d.io = std::make_unique<ChannelIo>(std::move(ch), D, S, false, "local");
    }
#if !defined(_WIN32)
    std::string why;
    if (!d.io && s_.transport != "tcp" && str == "rdma") {
        const rdma::Api* api = rdma::load(s_.libibverbs, s_.librdmacm, why);
        auto io = api == nullptr ? nullptr : std::make_unique<RdmaIo>(D, S, s_.timeout_ms);
        if (io && io->connect(api, d.host, d.port + s_.rdma_port_offset, d.token, d.ext_bytes, s_.connect_timeout_ms,
                              why)) {
            d.label = "rdma on " + io->device();
            d.io = std::move(io);
        }
    } else if (!d.io && s_.transport != "tcp") {
        why = "the daemon has no RDMA";
    }
    if (!d.io && s_.transport == "rdma") {
        error = ep + ": rdma: " + why;
        return false;
    }
    if (!d.io) {
        if (s_.transport == "auto")
            std::fprintf(stderr, "strata: kvib %s: rdma unavailable (%s); using tcp\n", ep.c_str(), why.c_str());
        std::vector<std::unique_ptr<Channel>> ch;
        for (int k = 0; k < s_.tcp_streams; ++k) {
            auto t = std::make_unique<TcpChannel>();
            if (!t->connect(d.host, d.port, s_.connect_timeout_ms, s_.timeout_ms, error)) return false;
            ch.push_back(std::move(t));
        }
        d.io = std::make_unique<ChannelIo>(std::move(ch), D, S, true, "tcp");
        d.label = "tcp x" + std::to_string(s_.tcp_streams);
    }
#endif
    return true;
}

// A daemon whose connection broke (it restarted, its box rebooted) is dialled again, at most every reconnect_ms; what
// it held is gone, so the names it held are forgotten (ConversationDisk's loads of them miss and drop the entries).
// Each failed dial doubles that daemon's wait (reconnect_ms, 2x, 4x ... up to reconnect_max_ms), so a host that stays
// down costs the caller (the serve thread, for a load) one connect timeout that often.
void FolderImpl::revive() {
    const auto now = std::chrono::steady_clock::now();
    if (busy_) return;
    for (size_t i = 0; i < ds_.size(); ++i) {
        if (ds_[i].ctl->alive() || now < ds_[i].next_dial) continue;
        Daemon d;
        std::string e;
        if (!dial(ds_[i].ep, d, e)) {
            const int k = std::min(ds_[i].fails++, 20);
            const long long wait = std::min<long long>((long long) s_.reconnect_ms << k, s_.reconnect_max_ms);
            ds_[i].next_dial = now + std::chrono::milliseconds(wait);
            if (ds_[i].fails == 1 || wait == s_.reconnect_max_ms)
                std::fprintf(stderr, "strata: kvib %s: cannot dial (%s); next try in %lld ms\n", ds_[i].ep.c_str(),
                             e.c_str(), wait);
            continue;
        }
        std::fprintf(stderr, "strata: kvib %s: reconnected (%s); its conversations are gone\n", d.ep.c_str(),
                     d.label.c_str());
        ds_[i] = std::move(d);
        std::erase_if(holder_, [&](const auto& kv) { return kv.second == i; });
        std::erase_if(read_cache_, [&](const auto& kv) { return kv.second.d == i; });
        std::erase_if(pending_, [&](const auto& kv) { return kv.second.d == i; });
    }
}

bool FolderImpl::prepare(std::string& error) {
    if (!ds_.empty()) return true;   // ConversationDisk::open after the bench or a test prepared it
    std::vector<Daemon> ds;
    size_t at = 0;
    const std::string hosts = local_ != nullptr ? "local" : hosts_;
    while (at <= hosts.size()) {
        const size_t comma = std::min(hosts.find(',', at), hosts.size());
        const std::string ep = hosts.substr(at, comma - at);
        Daemon d;
        if (!ep.empty() && !dial(ep, d, error)) return false;
        if (!ep.empty()) ds.push_back(std::move(d));
        at = comma + 1;
    }
    ds_ = std::move(ds);
    if (local_ != nullptr) return true;
    if (ds_.empty()) {
        error = "kvib: no daemon named (kvib://HOST:PORT[,HOST:PORT...])";
        return false;
    }
    for (const Daemon& d : ds_)
        std::fprintf(stderr, "strata: kvib %s:%d: %s, %.1f GiB arena\n", d.host.c_str(), d.port, d.label.c_str(),
                     (double) d.cap / (1u << 30));
    return true;
}

std::vector<FolderFile> FolderImpl::list() {
    revive();
    std::vector<FolderFile> out;
    holder_.clear();
    const auto now = std::filesystem::file_time_type::clock::now();
    for (size_t i = 0; i < ds_.size(); ++i) {
        Resp r;
        std::vector<uint8_t> d;
        std::string e;
        if (!ds_[i].ctl->simple(kList, 0, 0, 0, {}, r, &d, e)) continue;
        for (size_t at = 0; at + 18 <= d.size();) {
            uint64_t size = 0, age = 0;
            uint16_t len = 0;
            std::memcpy(&size, d.data() + at, 8);
            std::memcpy(&age, d.data() + at + 8, 8);
            std::memcpy(&len, d.data() + at + 16, 2);
            if (at + 18 + len > d.size()) break;
            FolderFile f;
            f.name.assign(reinterpret_cast<const char*>(d.data() + at + 18), len);
            f.bytes = size;
            f.time = now - std::chrono::duration_cast<std::filesystem::file_time_type::duration>(
                               std::chrono::nanoseconds(age));
            holder_[f.name] = i;
            out.push_back(std::move(f));
            at += 18 + len;
        }
    }
    std::sort(out.begin(), out.end(), [](const FolderFile& a, const FolderFile& b) { return a.time < b.time; });
    return out;
}

int FolderImpl::holder_of(const std::string& name) {
    auto it = holder_.find(name);
    if (it == holder_.end()) {
        list();
        it = holder_.find(name);
    }
    return it == holder_.end() ? -1 : (int) it->second;
}

std::unique_ptr<FolderStream> FolderImpl::open_read(const std::string& name) {
    const int d = busy_ ? -1 : holder_of(name);
    if (d < 0) return nullptr;
    Resp r;
    std::vector<uint8_t> data;
    std::string e;
    if (!ds_[(size_t) d].ctl->simple(kOpen, 0, 0, 0, name, r, &data, e)) return nullptr;
    Extents ex;
    ex.parse(data, r.c);
    return std::make_unique<Stream>(*this, (size_t) d, r.a, false, r.b, std::move(ex), name);
}

std::unique_ptr<FolderStream> FolderImpl::open_write(const std::string& name) {
    revive();
    if (busy_ || ds_.empty()) return nullptr;
    // the daemon with the most room
    size_t best = 0;
    uint64_t room = 0;
    for (size_t i = 0; i < ds_.size(); ++i) {
        Resp r;
        std::string e;
        if (!ds_[i].ctl->simple(kStat, 0, 0, 0, {}, r, nullptr, e)) continue;
        const uint64_t free = r.b > r.a ? r.b - r.a : 0;
        if (free > room) {
            room = free;
            best = i;
        }
    }
    Resp r;
    std::string e;
    if (!ds_[best].ctl->simple(kCreate, 0, 0, 0, {}, r, nullptr, e)) return nullptr;
    Extents ex;
    ex.ext_bytes = ds_[best].ext_bytes;
    return std::make_unique<Stream>(*this, best, r.a, true, 0, std::move(ex), name);
}

bool FolderImpl::rename(const std::string& from, const std::string& to) {
    auto it = pending_.find(from);
    if (it == pending_.end()) return false;
    const Pending p = it->second;
    pending_.erase(it);
    Resp r;
    std::string e;
    if (!ds_[p.d].ctl->simple(kCommit, p.h, 0, 0, to, r, nullptr, e)) return false;
    // one name, one daemon: the other daemons' copies of it go
    for (size_t i = 0; i < ds_.size(); ++i)
        if (i != p.d) ds_[i].ctl->simple(kRemove, 0, 0, 0, to, r, nullptr, e);
    holder_[to] = p.d;
    read_cache_.erase(to);
    return true;
}

bool FolderImpl::remove(const std::string& name) {
    Resp r;
    std::string e;
    if (auto it = pending_.find(name); it != pending_.end()) {
        ds_[it->second.d].ctl->simple(kAbort, it->second.h, 0, 0, {}, r, nullptr, e);
        pending_.erase(it);
        return true;
    }
    if (auto it = read_cache_.find(name); it != read_cache_.end()) {
        ds_[it->second.d].ctl->simple(kClose, it->second.h, 0, 0, {}, r, nullptr, e);
        read_cache_.erase(it);
    }
    const int d = holder_of(name);
    holder_.erase(name);
    return d >= 0 && ds_[(size_t) d].ctl->simple(kRemove, 0, 0, 0, name, r, nullptr, e);
}

void FolderImpl::touch(const std::string& name) {
    const int d = holder_of(name);
    Resp r;
    std::string e;
    if (d >= 0) ds_[(size_t) d].ctl->simple(kTouch, 0, 0, 0, name, r, nullptr, e);
}

bool FolderImpl::read_at(const std::string& name, uint64_t offset, void* dst, size_t n, std::string& error) {
    if (busy_) {
        error = "a stream is open";
        return false;
    }
    auto it = read_cache_.find(name);
    if (it == read_cache_.end()) {
        const int d = holder_of(name);
        if (d < 0) {
            error = "no such object";
            return false;
        }
        Resp r;
        std::vector<uint8_t> data;
        if (!ds_[(size_t) d].ctl->simple(kOpen, 0, 0, 0, name, r, &data, error)) return false;
        Opened o{(size_t) d, r.a, r.b, {}};
        o.ex.parse(data, r.c);
        it = read_cache_.emplace(name, std::move(o)).first;
    }
    const Opened& o = it->second;
    if (!ds_[o.d].io) {
        error = "the daemon's data path is down";
        return false;
    }
    Io& io = *ds_[o.d].io;
    if (offset > o.size || n > o.size - offset || n > io.bytes) {
        error = "read past the end or larger than a staging buffer";
        return false;
    }
    if (!io.post(0, false, o.h, offset, n, o.ex, error) || !io.wait(0, error)) {
        if (io.broken()) drop(o.d);
        return false;
    }
    std::memcpy(dst, io.stage(0), n);
    return true;
}

bool FolderImpl::usage(uint64_t& used, uint64_t& cap, std::string& error) {
    used = cap = 0;
    for (Daemon& d : ds_) {
        Resp r;
        if (!d.ctl->simple(kStat, 0, 0, 0, {}, r, nullptr, error)) return false;
        used += r.a;
        cap += r.b;
    }
    return true;
}

// a folder whose prepare() reports why it could not be made
class Refused final : public Folder {
public:
    explicit Refused(std::string why) : why_(std::move(why)) {}
    bool prepare(std::string& error) override { return error = why_, false; }
    std::vector<FolderFile> list() override { return {}; }
    std::unique_ptr<FolderStream> open_read(const std::string&) override { return nullptr; }
    std::unique_ptr<FolderStream> open_write(const std::string&) override { return nullptr; }
    bool rename(const std::string&, const std::string&) override { return false; }
    bool remove(const std::string&) override { return false; }
    void touch(const std::string&) override {}
    std::string where() const override { return "kvib"; }
    std::string transport() const override { return {}; }
    bool read_at(const std::string&, uint64_t, void*, size_t, std::string& error) override { return error = why_, false; }
    bool usage(uint64_t&, uint64_t&, std::string& error) override { return error = why_, false; }

private:
    std::string why_;
};
} // namespace

std::unique_ptr<Folder> folder(const std::string& hosts, const Settings& s) {
    return std::make_unique<FolderImpl>(hosts, s, nullptr);
}

std::unique_ptr<Folder> folder(const std::string& hosts) {
    Settings s;
    std::string why;
    if (!settings_from_env(s, why)) return std::make_unique<Refused>(why);
    return folder(hosts, s);
}

std::unique_ptr<Folder> folder_local(Server& server, const Settings& s) {
    return std::make_unique<FolderImpl>("", s, &server);
}

} // namespace strata::core::kvib
