// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// homelab kvib (kvib.hpp): the store's objects over the in-process transport and over loopback TCP - odd sizes, the
// header rewrite after a seek back, replace, abort, remove, a full arena, block reads - and --conversation-save's
// files: the bytes a parked conversation leaves in a kvib store are the bytes it leaves in a folder, and it loads back
// equal.  A daemon that goes away is a miss, never an error that stops the engine; bad settings are refused by name.
#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_park_queue.hpp"
#include "strata/core/conversation_prefetch.hpp"
#include "strata/core/kvib.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <mutex>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        std::exit(1);
    }
}

std::vector<uint8_t> pattern(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = (uint8_t) rng();
    return v;
}

bool put(ConversationFolder& f, const std::string& name, const std::vector<uint8_t>& bytes, size_t piece) {
    auto w = f.open_write(name + ".tmp");
    if (!w) return false;
    for (size_t at = 0; at < bytes.size(); at += piece)
        if (!w->write(bytes.data() + at, std::min(piece, bytes.size() - at))) return false;
    return w->finish() && (w.reset(), f.rename(name + ".tmp", name));
}

std::vector<uint8_t> get(ConversationFolder& f, const std::string& name) {
    auto r = f.open_read(name);
    std::vector<uint8_t> out;
    if (!r) return out;
    uint8_t buf[70000];
    for (size_t k; (k = r->read(buf, sizeof buf)) > 0;) out.insert(out.end(), buf, buf + k);
    return out;
}

uint64_t arena_used(kvib::Folder& f) {
    uint64_t used = 0, cap = 0;
    std::string e;
    f.usage(used, cap, e);
    return used;
}

// the object semantics, whatever the transport
void objects(kvib::Folder& f, const std::string& via) {
    const uint64_t base = arena_used(f);
    // an odd size over several extents (1 MiB) and staging buffers, written in odd pieces
    const auto a = pattern((size_t{7} << 19) + 17, 1);
    check(put(f, "a.xsc", a, 12345), via + ": a 3.5 MiB + 17 B object is written and committed");
    check(get(f, "a.xsc") == a, via + ": it reads back byte for byte");
    bool listed = false;
    for (const auto& file : f.list()) listed = listed || (file.name == "a.xsc" && file.bytes == a.size());
    check(listed, via + ": list() names it with its size");
    // a writer that goes back to put its header in, as ConversationDisk::save does
    {
        auto b = pattern((size_t{3} << 20) + 5, 2);
        auto w = f.open_write("b.tmp");
        check(w && w->write(b.data(), b.size()), via + ": a second object is written");
        check(w->tell() == (int64_t) b.size(), via + ": tell() is the length");
        const auto head = pattern(64, 3);
        check(w->seek(0) && w->write(head.data(), head.size()), via + ": seek back and rewrite the header");
        check(w->finish(), via + ": finish");
        w.reset();
        check(f.rename("b.tmp", "b.xsc"), via + ": commit");
        std::memcpy(b.data(), head.data(), head.size());
        check(get(f, "b.xsc") == b, via + ": the rewritten header and the body read back");
    }
    // a commit under an existing name replaces it
    const auto a2 = pattern(1000, 4);
    check(put(f, "a.xsc", a2, 1000), via + ": a.xsc is replaced");
    check(get(f, "a.xsc") == a2, via + ": the new bytes are read");
    check(!f.open_read("missing.xsc"), via + ": a missing object does not open");
    // block reads on demand
    {
        const auto b = get(f, "b.xsc");
        std::vector<uint8_t> blk(3264);
        std::string e;
        const uint64_t at = (uint64_t{1} << 20) - 1000;   // across an extent boundary
        check(f.read_at("b.xsc", at, blk.data(), blk.size(), e) && std::memcmp(blk.data(), b.data() + at, blk.size()) == 0,
              via + ": read_at across an extent boundary");
        check(!f.read_at("b.xsc", b.size() - 10, blk.data(), blk.size(), e), via + ": read_at past the end is refused");
    }
    // an abandoned write leaves nothing behind
    const uint64_t before = arena_used(f);
    {
        auto w = f.open_write("c.tmp");
        const auto c = pattern(size_t{5} << 20, 5);
        check(w && w->write(c.data(), c.size()), via + ": a write to abandon");
    }
    check(arena_used(f) == before, via + ": an abandoned write gives its extents back");
    check(!f.rename("c.tmp", "c.xsc"), via + ": an abandoned write cannot be committed");
    // a finished write that is removed instead of committed (ConversationDisk's failure path)
    {
        const auto c = pattern(size_t{2} << 20, 6);
        auto w = f.open_write("d.tmp");
        check(w && w->write(c.data(), c.size()) && w->finish(), via + ": a finished write");
        w.reset();
        check(f.remove("d.tmp") && arena_used(f) == before, via + ": removing it before the commit frees it");
    }
    // a full arena (64 MiB): the write fails, and gives back what it took
    {
        const auto big = pattern(size_t{1} << 20, 7);
        auto w = f.open_write("e.tmp");
        bool ok = w != nullptr;
        for (int i = 0; ok && i < 80; ++i) ok = w->write(big.data(), big.size());
        ok = ok && w->finish();
        check(!ok, via + ": 80 MiB does not fit a 64 MiB arena");
        w.reset();
        check(arena_used(f) == before, via + ": the failed write gave its extents back");
        check(get(f, "b.xsc").size() == (size_t{3} << 20) + 5, via + ": the other objects are intact");
    }
    check(f.remove("a.xsc") && f.remove("b.xsc"), via + ": remove");
    check(!f.remove("a.xsc"), via + ": a second remove finds nothing");
    check(arena_used(f) == base, via + ": the arena is back where it started");
    check(f.list().empty(), via + ": the store is empty");
}

SavedConversation conversation(uint64_t seed) {
    std::mt19937_64 rng(seed);
    SavedConversation s;
    s.geometry[0] = 1;
    s.live.ids.resize(5000);
    for (auto& id : s.live.ids) id = (int32_t) (rng() % 150000);
    auto bytes = [&](size_t n) { return pattern(n, rng()); };
    s.live.gdn = bytes(size_t{1} << 20);
    s.live.tails = bytes(999);
    ConversationCheckpoint half;
    half.ids.assign(s.live.ids.begin(), s.live.ids.begin() + 2500);
    half.gdn = bytes(size_t{1} << 20);
    s.checkpoints.push_back(half);
    s.kv.resize(3);
    for (auto& kv : s.kv) {
        kv.format = strata::kernels::kKvHybrid;
        kv.cells = 5000;
        kv.heads = 2;
        kv.head_dim = 512;
        kv.page_size = 4;
        for (ConversationBuffer* b : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale}) {
            const auto v = bytes((size_t{3} << 20) / 2 + 7);
            b->resize(v.size());
            b->visit(0, v.size(), [&](uint8_t* p, size_t n, size_t at) { std::memcpy(p, v.data() + at, n); return true; });
        }
    }
    return s;
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.live.ids != b.live.ids || a.live.gdn != b.live.gdn || a.live.tails != b.live.tails ||
        a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (a.checkpoints[i].ids != b.checkpoints[i].ids || a.checkpoints[i].gdn != b.checkpoints[i].gdn) return false;
    for (size_t i = 0; i < a.kv.size(); ++i)
        if (!(a.kv[i].k == b.kv[i].k) || !(a.kv[i].v == b.kv[i].v) || !(a.kv[i].k_scale == b.kv[i].k_scale) ||
            !(a.kv[i].v_scale == b.kv[i].v_scale) || a.kv[i].cells != b.kv[i].cells)
            return false;
    return true;
}

ConversationDiskOptions disk_options(std::shared_ptr<ConversationFolder> folder) {
    ConversationDiskOptions o;
    o.folder = std::move(folder);
    o.fingerprint = "conversation_kvib_test";
    o.budget = size_t{48} << 20;
    return o;
}

// the same conversation through ConversationDisk into a folder and into the store: the same file, loaded back equal
void files(kvib::Folder& store, const fs::path& dir, const std::string& via) {
    auto local = std::shared_ptr<ConversationFolder>(conversation_folder_local(dir));
    std::shared_ptr<ConversationFolder> remote(&store, [](ConversationFolder*) {});
    ConversationDisk on_disk, in_ram;
    std::string e;
    check(on_disk.open(disk_options(local), e), via + ": a folder opens: " + e);
    check(in_ram.open(disk_options(remote), e), via + ": the store opens: " + e);
    const SavedConversation s = conversation(11);
    size_t w1 = 0, w2 = 0;
    check(on_disk.save(s, w1, e) && in_ram.save(s, w2, e), via + ": the conversation is parked in both: " + e);
    check(w1 == w2 && w1 > 0, via + ": the same length");
    std::string name;
    for (const auto& f : local->list()) name = f.name;
    check(!name.empty() && get(*local, name) == get(*remote, name), via + ": the store holds the folder's bytes");
    // a fresh disk over the store (a restarted engine) finds it and loads it back
    ConversationDisk again;
    check(again.open(disk_options(remote), e) && again.size() == 1, via + ": a restarted engine indexes it");
    std::vector<int32_t> prompt = s.live.ids;
    prompt.push_back(7);
    const auto m = again.best(prompt, {}, true);
    check(m.tokens == (int64_t) s.live.ids.size() && m.live, via + ": the whole conversation matches the next turn");
    SavedConversation back;
    check(again.load(m.index, back, e), via + ": it loads: " + e);
    check(same(s, back), via + ": it loads back equal");
    // a damaged object is a miss: deleted, and the prompt is read the ordinary way
    {
        auto bytes = get(*remote, name);
        bytes[bytes.size() / 2] ^= 1;
        check(put(*remote, name, bytes, bytes.size()), via + ": damage it");
        ConversationDisk d;
        check(d.open(disk_options(remote), e) && d.size() == 1, via + ": the damaged one is indexed (the header is good)");
        SavedConversation x;
        check(!d.load(0, x, e) && e.find("checksum") != std::string::npos, via + ": the damage is caught: " + e);
        check(remote->list().empty(), via + ": and the object deleted");
    }
}

// ---- the wire as kvib.cpp lays it out (a request: header, name, data; a response: header, text, data), for requests
// no client would send
struct WireReq {
    uint32_t magic = 0x4b564942;
    uint16_t op = 0, flags = 0;
    uint64_t a = 0, b = 0, c = 0;
    uint32_t name_len = 0, data_len = 0;
};
struct WireResp {
    uint32_t magic = 0;
    int32_t status = 0;
    uint64_t a = 0, b = 0, c = 0;
    uint32_t str_len = 0, data_len = 0;
};
static_assert(sizeof(WireReq) == 40 && sizeof(WireResp) == 40, "the wire's headers");

int dial(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t) port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd >= 0 && ::connect(fd, (sockaddr*) &a, sizeof a) == 0) return fd;
    if (fd >= 0) ::close(fd);
    return -1;
}

bool io_all(int fd, void* p, size_t n, bool out) {
    auto* b = static_cast<uint8_t*>(p);
    while (n > 0) {
        const ssize_t k = out ? ::send(fd, b, n, MSG_NOSIGNAL) : ::recv(fd, b, n, 0);
        if (k <= 0) return false;
        b += k;
        n -= (size_t) k;
    }
    return true;
}

// one raw request; false when the connection broke
bool raw(int fd, WireReq q, const std::vector<uint8_t>& data, WireResp& r) {
    q.data_len = (uint32_t) data.size();
    std::vector<uint8_t> d = data;
    if (!io_all(fd, &q, sizeof q, true) || (!d.empty() && !io_all(fd, d.data(), d.size(), true)) ||
        !io_all(fd, &r, sizeof r, false))
        return false;
    std::vector<uint8_t> rest(r.str_len + (size_t) r.data_len);
    return rest.empty() || io_all(fd, rest.data(), rest.size(), false);
}

// A loopback TCP forwarder in front of a daemon, which can cut the connections it carries from the k-th on: the
// control connection is a folder's first, its data connections follow, so cutting from 1 breaks only the data path.
struct Proxy {
    int lfd = -1, port = 0, upstream = 0;
    std::atomic<bool> stop{false};
    std::mutex m;
    std::vector<std::pair<int, int>> pairs;
    std::vector<std::thread> threads;
    bool start(int up) {
        upstream = up;
        lfd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t len = sizeof a;
        if (lfd < 0 || ::bind(lfd, (sockaddr*) &a, sizeof a) != 0 || ::listen(lfd, 16) != 0 ||
            ::getsockname(lfd, (sockaddr*) &a, &len) != 0)
            return false;
        port = ntohs(a.sin_port);
        threads.emplace_back([this] {
            while (!stop) {
                pollfd p{lfd, POLLIN, 0};
                if (::poll(&p, 1, 50) <= 0) continue;
                const int c = ::accept(lfd, nullptr, nullptr);
                if (c < 0) continue;
                const int u = dial(upstream);
                std::lock_guard<std::mutex> lock(m);
                pairs.emplace_back(c, u);
                if (u < 0) {
                    ::shutdown(c, SHUT_RDWR);
                    continue;
                }
                for (const auto& [from, to] : {std::make_pair(c, u), std::make_pair(u, c)})
                    threads.emplace_back([from, to] {
                        uint8_t buf[65536];
                        for (ssize_t k; (k = ::recv(from, buf, sizeof buf, 0)) > 0;)
                            if (!io_all(to, buf, (size_t) k, true)) break;
                        ::shutdown(from, SHUT_RDWR);
                        ::shutdown(to, SHUT_RDWR);
                    });
            }
        });
        return true;
    }
    void cut(size_t from, size_t to) {
        std::lock_guard<std::mutex> lock(m);
        for (size_t k = from; k < to && k < pairs.size(); ++k) {
            ::shutdown(pairs[k].first, SHUT_RDWR);
            if (pairs[k].second >= 0) ::shutdown(pairs[k].second, SHUT_RDWR);
        }
    }
    size_t carried() {
        std::lock_guard<std::mutex> lock(m);
        return pairs.size();
    }
    ~Proxy() {
        stop = true;
        cut(0, SIZE_MAX);
        threads[0].join();   // the accept thread: the only one that adds threads
        for (size_t k = 1; k < threads.size(); ++k) threads[k].join();
        for (const auto& [c, u] : pairs) {
            ::close(c);
            if (u >= 0) ::close(u);
        }
        if (lfd >= 0) ::close(lfd);
    }
};

// ---- homelab kvib async: the faster format path and the background park

// a conversation whose K/V buffers span several 16 MiB segments (the copy threads split their reads and writes)
SavedConversation big_conversation(uint64_t seed) {
    SavedConversation s = conversation(seed);
    std::mt19937_64 rng(seed * 31 + 7);
    for (auto& kv : s.kv)
        for (ConversationBuffer* b : {&kv.k, &kv.v}) {
            const size_t n = (size_t{37} << 20) / 2 + 333;   // 18.5 MiB + 333 B: two segments, an odd tail
            b->resize(n);
            b->visit(0, n, [&](uint8_t* p, size_t k, size_t) {
                for (size_t i = 0; i < k; ++i) p[i] = (uint8_t) rng();
                return true;
            });
        }
    return s;
}

std::vector<uint8_t> only_file(ConversationFolder& f) {
    std::string name;
    for (const auto& x : f.list())
        if (x.name.size() > 4 && x.name.compare(x.name.size() - 4, 4, ".xsc") == 0) name = x.name;
    return name.empty() ? std::vector<uint8_t>{} : get(f, name);
}

// the copy threads: every size copies every byte (a split whose pieces fall 2 bytes short of n once cost the last two
// bytes of a 15 MiB copy, measured on droopy), and a large read from a local file does too
void io_copies(const fs::path& dir) {
    const auto src = pattern((size_t{40} << 20) + 4099, 77);
    std::vector<size_t> sizes = {0, 1, 63, 65, 4099, (size_t{1} << 20) + 7, (size_t{2} << 20) - 1, size_t{2} << 20,
                                 (size_t{2} << 20) + 1, (size_t{3} << 20) + 1, 15001090, 15001088,
                                 (size_t{16} << 20) - 2, size_t{37} << 20, src.size()};
    for (size_t r = 1; r < 64; ++r) sizes.push_back((size_t{4} << 20) * 3 + r * 4);   // n % pieces != 0 with n / pieces % 64 == 0
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto f = conversation_folder_local(dir);
    check(put(*f, "odd.bin", src, src.size()), "a 40 MiB + 4099 B file");
    // every thread count's split: fewer pieces than threads (a 2-3 MiB copy over 64 threads), odd piece counts
    for (const int t : {1, 2, 3, 4, 5, 7, 8, 13, 64}) {
        conversation_io_set_threads(t);
        bool all = true;
        for (const size_t n : sizes) {
            std::vector<uint8_t> dst(n + 64, 0xee);
            conversation_io_copy(dst.data() + 3, src.data() + 1, n);
            all = all && std::memcmp(dst.data() + 3, src.data() + 1, n) == 0 && dst[2] == 0xee && dst[n + 3] == 0xee;
        }
        check(all, std::to_string(t) + " copy threads: " + std::to_string(sizes.size()) +
                       " sizes copy exactly their bytes");
        auto r = f->open_read("odd.bin");
        std::vector<uint8_t> back(src.size());
        size_t at = 0;
        for (const size_t n : {size_t{5}, (size_t{12} << 20) + 6, size_t{65536}, (size_t{4} << 20) * 3 + 8,
                               (size_t{2} << 20) + 1}) {
            at += r->read(back.data() + at, n);
        }
        at += r->read(back.data() + at, back.size() - at);
        check(at == src.size() && back == src && r->read(back.data(), 1) == 0,
              std::to_string(t) + " copy threads: large local reads (preads) at odd offsets read every byte, then the end");
    }
    conversation_io_set_threads(ConversationIoSettings::kThreadsDefault);
}

// the same file and the same checksum whatever the copy thread count; a file from one count loads under the other
void format_path(kvib::Folder& store, const fs::path& dir) {
    const SavedConversation s = big_conversation(31);
    std::vector<std::vector<uint8_t>> files;
    for (const int t : {1, 4, 3, 64}) {
        setenv("STRATA_CONV_IO_THREADS", std::to_string(t).c_str(), 1);
        const fs::path d = dir / ("threads" + std::to_string(t));
        auto local = std::shared_ptr<ConversationFolder>(conversation_folder_local(d));
        ConversationDiskOptions o = disk_options(local);
        o.budget = size_t{1} << 30;
        ConversationDisk disk;
        std::string e;
        check(disk.open(o, e) && conversation_io_threads() == t, "io threads " + std::to_string(t) + ": open: " + e);
        size_t w = 0;
        check(disk.save(s, w, e), "io threads " + std::to_string(t) + ": a 113 MiB conversation is written: " + e);
        files.push_back(only_file(*local));
        SavedConversation back;
        check(disk.load(0, back, e) && same(s, back), "io threads " + std::to_string(t) + ": it loads back equal: " + e);
        // through the store too (parallel copies in and out of 4 MiB staging buffers)
        std::shared_ptr<ConversationFolder> remote(&store, [](ConversationFolder*) {});
        ConversationDisk in_ram;
        check(in_ram.open([&] { ConversationDiskOptions r = disk_options(remote); r.budget = size_t{1} << 30; return r; }(), e),
              "the store opens: " + e);
        check(in_ram.save(s, w, e), "io threads " + std::to_string(t) + ": parked in the store: " + e);
        check(only_file(*remote) == files.back(), "io threads " + std::to_string(t) + ": the store holds the folder's bytes");
        SavedConversation r2;
        check(in_ram.load(0, r2, e) && same(s, r2), "io threads " + std::to_string(t) + ": loads back equal from the store");
        for (const auto& x : remote->list()) remote->remove(x.name);
    }
    check(!files[0].empty() && files[0] == files[1], "1 and 4 copy threads write the same bytes (the same checksum)");
    check(files[2] == files[0] && files[3] == files[0], "3 and 64 copy threads write the same bytes too");
    // a file written with one thread, read with four, damaged: still caught
    {
        setenv("STRATA_CONV_IO_THREADS", "4", 1);
        auto local = std::shared_ptr<ConversationFolder>(conversation_folder_local(dir / "threads1"));
        auto bytes = files[0];
        bytes[bytes.size() - (size_t{5} << 20)] ^= 4;
        std::string name;
        for (const auto& x : local->list()) name = x.name;
        check(put(*local, name, bytes, bytes.size()), "damage the 1-thread file");
        ConversationDiskOptions o = disk_options(local);
        o.budget = size_t{1} << 30;
        ConversationDisk d;
        std::string e;
        SavedConversation x;
        check(d.open(o, e) && d.size() == 1 && !d.load(0, x, e) && e.find("checksum") != std::string::npos,
              "4 threads catch a flipped byte deep in the body: " + e);
    }
    {   // review of 9350e89 (b): the checksum queue's copied bytes are bounded; the smallest bound gives the same file
        setenv("STRATA_CONV_HASH_QUEUE_MIB", "1", 1);
        auto local = std::shared_ptr<ConversationFolder>(conversation_folder_local(dir / "hashq"));
        ConversationDiskOptions o = disk_options(local);
        o.budget = size_t{1} << 30;
        o.compress = conversation_disk_can_compress();   // compressed chunks are the copied runs
        ConversationDisk d;
        std::string e;
        size_t w = 0;
        SavedConversation back;
        check(d.open(o, e) && conversation_io_hash_queue() == (size_t{1} << 20) && d.save(s, w, e) &&
                  d.load(0, back, e) && same(s, back),
              "STRATA_CONV_HASH_QUEUE_MIB=1: a 113 MiB conversation is written and read back equal: " + e);
        if (!o.compress) check(only_file(*local) == files[0], "STRATA_CONV_HASH_QUEUE_MIB=1: the same file bytes");
        for (const char* bad : {"0", "4097", "x"}) {
            setenv("STRATA_CONV_HASH_QUEUE_MIB", bad, 1);
            ConversationDisk r;
            check(!r.open(disk_options(local), e) && e.find("STRATA_CONV_HASH_QUEUE_MIB") != std::string::npos,
                  std::string("STRATA_CONV_HASH_QUEUE_MIB=") + bad + " is refused by name: " + e);
        }
        unsetenv("STRATA_CONV_HASH_QUEUE_MIB");
        check(d.open(disk_options(local), e) && conversation_io_hash_queue() == (size_t{64} << 20),
              "unset: the 64 MiB default");
    }
    setenv("STRATA_CONV_IO_THREADS", "0", 1);
    {
        ConversationDisk d;
        std::string e;
        check(!d.open(disk_options(std::shared_ptr<ConversationFolder>(conversation_folder_local(dir / "bad"))), e) &&
                  e.find("STRATA_CONV_IO_THREADS") != std::string::npos, "STRATA_CONV_IO_THREADS=0 is refused by name: " + e);
    }
    unsetenv("STRATA_CONV_IO_THREADS");
}

// A folder whose writes wait at a gate (a slow store), or fail at once.
struct Gate {
    std::mutex m;
    std::condition_variable cv;
    bool open = true;
    int entered = 0;
    void close() { std::lock_guard<std::mutex> l(m); open = false; }
    void release() { { std::lock_guard<std::mutex> l(m); open = true; } cv.notify_all(); }
    int now() { std::lock_guard<std::mutex> l(m); return entered; }
    bool wait_entered(int n) {
        for (int i = 0; i < 500; ++i) {
            { std::lock_guard<std::mutex> l(m); if (entered >= n) return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    }
};
class GateStream final : public FolderStream {
public:
    GateStream(std::unique_ptr<FolderStream> s, Gate& g) : s_(std::move(s)), g_(g) {}
    size_t read(void* p, size_t n) override { return s_->read(p, n); }
    bool write(const void* p, size_t n) override {
        if (!first_) {
            first_ = true;
            std::unique_lock<std::mutex> l(g_.m);
            ++g_.entered;
            g_.cv.wait(l, [&] { return g_.open; });
        }
        return s_->write(p, n);
    }
    bool seek(uint64_t o) override { return s_->seek(o); }
    int64_t tell() override { return s_->tell(); }
    bool failed() const override { return s_->failed(); }
    bool finish() override { return s_->finish(); }
private:
    std::unique_ptr<FolderStream> s_;
    Gate& g_;
    bool first_ = false;
};
class GateFolder final : public ConversationFolder {
public:
    GateFolder(std::shared_ptr<ConversationFolder> f, Gate& g, bool fail) : f_(std::move(f)), g_(g), fail_(fail) {}
    bool prepare(std::string& e) override { return f_->prepare(e); }
    std::vector<FolderFile> list() override { return f_->list(); }
    std::unique_ptr<FolderStream> open_read(const std::string& n) override { return f_->open_read(n); }
    std::unique_ptr<FolderStream> open_write(const std::string& n) override {
        if (fail_) return nullptr;
        auto s = f_->open_write(n);
        return s ? std::make_unique<GateStream>(std::move(s), g_) : nullptr;
    }
    bool rename(const std::string& a, const std::string& b) override { return f_->rename(a, b); }
    bool remove(const std::string& n) override { return f_->remove(n); }
    void touch(const std::string& n) override { f_->touch(n); }
    std::string where() const override { return "gate:" + f_->where(); }
private:
    std::shared_ptr<ConversationFolder> f_;
    Gate& g_;
    bool fail_;
};

size_t xsc_files(ConversationFolder& f) {
    size_t n = 0;
    for (const auto& x : f.list()) n += x.name.size() > 4 && x.name.compare(x.name.size() - 4, 4, ".xsc") == 0;
    return n;
}

// the background park: a write behind the decode, its image taken back by the conversation's next turn, a full queue,
// a failed write - over a folder and over the in-process store
void park_queue(std::shared_ptr<ConversationFolder> main, std::function<std::shared_ptr<ConversationFolder>()> second,
                const std::string& via) {
    Gate gate;
    std::atomic<bool> fail{false};
    ConversationDiskOptions o = disk_options(main);
    o.budget = size_t{256} << 20;
    o.folder_factory = [&]() -> std::shared_ptr<ConversationFolder> {
        return std::make_shared<GateFolder>(second(), gate, fail.load());
    };
    ConversationDisk disk;
    std::string e;
    check(disk.open(o, e), via + ": park queue: the folder opens: " + e);
    auto make_queue = [&](size_t jobs, size_t bytes) {
        auto f = disk.open_writer_folder(e);
        check(f != nullptr, via + ": a second connection for the writer: " + e);
        return std::make_unique<ConversationParkQueue>(disk.options(), f, jobs, bytes);
    };
    auto next_turn = [](const SavedConversation& s) {
        std::vector<int32_t> p = s.live.ids;
        p.push_back(7);
        return p;
    };
    auto adopt_all = [&](ConversationParkQueue& q, size_t& ok, size_t& bad) {
        for (auto& d : q.done()) {
            if (d.ok) { disk.adopt(d.file); ++ok; } else { ++bad; }
        }
    };
    {   // written behind: done() hands the file to the index, the next turn loads it back equal
        auto q = make_queue(2, size_t{1} << 30);
        const SavedConversation s = conversation(40);
        double waited = 0;
        q->push(SavedConversation(s), waited);
        check(waited < 1000, via + ": an empty queue takes the image at once");
        q->drain();
        size_t ok = 0, bad = 0;
        adopt_all(*q, ok, bad);
        check(ok == 1 && bad == 0 && q->jobs() == 0 && q->bytes() == 0, via + ": the write finished and was adopted");
        const auto m = disk.best(next_turn(s), {}, true);
        SavedConversation back;
        check(m.tokens == (int64_t) s.live.ids.size() && m.live && disk.load(m.index, back, e) && same(s, back),
              via + ": the next turn finds the background-written file and loads it back equal: " + e);
    }
    {   // in flight when the next turn arrives: the image itself comes back, nothing is committed
        auto q = make_queue(2, size_t{1} << 30);
        const SavedConversation s = conversation(41);
        const size_t files_before = xsc_files(*main);
        gate.close();
        double waited = 0;
        const int e0 = gate.now();
        const uint64_t id = q->push(SavedConversation(s), waited);
        check(gate.wait_entered(e0 + 1), via + ": the writer is mid-write");
        const auto m = q->best(next_turn(s), {}, true);
        check(m.id == id && m.live && m.tokens == (int64_t) s.live.ids.size(), via + ": the next turn matches it");
        check(disk.best(next_turn(s), {}, true).tokens == 0, via + ": no file of it is visible meanwhile");
        std::thread opener([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            gate.release();
        });
        auto img = q->claim(id);
        opener.join();
        check(img && same(s, *img), via + ": the claimed image is the parked one, byte for byte");
        q->drain();
        size_t ok = 0, bad = 0;
        adopt_all(*q, ok, bad);
        check(ok == 0 && bad == 0, via + ": the cancelled write is neither adopted nor a failure");
        check(xsc_files(*main) == files_before, via + ": and left no file (a partial one is never committed)");
        // a sibling that shares only a shallow checkpoint matches it, but not as this chat's own turn
        SavedConversation t = conversation(42);
        ConversationCheckpoint root;
        root.ids.assign(t.live.ids.begin(), t.live.ids.begin() + 100);
        root.gdn = pattern(1000, 5);
        t.checkpoints.insert(t.checkpoints.begin(), root);
        gate.close();
        const int e1 = gate.now();
        const uint64_t id2 = q->push(SavedConversation(t), waited);
        check(gate.wait_entered(e1 + 1), via + ": a second write is under way");
        std::vector<int32_t> sib(t.live.ids.begin(), t.live.ids.begin() + 100);
        sib.push_back(99);
        sib.push_back(98);
        const auto ms = q->best(sib, {}, true);
        check(ms.id == id2 && ms.tokens == 100 && !ms.live && !ms.deepest,
              via + ": a sibling's match on the shared start is neither live nor the deepest (not claimed)");
        // queued behind it: taken back without being written
        const SavedConversation u = conversation(43);
        const uint64_t id3 = q->push(SavedConversation(u), waited);
        check(q->jobs() == 2, via + ": two jobs held");
        auto img3 = q->claim(id3);
        check(img3 && same(u, *img3) && q->jobs() == 1, via + ": a queued image is taken back at once, unwritten");
        // a full queue (1 job, the one in flight): the next push waits for it
        std::atomic<bool> pushed{false};
        double waited4 = 0;
        auto q1 = std::move(q);
        std::thread pusher([&] {
            // max_jobs is 2 here: fill it, then the third waits
            q1->push(conversation(44), waited4);
            q1->push(conversation(45), waited4);
            pushed = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        check(!pushed.load(), via + ": a push into a full queue waits");
        gate.release();
        pusher.join();
        check(pushed.load() && waited4 > 100, via + ": and goes on once a write finishes (waited " +
                                                  std::to_string((int) waited4) + " ms)");
        q1->drain();
        size_t ok2 = 0, bad2 = 0;
        adopt_all(*q1, ok2, bad2);
        check(ok2 == 3 && bad2 == 0, via + ": the in-flight and the two queued writes landed");
    }
    {   // a byte bound: an image over it is still taken when the queue is empty, the next one waits
        gate.close();
        auto q = make_queue(8, 1);
        double waited = 0;
        const int e2 = gate.now();
        const size_t files_before = xsc_files(*main);
        q->push(conversation(46), waited);
        check(gate.wait_entered(e2 + 1), via + ": one image over the byte bound is taken into an empty queue");
        std::atomic<bool> pushed{false};
        std::thread pusher([&] { q->push(conversation(47), waited); pushed = true; });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        check(!pushed.load(), via + ": a second one waits for the byte bound");
        gate.release();
        pusher.join();
        q.reset();   // the destructor finishes every queued write
        check(xsc_files(*main) == files_before + 2, via + ": the destructor finished both writes");
    }
    {   // a writer that cannot write: reported, the image freed, the next turn finds no file
        fail = true;
        auto q = make_queue(2, size_t{1} << 30);
        const SavedConversation s = conversation(48);
        double waited = 0;
        q->push(SavedConversation(s), waited);
        q->drain();
        size_t ok = 0, bad = 0;
        adopt_all(*q, ok, bad);
        check(ok == 0 && bad == 1 && q->bytes() == 0, via + ": a failed write is reported and its image freed");
        check(disk.best(next_turn(s), {}, true).tokens == 0, via + ": the next turn finds no file (reads its history)");
        fail = false;
    }
}

// A folder whose reads can wait at a gate or fail part way (a slow or dying store under the prefetch's loader).
struct ReadHook {
    Gate gate;
    std::atomic<long long> fail_after{-1};   // >= 0: a stream's reads fail once it has read this many bytes
};
class HookStream final : public FolderStream {
public:
    HookStream(std::unique_ptr<FolderStream> s, ReadHook& h) : s_(std::move(s)), h_(h) {}
    size_t read(void* p, size_t n) override {
        if (!first_) {
            first_ = true;
            std::unique_lock<std::mutex> l(h_.gate.m);
            ++h_.gate.entered;
            h_.gate.cv.wait(l, [&] { return h_.gate.open; });
        }
        const long long cut = h_.fail_after.load();
        if (cut >= 0 && read_ + (long long) n > cut) {
            failed_ = true;
            return 0;
        }
        const size_t k = s_->read(p, n);
        read_ += (long long) k;
        return k;
    }
    bool write(const void* p, size_t n) override { return s_->write(p, n); }
    bool seek(uint64_t o) override { return s_->seek(o); }
    int64_t tell() override { return s_->tell(); }
    bool failed() const override { return failed_ || s_->failed(); }
    bool finish() override { return s_->finish(); }
private:
    std::unique_ptr<FolderStream> s_;
    ReadHook& h_;
    bool first_ = false, failed_ = false;
    long long read_ = 0;
};
class HookFolder final : public ConversationFolder {
public:
    HookFolder(std::shared_ptr<ConversationFolder> f, ReadHook& h) : f_(std::move(f)), h_(h) {}
    bool prepare(std::string& e) override { return f_->prepare(e); }
    std::vector<FolderFile> list() override { return f_->list(); }
    std::unique_ptr<FolderStream> open_read(const std::string& n) override {
        auto s = f_->open_read(n);
        return s ? std::make_unique<HookStream>(std::move(s), h_) : nullptr;
    }
    std::unique_ptr<FolderStream> open_write(const std::string& n) override { return f_->open_write(n); }
    bool rename(const std::string& a, const std::string& b) override { return f_->rename(a, b); }
    bool remove(const std::string& n) override { return f_->remove(n); }
    void touch(const std::string& n) override { f_->touch(n); }
    std::string where() const override { return "hook:" + f_->where(); }
    size_t staging_bytes() const override { return f_->staging_bytes(); }
private:
    std::shared_ptr<ConversationFolder> f_;
    ReadHook& h_;
};

std::string hint_text(const std::vector<int32_t>& ids, const char* keys = "") {
    std::string t = keys;
    for (size_t i = 0; i < ids.size(); ++i) t += (i ? "," : "") + std::to_string(ids[i]);
    return t;
}

template<class F> bool eventually(F done, int ms = 5000) {
    for (int i = 0; i < ms / 5; ++i) {
        if (done()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return done();
}

// homelab restore prefetch: a parked conversation's file read ahead on the loader thread, through its own connection
// - hit, miss, an image still in the background park, a newer write of the same name, the timeout, the bounds, a read
// that fails part way, the serve thread's own request, a take while the read runs, a drop - over a folder and a store
void prefetch(std::shared_ptr<ConversationFolder> main, std::function<std::shared_ptr<ConversationFolder>()> second,
              const std::string& via) {
    ReadHook hook;
    ConversationDiskOptions o = disk_options(main);
    o.budget = size_t{256} << 20;
    o.folder_factory = [&]() -> std::shared_ptr<ConversationFolder> { return std::make_shared<HookFolder>(second(), hook); };
    ConversationDisk disk;
    std::string e;
    check(disk.open(o, e), via + ": prefetch: the folder opens: " + e);
    for (const auto& f : main->list()) main->remove(f.name);
    check(disk.open(o, e) && disk.size() == 0, via + ": prefetch: an empty folder");
    std::unique_ptr<ConversationParkQueue> park;   // set for the race below
    auto make = [&](size_t jobs, size_t bytes, int timeout_ms) {
        ConversationPrefetch::Options po;
        po.disk = disk.options();
        po.folder = disk.open_writer_folder(e);
        check(po.folder != nullptr, via + ": prefetch: the loader's connection: " + e);
        po.max_jobs = jobs;
        po.max_bytes = bytes + po.folder->staging_bytes();
        po.timeout = std::chrono::milliseconds(timeout_ms);
        po.locate = [&](const ConversationPrefetch::Hint& h) { return disk.locate(h.ids, {}, h.cvec); };
        po.parked = [&](const ConversationPrefetch::Hint& h, int64_t tokens) {
            if (!park) return false;
            const auto m = park->best(h.ids, {}, h.cvec);
            return (m.live || m.deepest) && m.tokens >= tokens;
        };
        return std::make_unique<ConversationPrefetch>(std::move(po));
    };
    auto next_turn = [](const SavedConversation& s) {
        std::vector<int32_t> p = s.live.ids;
        p.push_back(7);
        return p;
    };
    auto target_of = [&](const SavedConversation& s) {
        const auto m = disk.best(next_turn(s), {}, true);
        return disk.target(m.index, m);
    };
    size_t w = 0;
    const SavedConversation a = conversation(60), b = conversation(61), c = conversation(62);
    check(disk.save(a, w, e) && disk.save(b, w, e) && disk.save(c, w, e), via + ": prefetch: three parked files: " + e);
    {   // the hint's line: ids, cvec, pictures refused
        ConversationPrefetch::Hint h;
        std::string why;
        check(ConversationPrefetch::parse("cvec=0 temperature=0.5 5,6,7", h, why) && !h.cvec && h.ids.size() == 3 &&
                  h.ids[2] == 7, via + ": prefetch: a hint line parses (cvec=0, other keys skipped)");
        check(ConversationPrefetch::parse("1,2", h, why) && h.cvec, via + ": prefetch: cvec defaults to 1");
        check(!ConversationPrefetch::parse("img_keys=1:2 1,2", h, why), via + ": prefetch: a hint with pictures is refused");
        check(!ConversationPrefetch::parse("1,x", h, why) && !ConversationPrefetch::parse("", h, why),
              via + ": prefetch: a bad or empty token list is refused");
    }
    {   // hit: read ahead, taken by the request path's restore, byte for byte; miss: nothing read
        auto p = make(2, size_t{1} << 30, 30000);
        p->hint(hint_text(next_turn(a)));
        check(eventually([&] { return p->stats().loads == 1; }), via + ": prefetch hit: the file is read ahead");
        double waited = 0;
        std::string note;
        auto img = p->take(target_of(a), waited, note);
        check(img && same(a, *img) && waited == 0, via + ": prefetch hit: taken at once, byte for byte");
        check(p->stats().taken == 1 && p->stats().jobs == 0, via + ": prefetch hit: nothing held after the take");
        p->hint(hint_text({1, 2, 3, 4}));
        check(eventually([&] { return p->stats().none == 1; }) && p->stats().loads == 1,
              via + ": prefetch miss: no file starts the prompt, nothing is read");
        check(!p->take(target_of(b), waited, note) && note.empty(), via + ": prefetch miss: the restore reads as before");
        p->hint(hint_text(next_turn(a)));   // the same hint twice: one read
        p->hint(hint_text(next_turn(a)));
        check(eventually([&] { const auto s = p->stats(); return s.loads == 2 && s.duplicate == 1; }),
              via + ": prefetch: a second hint for a file already held is not read again");
        check(p->drop(target_of(a).name) && p->stats().dropped == 1 && p->stats().jobs == 0,
              via + ": prefetch: a read the request path did not use is dropped");
    }
    {   // the serve thread's own request: ahead of hints, settled once read; a take while it reads waits for it
        auto p = make(1, 1, 30000);
        hook.gate.close();
        const int e0 = hook.gate.now();
        const uint64_t t = p->load(target_of(b));
        check(t != 0 && hook.gate.wait_entered(e0 + 1) && !p->settled(t), via + ": prefetch: the admission's own read runs "
                                                                                "past the bounds and is not settled yet");
        std::thread opener([&] { std::this_thread::sleep_for(std::chrono::milliseconds(80)); hook.gate.release(); });
        double waited = 0;
        std::string note;
        auto img = p->take(target_of(b), waited, note);
        opener.join();
        check(img && same(b, *img) && waited > 40 && p->stats().waited == 1,
              via + ": prefetch: a take while the read runs waits for it (" + std::to_string((int) waited) + " ms)");
        check(p->settled(t), via + ": prefetch: a taken ticket is settled");
        const uint64_t t2 = p->load(target_of(c));
        p->wait(t2);
        check(p->settled(t2) && p->take(target_of(c), waited, note) && waited == 0,
              via + ": prefetch: wait() returns once the read is done; the take is immediate");
    }
    {   // racing the background park: an image of the conversation still in the writer is never read from an older file
        auto wf = disk.open_writer_folder(e);
        Gate wgate;
        park = std::make_unique<ConversationParkQueue>(disk.options(), std::make_shared<GateFolder>(wf, wgate, false), 2,
                                                       size_t{1} << 30);
        auto p = make(2, size_t{1} << 30, 30000);
        // the conversation went on past the parked file a: its newer state is being written
        SavedConversation a2 = conversation(60);
        for (int k = 0; k < 300; ++k) a2.live.ids.push_back(500 + k);
        wgate.close();
        const int e1 = wgate.now();
        double waited = 0;
        park->push(SavedConversation(a2), waited);
        check(wgate.wait_entered(e1 + 1), via + ": prefetch race: the newer image is mid-write");
        p->hint(hint_text(next_turn(a2)));
        check(eventually([&] { return p->stats().parked == 1; }) && p->stats().loads == 0,
              via + ": prefetch race: the older file a is not read while the newer image is in the park");
        wgate.release();
        park->drain();
        for (auto& d : park->done()) if (d.ok) disk.adopt(d.file);
        park.reset();
        p->hint(hint_text(next_turn(a2)));
        check(eventually([&] { return p->stats().loads == 1; }), via + ": prefetch race: once adopted, the new file is read");
        std::string note;
        auto img = p->take(target_of(a2), waited, note);
        check(img && same(a2, *img), via + ": prefetch race: the read is the newer conversation, byte for byte");
        // a newer write under the same name (the same tokens parked again): a read of the old one is dropped, not used
        p->hint(hint_text(next_turn(b)));
        check(eventually([&] { return p->stats().loads == 2; }), via + ": prefetch: b is read ahead");
        const auto old_t = target_of(b);
        check(disk.save(b, w, e), via + ": prefetch: b is written again: " + e);
        const auto new_t = target_of(b);
        check(new_t.name == old_t.name && new_t.serial != old_t.serial, via + ": prefetch: same name, a new index serial");
        check(!p->take(new_t, waited, note) && note.find("newer write") != std::string::npos && p->stats().stale == 1,
              via + ": prefetch: the read of the older write is never used: " + note);
    }
    {   // the timeout: a read nobody takes is dropped; the bounds: a hint past them is not read
        auto p = make(2, size_t{1} << 30, 200);
        p->hint(hint_text(next_turn(c)));
        check(eventually([&] { return p->stats().loads == 1; }), via + ": prefetch timeout: read ahead");
        check(eventually([&] { return p->stats().expired == 1 && p->stats().jobs == 0; }, 3000),
              via + ": prefetch timeout: not taken within 200 ms, dropped");
        double waited = 0;
        std::string note;
        check(!p->take(target_of(c), waited, note), via + ": prefetch timeout: the restore reads as before");
        auto q = make(1, size_t{1} << 30, 30000);
        hook.gate.close();
        const int e2 = hook.gate.now();
        q->hint(hint_text(next_turn(b)));
        check(hook.gate.wait_entered(e2 + 1) && eventually([&] { return q->stats().jobs == 1; }),
              via + ": prefetch cap: one read held");
        q->hint(hint_text(next_turn(c)));
        check(eventually([&] { return q->stats().capped == 1; }), via + ": prefetch cap: a second hint over STRATA_SLOT_PREFETCH_MAX=1 is dropped");
        hook.gate.release();
        auto r = make(4, size_t{1} << 20, 30000);
        r->hint(hint_text(next_turn(c)));
        check(eventually([&] { return r->stats().capped == 1; }) && r->stats().loads == 0,
              via + ": prefetch cap: a file over the byte bound (1 MiB) is not read");
    }
    {   // the store fails part way through the read: reported, nothing held, the restore reads as before
        auto p = make(2, size_t{1} << 30, 30000);
        hook.fail_after = 3 << 20;
        p->hint(hint_text(next_turn(c)));
        check(eventually([&] { return p->stats().failed == 1; }), via + ": prefetch: a read that fails part way is reported");
        hook.fail_after = -1;
        double waited = 0;
        std::string note;
        check(!p->take(target_of(c), waited, note) && !note.empty() && p->stats().jobs == 0,
              via + ": prefetch: the failed read is not used (" + note + ")");
        SavedConversation back;
        const auto m = disk.best(next_turn(c), {}, true);
        check(disk.load(m.index, back, e) && same(c, back), via + ": prefetch: the ordinary read still loads it: " + e);
        p->stop();
        p->hint(hint_text(next_turn(c)));
        check(p->stats().hints == 1, via + ": prefetch: a hint after stop() is ignored");
    }
}

void with_env(const char* key, const char* value, const std::string& expect) {
    setenv(key, value, 1);
    auto f = kvib::folder("127.0.0.1:1");
    std::string e;
    check(!f->prepare(e) && e.find(expect) != std::string::npos, std::string(key) + "=" + value + " is refused: " + e);
    unsetenv(key);
}
} // namespace

int main() {
    kvib::ServerOptions so;
    so.arena_bytes = uint64_t{64} << 20;
    so.extent_bytes = uint64_t{1} << 20;
    so.transport = "tcp";
    kvib::Settings cs;
    cs.stage_mib = 1;
    cs.stage_depth = 3;
    cs.tcp_streams = 3;
    cs.transport = "tcp";
    cs.timeout_ms = 3000;
    cs.reconnect_ms = 0;   // dial a lost daemon again on the next save (the default waits 5 s between tries)
    const fs::path dir = fs::temp_directory_path() / ("conversation_kvib_test." + std::to_string(::getpid()));
    {   // --prealloc-gib: extents mapped at the start, none in use; more than the arena is refused
        kvib::ServerOptions po = so;
        po.prealloc_bytes = uint64_t{4} << 20;
        kvib::Server server(po);
        std::string e;
        check(server.start(false, e) && server.stats().mapped == (uint64_t{4} << 20) && server.stats().used == 0,
              "a preallocated pool is mapped and unused: " + e);
        po.prealloc_bytes = uint64_t{65} << 20;
        kvib::Server over(po);
        check(!over.start(false, e) && e.find("arena") != std::string::npos, "a pool larger than the arena is refused");
    }
    {   // in process
        kvib::Server server(so);
        std::string e;
        check(server.start(false, e), "an in-process store starts: " + e);
        auto f = kvib::folder_local(server, cs);
        check(f->prepare(e) && f->transport() == "local", "the in-process folder connects: " + e);
        objects(*f, "local");
        files(*f, dir / "local", "local");
        check(server.stats().handles == 0, "local: no handle is left open");
        check(server.stats().mapped >= server.stats().used && server.stats().mapped <= (uint64_t{64} << 20),
              "local: freed extents stay mapped for reuse, inside the arena");
    }
    // over loopback TCP: a port from this process's id, so parallel test runs do not collide
    const int port = 20000 + (int) (::getpid() % 20000);
    const std::string ep = "127.0.0.1:" + std::to_string(port);
    so.listen = {ep};
    {
        auto server = std::make_unique<kvib::Server>(so);
        std::string e;
        check(server->start(true, e), "a TCP store starts on " + ep + ": " + e);
        {
            auto f = kvib::folder(ep, cs);
            check(f->prepare(e) && f->transport() == "tcp", "the TCP folder connects: " + e);
            objects(*f, "tcp");
            files(*f, dir / "tcp", "tcp");
            // a client that dies mid-write leaves nothing pinned
            auto w = f->open_write("orphan.tmp");
            const auto bytes = pattern(size_t{3} << 20, 9);
            check(w && w->write(bytes.data(), bytes.size()) && w->finish(), "tcp: a finished, uncommitted write");
            w.reset();
        }
        bool released = false;
        for (int i = 0; i < 200 && !released; ++i) {
            released = server->stats().handles == 0 && server->stats().used == 0;
            if (!released) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        check(released, "tcp: a client that went away released its handles and extents");
        // RDMA asked of a daemon without it is refused, not silently TCP
        kvib::Settings rs = cs;
        rs.transport = "rdma";
        auto r = kvib::folder(ep, rs);
        check(!r->prepare(e) && e.find("rdma") != std::string::npos, "transport=rdma without RDMA is refused: " + e);
        // the daemon goes away under a connected engine: a save and a load fail, the engine carries on
        auto f = kvib::folder(ep, cs);
        std::shared_ptr<ConversationFolder> remote(f.get(), [](ConversationFolder*) {});
        ConversationDisk disk;
        check(f->prepare(e) && disk.open(disk_options(remote), e), "an engine connects: " + e);
        const SavedConversation s = conversation(12);
        size_t w = 0;
        check(disk.save(s, w, e), "and parks a conversation: " + e);
        server.reset();
        check(!disk.save(conversation(13), w, e), "a save to a stopped daemon fails: " + e);
        SavedConversation x;
        check(disk.size() == 1 && !disk.load(0, x, e), "a load from a stopped daemon fails: " + e);
        check(disk.size() == 0, "and its entry is dropped");
        ConversationDisk fresh;
        auto g = kvib::folder(ep, cs);
        check(!g->prepare(e), "connecting to a stopped daemon fails: " + e);
        // a restarted daemon holds nothing: the engine finds no file and reads its prompts again
        server = std::make_unique<kvib::Server>(so);
        check(server->start(true, e), "the store restarts: " + e);
        auto h = kvib::folder(ep, cs);
        std::shared_ptr<ConversationFolder> hr(h.get(), [](ConversationFolder*) {});
        check(h->prepare(e) && fresh.open(disk_options(hr), e) && fresh.size() == 0,
              "a restarted store is empty, not an error: " + e);
        // the engine that was connected before reconnects on its next save and parks there again
        check(disk.save(conversation(14), w, e), "the old connection is dialled again and the save lands: " + e);
        check(disk.size() == 1 && disk.load(0, x, e) && same(x, conversation(14)),
              "only the conversation parked after the restart is held, and it loads back equal: " + e);
    }
    // malformed requests and a data path that breaks under a live control connection (review of lev-kvib)
    {
        const int port2 = port + 1;
        kvib::ServerOptions so2 = so;
        so2.listen = {"127.0.0.1:" + std::to_string(port2)};
        kvib::Server server(so2);
        std::string e;
        check(server.start(true, e), "a second TCP store starts: " + e);
        {
            const int fd = dial(port2);
            WireResp r;
            WireReq q;
            q.op = 1;   // hello
            q.a = kvib::kProtocol;
            check(fd >= 0 && raw(fd, q, {}, r) && r.status == 0, "raw: hello");
            check(raw(fd, q, {}, r) && r.status != 0, "raw: a second hello on one connection is refused");
            WireReq c;
            c.op = 4;   // create
            check(raw(fd, c, {}, r) && r.status == 0 && r.a != 0, "raw: create");
            WireReq w;
            w.op = 6;   // write, at an offset whose end wraps past 2^64
            w.a = r.a;
            w.b = UINT64_MAX - 3;
            check(raw(fd, w, std::vector<uint8_t>(8, 0x5a), r) && r.status != 0,
                  "raw: a write whose end wraps is refused (the daemon does not index past its extents)");
            WireReq st;
            st.op = 3;   // stat
            check(raw(fd, st, {}, r) && r.status == 0, "raw: the connection and the daemon carry on");
            ::close(fd);
        }
        Proxy px;
        check(px.start(port2), "a forwarder starts");
        const std::string pep = "127.0.0.1:" + std::to_string(px.port);
        auto f = kvib::folder(pep, cs);
        std::shared_ptr<ConversationFolder> remote(f.get(), [](ConversationFolder*) {});
        ConversationDisk disk;
        check(f->prepare(e) && disk.open(disk_options(remote), e), "an engine connects through the forwarder: " + e);
        check(px.carried() == 1 + (size_t) cs.tcp_streams, "one control and the data connections");
        size_t w = 0;
        check(disk.save(conversation(20), w, e), "a save through the forwarder: " + e);
        px.cut(1, 1 + (size_t) cs.tcp_streams);   // the data connections only; the control one stays up
        check(!disk.save(conversation(21), w, e), "a save over a broken data path fails (and the engine lives): " + e);
        check(disk.save(conversation(22), w, e), "the daemon is dialled again and the next save lands: " + e);
        SavedConversation x;
        bool found = false;
        for (size_t i = 0; i < disk.size(); ++i)
            found = found || (disk.load(i, x, e) && same(x, conversation(22)));
        check(found, "and it loads back equal: " + e);
    }
    // homelab kvib async: the format path's copy threads, the background park, the dial back-off
    {
        kvib::ServerOptions bo = so;
        bo.arena_bytes = uint64_t{1} << 30;
        kvib::Server server(bo);
        std::string e;
        check(server.start(false, e), "a 1 GiB in-process store starts: " + e);
        kvib::Settings bs = cs;
        bs.stage_mib = 4;
        auto f = kvib::folder_local(server, bs);
        check(f->prepare(e), "its folder connects: " + e);
        io_copies(dir / "io");
        format_path(*f, dir / "format");
        const fs::path pq = dir / "parkq";
        auto local = std::shared_ptr<ConversationFolder>(conversation_folder_local(pq));
        park_queue(local, [&] { return std::shared_ptr<ConversationFolder>(conversation_folder_local(pq)); }, "folder");
        std::shared_ptr<ConversationFolder> remote(f.get(), [](ConversationFolder*) {});
        park_queue(remote, [&]() -> std::shared_ptr<ConversationFolder> {
            std::shared_ptr<ConversationFolder> g(kvib::folder_local(server, bs));
            return g;
        }, "store");
        prefetch(local, [&] { return std::shared_ptr<ConversationFolder>(conversation_folder_local(pq)); }, "folder");
        prefetch(remote, [&]() -> std::shared_ptr<ConversationFolder> {
            std::shared_ptr<ConversationFolder> g(kvib::folder_local(server, bs));
            return g;
        }, "store");
    }
    {   // a daemon that stays away is dialled with a doubling wait, not on every call
        const int port3 = port + 2;
        kvib::ServerOptions so3 = so;
        so3.listen = {"127.0.0.1:" + std::to_string(port3)};
        auto server = std::make_unique<kvib::Server>(so3);
        std::string e;
        check(server->start(true, e), "a third TCP store starts: " + e);
        kvib::Settings rs = cs;
        rs.reconnect_ms = 300;
        rs.reconnect_max_ms = 1200;
        auto f = kvib::folder(so3.listen[0], rs);
        std::shared_ptr<ConversationFolder> remote(f.get(), [](ConversationFolder*) {});
        ConversationDisk disk;
        check(f->prepare(e) && disk.open(disk_options(remote), e), "an engine connects: " + e);
        size_t w = 0;
        check(disk.save(conversation(50), w, e), "a save lands: " + e);
        server.reset();
        check(!disk.save(conversation(51), w, e), "the daemon stops: a save fails (the connection is found dead)");
        check(!disk.save(conversation(51), w, e), "the first dial fails; the next waits 300 ms");
        std::this_thread::sleep_for(std::chrono::milliseconds(350));
        check(!disk.save(conversation(51), w, e), "the second dial fails; the next waits 600 ms");
        server = std::make_unique<kvib::Server>(so3);
        check(server->start(true, e), "the store restarts: " + e);
        std::this_thread::sleep_for(std::chrono::milliseconds(350));
        check(!disk.save(conversation(52), w, e), "350 ms later it is not dialled yet (the wait doubled)");
        std::this_thread::sleep_for(std::chrono::milliseconds(350));
        check(disk.save(conversation(52), w, e), "once the wait is over it is dialled and the save lands: " + e);
    }
    with_env("STRATA_KVIB_CONNECT_TIMEOUT_MS", "50", "STRATA_KVIB_CONNECT_TIMEOUT_MS");
    setenv("STRATA_KVIB_RECONNECT_MS", "5000", 1);
    with_env("STRATA_KVIB_RECONNECT_MAX_MS", "1000", "STRATA_KVIB_RECONNECT_MAX_MS");
    unsetenv("STRATA_KVIB_RECONNECT_MS");
    with_env("STRATA_KVIB_TRANSPORT", "carrier-pigeon", "STRATA_KVIB_TRANSPORT");
    with_env("STRATA_KVIB_STAGE_DEPTH", "1", "STRATA_KVIB_STAGE_DEPTH");
    with_env("STRATA_KVIB_TCP_STREAMS", "x", "STRATA_KVIB_TCP_STREAMS");
    {
        auto f = kvib::folder("no-port");
        std::string e;
        check(!f->prepare(e) && e.find("HOST:PORT") != std::string::npos, "a host without a port is refused: " + e);
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::printf("conversation_kvib_test: %d checks passed\n", checks);
    return 0;
}
