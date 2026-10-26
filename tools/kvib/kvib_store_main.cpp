// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// strata-kvib-store: the RAM store behind --conversation-save kvib://HOST:PORT (homelab, src/core/kvib.cpp).
//   strata-kvib-store --listen IP:PORT [--listen IP:PORT ...] [--arena-gib N] [--extent-mib N]
//                     [--prealloc-gib N] [--rdma-port-offset N] [--transport auto|rdma|tcp] [--stats-s N] [--verbose]
// Registering memory for RDMA needs a memlock limit as large as the arena (root, or `ulimit -l unlimited`).
#include "strata/core/kvib.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

bool number(const char* s, long long lo, long long hi, long long& v) {
    char* end = nullptr;
    const long long x = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || x < lo || x > hi) return false;
    v = x;
    return true;
}
} // namespace

int main(int argc, char** argv) {
    namespace kv = strata::core::kvib;
    kv::ServerOptions o;
    bool listen_given = false;
    long long stats_s = 60;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* next = i + 1 < argc ? argv[i + 1] : nullptr;
        long long v = 0;
        if (a == "--listen" && next) {
            if (!listen_given) o.listen.clear();
            listen_given = true;
            o.listen.push_back(argv[++i]);
        } else if (a == "--arena-gib" && next && number(next, 1, 1 << 20, v)) {
            o.arena_bytes = (uint64_t) v << 30;
            ++i;
        } else if (a == "--extent-mib" && next && number(next, 1, 4096, v)) {
            o.extent_bytes = (uint64_t) v << 20;
            ++i;
        } else if (a == "--prealloc-gib" && next && number(next, 0, 1 << 20, v)) {
            o.prealloc_bytes = (uint64_t) v << 30;
            ++i;
        } else if (a == "--rdma-port-offset" && next && number(next, 1, 1000, v)) {
            o.rdma_port_offset = (int) v;
            ++i;
        } else if (a == "--transport" && next &&
                   (std::strcmp(next, "auto") == 0 || std::strcmp(next, "rdma") == 0 || std::strcmp(next, "tcp") == 0)) {
            o.transport = argv[++i];
        } else if (a == "--stats-s" && next && number(next, 0, 86400, v)) {
            stats_s = v;
            ++i;
        } else if (a == "--verbose") {
            o.verbose = true;
        } else {
            std::fprintf(stderr, "strata-kvib-store: bad argument '%s' (see the header of kvib_store_main.cpp)\n",
                         a.c_str());
            return 2;
        }
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    kv::Server server(o);
    std::string err;
    if (!server.start(true, err)) {
        std::fprintf(stderr, "strata-kvib-store: %s\n", err.c_str());
        return 1;
    }
    const kv::ServerStats st0 = server.stats();
    for (const auto& ep : o.listen)
        std::fprintf(stderr, "strata-kvib-store: listening on %s (rdma: %s, port +%d), arena %.1f GiB in %llu MiB "
                             "extents\n", ep.c_str(), st0.rdma_devices ? "yes" : "no", o.rdma_port_offset,
                     (double) o.arena_bytes / (1ull << 30), (unsigned long long) (o.extent_bytes >> 20));
    auto last = std::chrono::steady_clock::now();
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (stats_s > 0 && std::chrono::steady_clock::now() - last > std::chrono::seconds(stats_s)) {
            last = std::chrono::steady_clock::now();
            const kv::ServerStats st = server.stats();
            std::fprintf(stderr, "strata-kvib-store: %llu objects, %llu open handles, %.2f GiB used, %.2f mapped of %.1f\n",
                         (unsigned long long) st.objects, (unsigned long long) st.handles,
                         (double) st.used / (1ull << 30), (double) st.mapped / (1ull << 30), (double) st.cap / (1ull << 30));
        }
    }
    server.stop();
    return 0;
}
