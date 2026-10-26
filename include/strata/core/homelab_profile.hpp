// SPDX-License-Identifier: MIT
// Opt-in host intervals, not GPU busy time. No token IDs, prompts, or environment dumps.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <unistd.h>

namespace homelab_profile {
inline unsigned long long setting(const char* key, unsigned long long fallback) {
    const char* s = std::getenv(key);
    if (!s || !*s) return fallback;
    char* end = nullptr;
    const auto n = std::strtoull(s, &end, 10);
    return *s != '-' && end && !*end ? n : fallback;
}
inline bool enabled() {
    static const bool on = setting("STRATA_HOST_PROFILE", 0) == 1;
    return on;
}
inline std::atomic<unsigned long long> count{0};
inline std::atomic<unsigned long long> windows{0};
inline thread_local unsigned long long window = 0;
inline thread_local int row_count = 0;
inline thread_local int active_count = 0;
inline long long steady_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Window {
    unsigned long long previous = window;
    int previous_rows = row_count, previous_active = active_count;
    Window(int rows, int active) {
        if (!enabled() || window) return;
        window = ++windows;
        row_count = rows;
        active_count = active;
    }
    ~Window() { window = previous; row_count = previous_rows; active_count = previous_active; }
};
struct Span {
    const char* name;
    long long start = 0, epoch = 0;
    int layer;
    bool live = false;
    long long resident_lookups = -1, cpu_distinct_pairs = -1, pcie_distinct_experts = -1;
    explicit Span(const char* label, int l = -1) : name(label), layer(l) {
        if (!enabled()) return;
        static const char* trigger = std::getenv("STRATA_HOST_PROFILE_TRIGGER");
        if (trigger && *trigger && ::access(trigger, F_OK) != 0) return;
        static const auto limit = setting("STRATA_HOST_PROFILE_MAX_EVENTS", 20000);
        static const auto skip = setting("STRATA_HOST_PROFILE_SKIP_WINDOWS", 0);
        if (window <= skip) return;
        const auto index = count.fetch_add(1);
        if (index >= limit) {
            if (index == limit) std::fprintf(stderr, "HOST_PROFILE_TRUNCATED limit=%llu\n", limit);
            return;
        }
        live = true;
        start = steady_us();
        epoch = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;
    void end() {
        if (!live) return;
        live = false;
        const auto duration = steady_us() - start;
        // One locked stderr write per completed span, bounded per process. Its own cost is not in duration.
        std::fprintf(stderr, "HOST_PROFILE {\"name\":\"%s\",\"steady_us\":%lld,\"epoch_us\":%lld,"
                     "\"dur_us\":%lld,\"pid\":%d,\"tid\":%llu,\"window\":%llu,"
                     "\"rows\":%d,\"active\":%d,\"layer\":%d,\"resident_lookups\":%lld,"
                     "\"cpu_distinct_pairs\":%lld,\"pcie_distinct_experts\":%lld}\n",
                     name, start, epoch, duration, (int) getpid(),
                     (unsigned long long) std::hash<std::thread::id>{}(std::this_thread::get_id()),
                     window, row_count, active_count, layer, resident_lookups, cpu_distinct_pairs, pcie_distinct_experts);
    }
    ~Span() { end(); }
};
} // namespace homelab_profile
