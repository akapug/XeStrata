// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/graph_dispatch_probe.cpp - homelab K2: what a node of a recorded graph costs on this device beside its
// own work, and what the host side of a window launch costs, so a batch window's time splits into kernel work,
// per-node dispatch and host round trips.
//
//     build/graph_dispatch_probe [reps 30] [nodes 64,256]
//
// Each kind is recorded twice as a graph of n in-order nodes (n from the list) on the compute queue and replayed;
// the per-node cost is the slope (t(n2) - t(n1)) / (n2 - n1), on the device (timing tags before and after the replay)
// and on the host (submit to wait).  Kinds:
//   empty     a one-work-group kernel that writes one word
//   wide      256 work-groups of 256 that write one word each (a full-device launch with no work)
//   flag      wait_flag_ge on a mapped host flag already set (the window's per-layer doorbell waits)
//   in40k     copy_from_mapped 40 KiB host -> device (one row's residual streams entering a later stage)
//   out40k    copy_from_mapped 40 KiB device -> host (leaving an earlier stage)
//   stamp     gpu_stamp (the profiler's stamp kernel)
// then the hand-off at 16 rows: 48 per-row launches against copy_rows_mapped_multi's one (and its bytes compared),
// the host cost of the graph submit call for a 1-node and a 1024-node graph, and a 1-node graph's round trip
// (submit to the host seeing it done) through wait() and through the engine's ext_oneapi_empty() poll.
#include "strata/core/runtime.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {
namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;

sycl::queue& Q() { return strata::core::Runtime::get().compute(); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

struct Timed {
    double dev_us = 0, host_us = 0;
};

sx::command_graph<sx::graph_state::executable> record(const std::function<void(sycl::queue&)>& body, int n) {
    sycl::queue& q = Q();
    sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
    g.begin_recording(q);
    for (int i = 0; i < n; ++i) body(q);
    g.end_recording(q);
    return g.finalize();
}

// median device span (tags around the replay) and host span (submit to wait) of the graph
Timed replay(sx::command_graph<sx::graph_state::executable>& ex, int reps) {
    sycl::queue& q = Q();
    std::vector<double> dev, host;
    for (int r = 0; r < reps + 5; ++r) {
        const auto h0 = Clock::now();
        const sycl::event a = sx::submit_profiling_tag(q);
        q.ext_oneapi_graph(ex);
        const sycl::event b = sx::submit_profiling_tag(q);
        q.wait();
        const auto h1 = Clock::now();
        if (r < 5) continue;
        const auto t0 = a.get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto t1 = b.get_profiling_info<sycl::info::event_profiling::command_end>();
        dev.push_back((double) (t1 - t0) / 1e3);
        host.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count());
    }
    return {median(dev), median(host)};
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::max(5, std::atoi(argv[1])) : 30;
    std::vector<int> ns{64, 256};
    if (argc > 2) {
        ns.clear();
        for (const char* c = argv[2]; *c;) {
            char* e = nullptr;
            const long v = std::strtol(c, &e, 10);
            if (e == c) break;
            ns.push_back((int) v);
            c = *e == ',' ? e + 1 : e;
        }
    }
    if (ns.size() < 2) { std::fprintf(stderr, "need two node counts\n"); return 2; }
    auto& rt = strata::core::Runtime::get();
    sycl::queue& q = Q();
    std::printf("device: %s; reps %d; node counts %d, %d\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                reps, ns[0], ns[1]);
    for (const char* e : {"SYCL_UR_USE_LEVEL_ZERO_V2", "UR_L0_USE_DRIVER_INORDER_LISTS", "UR_L0_USE_IMMEDIATE_COMMANDLISTS",
                          "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "UR_L0_CMD_BUFFER_USE_IMMEDIATE_APPEND_PATH",
                          "UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS"})
        if (const char* v = std::getenv(e)) std::printf("env %s=%s\n", e, v);

    constexpr int64_t kRow = 10240;   // floats: one row's hc residual streams (4 x 2560)
    constexpr int kRows = 16;
    uint32_t* d_word = sycl::malloc_device<uint32_t>(65536 + 64, q);
    uint32_t* h_flag = sycl::malloc_host<uint32_t>(16, q);
    float* h_rows = sycl::malloc_host<float>((size_t) kRows * 12804 + 64, q);
    float* d_rows = sycl::malloc_device<float>((size_t) kRows * kRow + 64, q);
    float* d_bo = sycl::malloc_device<float>((size_t) kRows * 2560 + 64, q);
    float* d_inj = sycl::malloc_device<float>((size_t) kRows * 4 + 64, q);
    float* d_rows2 = sycl::malloc_device<float>((size_t) kRows * kRow + 64, q);
    float* d_bo2 = sycl::malloc_device<float>((size_t) kRows * 2560 + 64, q);
    float* d_inj2 = sycl::malloc_device<float>((size_t) kRows * 4 + 64, q);
    unsigned long long* d_stamps = sycl::malloc_device<unsigned long long>(4096, q);
    if (!d_word || !h_flag || !h_rows || !d_rows || !d_bo || !d_inj || !d_rows2 || !d_bo2 || !d_inj2 || !d_stamps) {
        std::fprintf(stderr, "allocation failed\n");
        return 3;
    }
    *(volatile uint32_t*) h_flag = 7;
    for (size_t i = 0; i < (size_t) kRows * 12804; ++i) h_rows[i] = (float) (i % 977) * 0.25f - 3.0f;
    q.memset(d_rows, 0, sizeof(float) * kRows * kRow).wait();
    const bool stamp_ok = k::gpu_stamp_available();

    struct Kind {
        const char* name;
        std::function<void(sycl::queue&)> body;
    };
    int slot = 0;
    std::vector<Kind> kinds = {
        {"empty", [&](sycl::queue& qq) {
             uint32_t* w = d_word;
             qq.parallel_for(sycl::nd_range<1>(64, 64), [=](sycl::nd_item<1> it) { if (it.get_global_id(0) == 0) w[0] += 1u; });
         }},
        {"wide", [&](sycl::queue& qq) {
             uint32_t* w = d_word;
             qq.parallel_for(sycl::nd_range<1>(256 * 256, 256), [=](sycl::nd_item<1> it) {
                 if (it.get_local_id(0) == 0) w[it.get_group(0)] = (uint32_t) it.get_group(0);
             });
         }},
        {"flag", [&](sycl::queue& qq) { k::wait_flag_ge(h_flag, 1u, &qq); }},
        {"in40k", [&](sycl::queue& qq) { k::copy_from_mapped(d_rows, h_rows, kRow, &qq); }},
        {"out40k", [&](sycl::queue& qq) { k::copy_from_mapped(h_rows, d_rows, kRow, &qq); }},
    };
    if (stamp_ok) kinds.push_back({"stamp", [&](sycl::queue& qq) { k::gpu_stamp(d_stamps, (slot++) & 4095, &qq); }});

    std::printf("\nper-node cost in a recorded in-order graph (slope over %d -> %d nodes), us\n", ns[0], ns[1]);
    std::printf("%-8s %10s %10s   %12s %12s\n", "kind", "device", "host", "dev@n1", "dev@n2");
    for (Kind& kd : kinds) {
        auto g1 = record(kd.body, ns[0]);
        auto g2 = record(kd.body, ns[1]);
        const Timed a = replay(g1, reps), b = replay(g2, reps);
        const double dn = (double) (ns[1] - ns[0]);
        std::printf("%-8s %10.2f %10.2f   %12.1f %12.1f\n", kd.name, (b.dev_us - a.dev_us) / dn,
                    (b.host_us - a.host_us) / dn, a.dev_us, b.dev_us);
        std::fflush(stdout);
    }

    // eager submission of the same empty kernel (no graph): device and host per kernel
    {
        std::vector<double> dev, host;
        for (int r = 0; r < reps + 3; ++r) {
            const auto h0 = Clock::now();
            const sycl::event a = sx::submit_profiling_tag(q);
            for (int i = 0; i < ns[1]; ++i) kinds[0].body(q);
            const sycl::event b = sx::submit_profiling_tag(q);
            const auto hs = Clock::now();
            q.wait();
            if (r < 3) continue;
            dev.push_back((double) (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                    a.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e3 / ns[1]);
            host.push_back(std::chrono::duration<double, std::micro>(hs - h0).count() / ns[1]);
        }
        std::printf("eager empty kernel: device %.2f us a kernel, host submit %.2f us a kernel\n", median(dev), median(host));
    }

    // the hand-off at 16 rows: 3 launches a row against one, in graphs, and the bytes
    {
        const int64_t HB = 12804;
        auto per_row_in = [&](sycl::queue& qq) {
            for (int t = 0; t < kRows; ++t) {
                const float* hin = h_rows + (size_t) t * HB;
                k::copy_from_mapped(d_rows + (size_t) t * kRow, hin, kRow, &qq);
                k::copy_from_mapped(d_bo + (size_t) t * 2560, hin + kRow, 2560, &qq);
                k::copy_from_mapped(d_inj + (size_t) t * 4, hin + kRow + 2560, 4, &qq);
            }
        };
        auto one_in = [&](sycl::queue& qq) {
            const k::RowsCopy pc[3] = {{d_rows2, h_rows, kRow, kRow, HB}, {d_bo2, h_rows + kRow, 2560, 2560, HB},
                                       {d_inj2, h_rows + kRow + 2560, 4, 4, HB}};
            k::copy_rows_mapped_multi(pc, 3, kRows, &qq);
        };
        auto per_row_out = [&](sycl::queue& qq) {
            for (int t = 0; t < kRows; ++t) {
                float* hout = h_rows + (size_t) t * HB;
                k::copy_from_mapped(hout, d_rows + (size_t) t * kRow, kRow, &qq);
                k::copy_from_mapped(hout + kRow, d_bo + (size_t) t * 2560, 2560, &qq);
                k::copy_from_mapped(hout + kRow + 2560, d_inj + (size_t) t * 4, 4, &qq);
            }
        };
        auto one_out = [&](sycl::queue& qq) {
            const k::RowsCopy pc[3] = {{h_rows, d_rows, kRow, HB, kRow}, {h_rows + kRow, d_bo, 2560, HB, 2560},
                                       {h_rows + kRow + 2560, d_inj, 4, HB, 4}};
            k::copy_rows_mapped_multi(pc, 3, kRows, &qq);
        };
        auto ga = record(per_row_in, 1), gb = record(one_in, 1), gc = record(per_row_out, 1), gd = record(one_out, 1);
        const Timed a = replay(ga, reps), b = replay(gb, reps), c = replay(gc, reps), d = replay(gd, reps);
        std::vector<float> x((size_t) kRows * kRow), y(x.size()), bx((size_t) kRows * 2560), by(bx.size());
        q.memcpy(x.data(), d_rows, x.size() * 4);
        q.memcpy(y.data(), d_rows2, y.size() * 4);
        q.memcpy(bx.data(), d_bo, bx.size() * 4);
        q.memcpy(by.data(), d_bo2, by.size() * 4).wait();
        const bool same = std::memcmp(x.data(), y.data(), x.size() * 4) == 0 && std::memcmp(bx.data(), by.data(), bx.size() * 4) == 0;
        std::printf("hand-off in, 16 rows: 48 launches %.1f us device (%.1f host), one launch %.1f us (%.1f host); bytes %s\n",
                    a.dev_us, a.host_us, b.dev_us, b.host_us, same ? "equal" : "DIFFER");
        std::printf("hand-off out, 16 rows: 48 launches %.1f us device (%.1f host), one launch %.1f us (%.1f host)\n",
                    c.dev_us, c.host_us, d.dev_us, d.host_us);
        std::printf("KEY handoff_in_us %.1f->%.1f out_us %.1f->%.1f bytes %s\n", a.dev_us, b.dev_us, c.dev_us, d.dev_us,
                    same ? "equal" : "DIFFER");
        if (!same) return 4;
    }

    // host cost of the submit call, and a 1-node graph's round trip
    {
        auto g1 = record(kinds[0].body, 1), gk = record(kinds[0].body, 1024);
        auto submit_cost = [&](sx::command_graph<sx::graph_state::executable>& ex) {
            std::vector<double> v;
            for (int r = 0; r < reps + 3; ++r) {
                const auto h0 = Clock::now();
                q.ext_oneapi_graph(ex);
                const auto h1 = Clock::now();
                q.wait();
                if (r >= 3) v.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count());
            }
            return median(v);
        };
        std::printf("graph submit call: 1 node %.1f us, 1024 nodes %.1f us (host)\n", submit_cost(g1), submit_cost(gk));
        std::vector<double> rw, rp;
        for (int r = 0; r < reps + 3; ++r) {
            auto h0 = Clock::now();
            q.ext_oneapi_graph(g1);
            q.wait();
            auto h1 = Clock::now();
            if (r >= 3) rw.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count());
            h0 = Clock::now();
            q.ext_oneapi_graph(g1);
            while (!q.ext_oneapi_empty()) {}
            h1 = Clock::now();
            if (r >= 3) rp.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count());
        }
        q.wait();
        std::printf("1-node graph round trip: wait() %.1f us, ext_oneapi_empty() poll %.1f us\n", median(rw), median(rp));
    }
    for (void* p : {(void*) d_word, (void*) h_flag, (void*) h_rows, (void*) d_rows, (void*) d_bo, (void*) d_inj,
                    (void*) d_rows2, (void*) d_bo2, (void*) d_inj2, (void*) d_stamps})
        sycl::free(p, q);
    (void) rt;
    std::printf("KEY probe rc=0\n");
    return 0;
}
