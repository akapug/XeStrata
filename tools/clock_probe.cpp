// clock_probe: which sycl_ext_oneapi_clock scopes a device has, and whether each one is a timeline the stage
// stamps can use - one counter that every kernel reads, so that a stamp in one kernel minus a stamp in an earlier one
// is the device time between them.  For each scope: stamps around spins whose device time the queue's profiling
// events measure (the tick rate, and whether it holds at two lengths and across a host sleep), a run of back-to-back
// stamps (monotonic?), one wide kernel's per-work-group stamps (one counter across the EUs, or one per EU?), and the
// same stamps replayed in a recorded graph, as the verify window is.
//   build: clang++ -fsycl -O2 tools/clock_probe.cpp -o clock_probe
//   run:   clock_probe [force]   (force: try the scopes the device does not report too)
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

namespace sx = sycl::ext::oneapi::experimental;
using u64 = unsigned long long;

static double ev_ns(const sycl::event& e) {
    return (double) (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                     e.get_profiling_info<sycl::info::event_profiling::command_start>());
}

static double host_ns() {
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A spin the compiler cannot fold: n relaxed atomic adds to one device word.
static sycl::event spin(sycl::queue& q, float* out, long n) {
    return q.single_task([=] {
        sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*(unsigned*) out);
        for (long k = 0; k < n; ++k) a.fetch_add(1u);
    });
}

// The GPU's actual frequency (MHz) as the xe driver reports it (STRATA probe only: card0 is the first B70 on odie).
static int act_mhz() {
    std::ifstream f(std::getenv("PROBE_FREQ") ? std::getenv("PROBE_FREQ") : "/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq");
    int v = -1;
    f >> v;
    return v;
}

template <sx::clock_scope S> static sycl::event stamp(sycl::queue& q, u64* buf, int i) {
    return q.single_task([=] { buf[i] = (u64) sx::clock<S>(); });
}

template <sx::clock_scope S> static void probe(sycl::queue& q, const char* name) {
    std::printf("== scope %s\n", name);
    u64* st = sycl::malloc_shared<u64>(8192, q);
    float* sink = sycl::malloc_device<float>(1, q);
    std::memset(st, 0, 8192 * 8);
    // the spin lengths: n0 iterations timed by their event, scaled to 5 and 20 ms
    spin(q, sink, 1 << 16).wait();
    const long n0 = 1 << 16;
    sycl::event c = spin(q, sink, n0);
    c.wait();
    const double per = ev_ns(c) / (double) n0;
    const long n5 = (long) (5e6 / per), n20 = (long) (20e6 / per);
    std::printf("spin: %.3f ns/iter (n5=%ld n20=%ld)\n", per, n5, n20);

    // 1. stamps around the spins and a host sleep, on one in-order queue
    stamp<S>(q, st, 0);
    sycl::event a = spin(q, sink, n5);
    stamp<S>(q, st, 1);
    sycl::event b = spin(q, sink, n20);
    stamp<S>(q, st, 2);
    stamp<S>(q, st, 3);
    q.wait();
    const double h0 = host_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stamp<S>(q, st, 4);
    q.wait();
    const double hs = host_ns() - h0;
    const double ea = ev_ns(a), eb = ev_ns(b);
    for (int i = 0; i < 5; ++i) std::printf("st[%d]=%llu (hi32=%llu)\n", i, st[i], st[i] >> 32);
    const double d01 = (double) (st[1] - st[0]), d12 = (double) (st[2] - st[1]), d23 = (double) (st[3] - st[2]);
    const double d34 = (double) (st[4] - st[3]);
    std::printf("spin5: event %.3f ms, stamps %.0f ticks -> %.4f ticks/ns\n", ea / 1e6, d01, d01 / ea);
    std::printf("spin20: event %.3f ms, stamps %.0f ticks -> %.4f ticks/ns\n", eb / 1e6, d12, d12 / eb);
    const double rate = d12 / eb;
    std::printf("back-to-back stamps: %.0f ticks = %.2f us; across a 50 ms host sleep (host %.2f ms): %.0f ticks = "
                "%.2f ms\n", d23, d23 / rate / 1e3, hs / 1e6, d34, d34 / rate / 1e6);

    // 2. 512 back-to-back stamp kernels: monotonic?
    for (int i = 0; i < 512; ++i) stamp<S>(q, st, 16 + i);
    q.wait();
    int back = 0, same = 0;
    for (int i = 17; i < 16 + 512; ++i) { back += st[i] < st[i - 1]; same += st[i] == st[i - 1]; }
    std::printf("512 stamps: %d go backwards, %d repeat, span %.0f ticks (%.2f us/stamp)\n", back, same,
                (double) (st[16 + 511] - st[16]), (double) (st[16 + 511] - st[16]) / rate / 1e3 / 511);

    // 3. one wide kernel: each work-group's first item stamps; one counter shows a spread under the kernel's length
    const int groups = 4096;
    sycl::event w = q.parallel_for(sycl::nd_range<1>(groups * 64, 64), [=](sycl::nd_item<1> it) {
        if (it.get_local_id(0) == 0) st[4096 + it.get_group(0)] = (u64) sx::clock<S>();
    });
    w.wait();
    const auto [mn, mx] = std::minmax_element(st + 4096, st + 4096 + groups);
    std::printf("wide kernel %d groups: event %.2f us, stamp spread %.0f ticks = %.2f us, min %llu max %llu\n",
                groups, ev_ns(w) / 1e3, (double) (*mx - *mn), (double) (*mx - *mn) / rate / 1e3, *mn, *mx);

    // 5. idle gaps: a stamp, a host sleep of g ms with the GPU idle, a stamp - does the counter survive the idle?
    for (const int g : {0, 1, 2, 5, 10, 20, 50, 100}) {
        stamp<S>(q, st, 600);
        q.wait();
        const double hg = host_ns();
        if (g > 0) std::this_thread::sleep_for(std::chrono::milliseconds(g));
        stamp<S>(q, st, 601);
        q.wait();
        const double hd = host_ns() - hg;
        std::printf("idle %3d ms (host %.2f ms): %llu -> %llu, d %.3f ms%s\n", g, hd / 1e6, st[600], st[601],
                    ((double) st[601] - (double) st[600]) / rate / 1e6, st[601] < st[600] ? "  BACKWARDS" : "");
    }
    // 6. a long busy stretch, stamped every ~0.5 s: does the counter carry past 32 bits, and does the rate hold?
    {
        const long nh = (long) (500e6 / per);
        sycl::event ev[20];
        stamp<S>(q, st, 700);
        for (int i = 1; i <= 18; ++i) { ev[i] = spin(q, sink, nh); stamp<S>(q, st, 700 + i); }
        q.wait();
        for (int i = 0; i <= 18; ++i)
            std::printf("long %2d: %llu (hi32=%llu)%s rate %.4f ticks/ns\n", i, st[700 + i], st[700 + i] >> 32,
                        i > 0 && st[700 + i] < st[699 + i] ? "  BACKWARDS" : "",
                        i > 0 ? ((double) st[700 + i] - (double) st[699 + i]) / ev_ns(ev[i]) : 0.0);
    }
    // 7. the rate inside one kernel (a stamp at its start and end, no launch gaps), at several lengths, against its
    // event, with the GPU's frequency read while it runs; then with every EU busy
    for (const double ms : {0.1, 1.0, 10.0, 100.0}) {
        const long n = (long) (ms * 1e6 / per);
        sycl::event e = q.single_task([=] {
            const u64 t0 = (u64) sx::clock<S>();
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*(unsigned*) sink);
            for (long k = 0; k < n; ++k) a.fetch_add(1u);
            st[800] = t0;
            st[801] = (u64) sx::clock<S>();
        });
        std::this_thread::sleep_for(std::chrono::microseconds((long) (ms * 500)));
        const int f = act_mhz();
        e.wait();
        std::printf("in-kernel %6.1f ms: event %.4f ms, %.0f ticks -> %.4f ticks/ns (act %d MHz)\n", ms, ev_ns(e) / 1e6,
                    (double) (st[801] - st[800]), (double) (st[801] - st[800]) / ev_ns(e), f);
    }
    {
        const int groups = 2048;
        const long n = (long) (100e6 / per / 50);
        unsigned* cnt = sycl::malloc_device<unsigned>(groups, q);
        sycl::event e = q.parallel_for(sycl::nd_range<1>(groups * 64, 64), [=](sycl::nd_item<1> it) {
            const int g = (int) it.get_group(0);
            const u64 t0 = (u64) sx::clock<S>();
            float x = (float) it.get_global_id(0);
            for (long k = 0; k < n; ++k) x = sycl::sin(x) + 1.0f;
            if (x == 12345.0f) cnt[g] = 1;
            if (it.get_local_id(0) == 0) { st[1024 + 2 * g] = t0; st[1025 + 2 * g] = (u64) sx::clock<S>(); }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const int f = act_mhz();
        e.wait();
        u64 a = ~0ull, b = 0;
        for (int g = 0; g < groups; ++g) { a = std::min(a, st[1024 + 2 * g]); b = std::max(b, st[1025 + 2 * g]); }
        std::printf("wide busy kernel: event %.3f ms, first start to last end %.0f ticks -> %.4f ticks/ns (act %d MHz)\n",
                    ev_ns(e) / 1e6, (double) (b - a), (double) (b - a) / ev_ns(e), f);
        sycl::free(cnt, q);
    }

    // 4. the stamps of 1. recorded in a graph and replayed three times
    try {
        sx::command_graph<sx::graph_state::modifiable> g(q.get_context(), q.get_device());
        g.begin_recording(q);
        stamp<S>(q, st, 32 + 512);
        spin(q, sink, n5);
        stamp<S>(q, st, 33 + 512);
        spin(q, sink, n20);
        stamp<S>(q, st, 34 + 512);
        g.end_recording(q);
        auto ex = g.finalize();
        u64 last = 0;
        for (int r = 0; r < 3; ++r) {
            const double hr = host_ns();
            q.ext_oneapi_graph(ex).wait();
            const double hw = host_ns() - hr;
            const u64* s = st + 544;
            std::printf("graph replay %d: d01 %.3f ms d12 %.3f ms (host %.3f ms), after previous replay: %s\n", r,
                        (double) (s[1] - s[0]) / rate / 1e6, (double) (s[2] - s[1]) / rate / 1e6, hw / 1e6,
                        r == 0 ? "-" : (s[0] > last ? "yes" : "NO"));
            last = s[2];
        }
    } catch (const std::exception& e) {
        std::printf("graph: %s\n", e.what());
    }
    sycl::free(st, q);
    sycl::free(sink, q);
}

template <sx::clock_scope S> static void run(sycl::queue& q, const char* name, sycl::aspect asp, bool force) {
    const bool has = q.get_device().has(asp);
    std::printf("aspect %s: %s\n", name, has ? "yes" : "no");
    if (!has && !force) return;
    try { probe<S>(q, name); } catch (const std::exception& e) { std::printf("scope %s: %s\n", name, e.what()); }
}

int main(int argc, char** argv) {
    const bool force = argc > 1 && std::strcmp(argv[1], "force") == 0;
    sycl::queue q(sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const auto d = q.get_device();
    std::printf("device: %s | driver %s | timer resolution %zu ns\n", d.get_info<sycl::info::device::name>().c_str(),
                d.get_info<sycl::info::device::driver_version>().c_str(),
                (size_t) d.get_info<sycl::info::device::profiling_timer_resolution>());
    run<sx::clock_scope::sub_group>(q, "sub_group", sycl::aspect::ext_oneapi_clock_sub_group, force);
    run<sx::clock_scope::work_group>(q, "work_group", sycl::aspect::ext_oneapi_clock_work_group, force);
    run<sx::clock_scope::device>(q, "device", sycl::aspect::ext_oneapi_clock_device, force);
    return 0;
}
