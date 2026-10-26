// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/hc_mode_probe.cpp - homelab: why batch_window_ubench's "hc read (Q8_0, x2 a layer)" row is bimodal at 16
// rows (144 or 189 us a layer on the B70, the same form, from process to process).
//
// The read (STRATA_HC_Q8_WIN 2, synthetic Q8_0 projections in the model's geometry) is timed on several sets of its
// buffers in one process, each set placed differently, under three cache conditions:
//   * fill  - the ubench's method: a constant fill of a 128 MiB buffer, then tag | read | tag
//   * copy  - a copy kernel over 2 x 128 MiB of varying data (every line read and written), then tag | read | tag
//   * hot   - no flush (the previous call's lines stay in L2)
// and, after a fill, its down and up launches alone.  Per set it prints each buffer's address modulo 2 MiB and the
// Level Zero page size of its allocation.
//
// The hypotheses it separates (run it in 2-3 processes):
//   H1 placement: sets in one process differ, and the slow ones share an address / page-size property
//   H2 process:   every set in a process shares one mode, and processes differ (JIT, scratch, context state)
//   H3 clock:     set 0 timed again at the end differs from set 0 at the start
//   H4 flush:     "hot" matches the fast mode, "copy" the slow mode, and only "fill" flips between them (the constant
//                 fill does not always evict the weights: a measurement artifact, not a property of the read)
//
// Then STRATA_HC_PREFETCH's question (set 0, 16 rows, every call after a copy flush; only the read between tags):
// does a prefetch launch (l2_prefetch) of the read's projections make it hot, and do the prefetched lines survive a
// 10.7 MB streaming read between (gdn ssm_out's weight size; once, or 8 times for ssm_out's ~200 us at 16 rows)?
// The launch's own time and its effect on the streaming read's, and the read's bits with and without it.
//
//     build/hc_mode_probe [reps 25] [warm 3] [flush_mib 128]
#include "strata/core/runtime.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/l2_prefetch.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#if __has_include(<level_zero/ze_api.h>) && __has_include(<sycl/ext/oneapi/backend/level_zero.hpp>)
#include <dlfcn.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#define PROBE_ZE 1
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, MT = 16;
constexpr size_t kAlign2M = (size_t) 2 << 20;

sycl::queue& Q() { return strata::core::Runtime::get().compute(); }

void* raw(size_t bytes) {
    auto& rt = strata::core::Runtime::get();
    void* p = sycl::malloc_device(bytes + 64, rt.device(), rt.context());   // as batch_window_ubench's Arena
    if (p == nullptr) throw std::runtime_error("device allocation of " + std::to_string(bytes) + " B failed");
    return p;
}
void* aligned(size_t bytes, size_t al) {
    auto& rt = strata::core::Runtime::get();
    void* p = sycl::aligned_alloc_device(al, bytes + 64, rt.device(), rt.context());
    if (p == nullptr) throw std::runtime_error("aligned device allocation failed");
    return p;
}

// the allocation's page size as Level Zero reports it (0: unknown)
size_t page_size(const void* p) {
#ifdef PROBE_ZE
    using Fn = decltype(&zeMemGetAllocProperties);
    static const Fn fn = [] {
        void* lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (lib == nullptr) lib = dlopen("libze_loader.so.1", RTLD_NOW);
        return lib ? reinterpret_cast<Fn>(dlsym(lib, "zeMemGetAllocProperties")) : nullptr;
    }();
    auto& rt = strata::core::Runtime::get();
    if (fn == nullptr || rt.device().get_backend() != sycl::backend::ext_oneapi_level_zero) return 0;
    ze_memory_allocation_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t dev = nullptr;
    const auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(rt.context());
    return fn(ctx, p, &props, &dev) == ZE_RESULT_SUCCESS ? (size_t) props.pageSize : 0;
#else
    (void) p;
    return 0;
#endif
}

inline uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
void fill_bytes(uint8_t* b, size_t n, uint32_t seed) {
    Q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = (uint8_t) (mix((uint32_t) i[0] * 2654435761u ^ seed) % 0x3c); });
}
void fill_f32(float* p, size_t n, uint32_t seed, float lo, float hi) {
    Q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        p[i] = lo + (hi - lo) * (float) (mix((uint32_t) i[0] * 2654435761u ^ seed) & 0xffffff) / 16777216.0f;
    });
}
void fill_bf16(uint16_t* p, size_t n, uint32_t seed) {
    Q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float v = 0.05f * ((float) (mix((uint32_t) i[0] * 2654435761u ^ seed) & 0xffff) / 32768.0f - 1.0f);
        p[i] = (uint16_t) (sycl::bit_cast<uint32_t>(v) >> 16);
    });
}

struct Set {
    std::string name;
    uint8_t *q8d, *q8u;
    float* wn;
    uint16_t* wi;
};

struct Probe {
    int reps, warm;
    float *flush_a, *flush_b, *flush_c;   // fill target (ubench's, raw); the copy's source and destination
    size_t flush_n;
    float *R, *bo, *inj, *inj2, *lo, *rs, *xn, *mixed;

    std::vector<k::FusedGrArgs> args(const Set& s) const {
        std::vector<k::FusedGrArgs> fa(MT);
        for (int t = 0; t < MT; ++t) {
            k::FusedGrArgs& x = fa[t];
            x.R = R + (size_t) t * D; x.R_out = R + (size_t) t * D; x.apply = true;
            x.bo_prev = bo + (size_t) t * N; x.inj_prev = inj2 + (size_t) t * HC;
            x.w_norm = s.wn; x.w_inject = s.wi; x.q8_down = s.q8d; x.q8_up = s.q8u; x.eps = 1e-6f;
            x.lo = lo + (size_t) t * LR; x.rs = rs + (size_t) t * HC; x.inject_out = inj + (size_t) t * HC;
            x.mixed = mixed + (size_t) t * N;
        }
        return fa;
    }
    // 0 fill (flush_a), 1 copy (flush_b -> flush_c), 2 none, 3 fill of `alt`
    void flush(int how, int rep, float* alt) const {
        auto& q = Q();
        if (how == 0) q.fill(flush_a, (float) (rep & 7), flush_n);
        if (how == 3) q.fill(alt, (float) (rep & 7), flush_n);
        if (how == 1) {
            const float* src = flush_b;
            float* dst = flush_c;
            const float r = (float) rep;
            q.parallel_for(sycl::range<1>(flush_n), [=](sycl::id<1> i) { dst[i] = src[i] + r; });
        }
    }
    double time(const Set& s, int T, int how, int parts, float* alt = nullptr) const {
        const auto fa = args(s);
        auto& q = Q();
        k::fused_gr_q8_set_parts(7);
        k::fused_gr_read_multi(fa.data(), T, xn, &q);   // the partials a lone up reads
        k::fused_gr_q8_set_parts(parts);
        std::vector<double> v;
        for (int i = 0; i < warm + reps; ++i) {
            flush(how, i, alt);
            sycl::event a = sx::submit_profiling_tag(q);
            k::fused_gr_read_multi(fa.data(), T, xn, &q);
            sycl::event b = sx::submit_profiling_tag(q);
            b.wait_and_throw();
            const auto ta = a.get_profiling_info<sycl::info::event_profiling::command_end>();
            const auto tb = b.get_profiling_info<sycl::info::event_profiling::command_end>();
            if (i >= warm) v.push_back((double) (int64_t) (tb - ta) / 1e3);
        }
        k::fused_gr_q8_set_parts(7);
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
};

Set make_set(const std::string& name, size_t pad, size_t al, uint32_t seed) {
    if (pad > 0) (void) raw(pad);   // kept: it moves what follows
    const size_t bd = (size_t) LR * (D + D / 32 * 2), bu = (size_t) D * (LR + LR / 32 * 2);
    const auto get = [&](size_t bytes) { return al ? aligned(bytes, al) : raw(bytes); };
    Set s{name, static_cast<uint8_t*>(get(bd)), static_cast<uint8_t*>(get(bu)), static_cast<float*>(get((size_t) D * 4)),
          static_cast<uint16_t*>(get((size_t) HC * D * 2))};
    fill_bytes(s.q8d, bd, 21 + seed);
    fill_bytes(s.q8u, bu, 22 + seed);
    fill_f32(s.wn, D, 23, 0.5f, 1.5f);
    fill_bf16(s.wi, (size_t) HC * D, 24);
    Q().wait_and_throw();
    return s;
}

}  // namespace

int main(int argc, char** argv) try {
    Probe p{};
    p.reps = argc > 1 ? std::atoi(argv[1]) : 25;
    p.warm = argc > 2 ? std::atoi(argv[2]) : 3;
    const int flush_mib = argc > 3 ? std::atoi(argv[3]) : 128;
    k::fused_gr_set_q8_win(2);   // the live form
    auto& q = Q();
    std::printf("device %s, form %d, reps %d warm %d flush %d MiB\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), k::fused_gr_q8_win(), p.reps, p.warm, flush_mib);
    // the ubench's order: the flush buffer, then the window's activations, then (per phase) the weights
    p.flush_n = (size_t) flush_mib << 18;
    p.flush_a = static_cast<float*>(raw(p.flush_n * 4));
    p.R = static_cast<float*>(raw((size_t) MT * D * 4));
    p.bo = static_cast<float*>(raw((size_t) MT * N * 4));
    p.inj = static_cast<float*>(raw((size_t) MT * HC * 4));
    p.inj2 = static_cast<float*>(raw((size_t) MT * HC * 4));
    p.lo = static_cast<float*>(raw((size_t) MT * LR * 4));
    p.rs = static_cast<float*>(raw((size_t) MT * HC * 4));
    p.xn = static_cast<float*>(raw((size_t) MT * D * 4));
    p.mixed = static_cast<float*>(raw((size_t) MT * N * 4));
    p.flush_b = static_cast<float*>(raw(p.flush_n * 4));
    p.flush_c = static_cast<float*>(raw(p.flush_n * 4));
    float* flush_al = static_cast<float*>(aligned(p.flush_n * 4, kAlign2M));
    fill_f32(p.R, (size_t) MT * D, 11, -1, 1);
    fill_f32(p.bo, (size_t) MT * N, 12, -1, 1);
    fill_f32(p.inj, (size_t) MT * HC, 13, -1, 1);
    fill_f32(p.inj2, (size_t) MT * HC, 14, -1, 1);
    fill_f32(p.mixed, (size_t) MT * N, 15, -1, 1);
    fill_f32(p.flush_b, p.flush_n, 16, -1, 1);
    q.wait_and_throw();
    const auto m2 = [](const void* x) { return (unsigned long) ((uintptr_t) x % kAlign2M); };
    std::printf("buffers (addr mod 2 MiB / page size): flush %lx/%zu  flush(2M-aligned) %lx/%zu  R %lx/%zu  xn %lx/%zu\n",
                m2(p.flush_a), page_size(p.flush_a), m2(flush_al), page_size(flush_al), m2(p.R), page_size(p.R),
                m2(p.xn), page_size(p.xn));

    std::vector<Set> sets;
    sets.push_back(make_set("raw (ubench)", 0, 0, 0));
    sets.push_back(make_set("pad 4K", 4096, 0, 1));
    sets.push_back(make_set("pad 64K", 65536, 0, 2));
    sets.push_back(make_set("pad 1M", (size_t) 1 << 20, 0, 3));
    sets.push_back(make_set("pad 2M+4K", kAlign2M + 4096, 0, 4));
    sets.push_back(make_set("aligned 64K", 0, 65536, 5));
    sets.push_back(make_set("aligned 2M", 0, kAlign2M, 6));

    std::printf("\nus a call (median of %d), 16 rows unless noted; q8d/q8u: addr mod 2 MiB / page size\n", p.reps);
    std::printf("%-14s %-22s %-22s %8s %8s %8s %8s %8s %8s %8s\n", "set", "q8d", "q8u", "fill", "copy", "hot",
                "fill-al", "down", "up", "fill T8");
    const auto row = [&](const Set& s) {
        char a[32], b[32];
        std::snprintf(a, sizeof a, "%lx/%zu", m2(s.q8d), page_size(s.q8d));
        std::snprintf(b, sizeof b, "%lx/%zu", m2(s.q8u), page_size(s.q8u));
        std::printf("%-14s %-22s %-22s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n", s.name.c_str(), a, b,
                    p.time(s, 16, 0, 7), p.time(s, 16, 1, 7), p.time(s, 16, 2, 7), p.time(s, 16, 3, 7, flush_al),
                    p.time(s, 16, 0, 1), p.time(s, 16, 0, 4), p.time(s, 8, 0, 7));
        std::fflush(stdout);
    };
    for (const Set& s : sets) row(s);
    Set again = sets[0];
    again.name = "raw again";
    row(again);   // H3: the clock / card state moved during the run

    // ---- the prefetch (STRATA_HC_PREFETCH)
    const Set& s0 = sets[0];
    const size_t bd = (size_t) LR * (D + D / 32 * 2), bu = (size_t) D * (LR + LR / 32 * 2);
    const size_t xn4 = (size_t) 10700000 / 16;   // float4s: ssm_out's 10.7 MB of weights at 16 rows
    sycl::float4* xw = static_cast<sycl::float4*>(raw(xn4 * 16));
    float* sink = static_cast<float*>(raw(4096 * 4));
    fill_f32(reinterpret_cast<float*>(xw), xn4 * 4, 31, -1, 1);
    q.wait_and_throw();
    const auto stream_read = [&](int times) {
        for (int r = 0; r < times; ++r)
            q.parallel_for(sycl::nd_range<1>(4096 * 256, 256), [=](sycl::nd_item<1> it) {
                sycl::float4 a{0.f};
                for (size_t i = it.get_global_id(0); i < xn4; i += 4096 * 256) a += xw[i];
                const float v = sycl::reduce_over_group(it.get_group(), a.x() + a.y() + a.z() + a.w(), sycl::plus<float>());
                if (it.get_local_id(0) == 0) sink[it.get_group(0)] = v;
            });
    };
    const auto pf = [&] { k::l2_prefetch(s0.q8d, bd, s0.q8u, bu, &q); };
    const auto fa = p.args(s0);
    const auto read16 = [&] { k::fused_gr_read_multi(fa.data(), 16, p.xn, &q); };
    // copy flush, then pre(), tag, timed(), tag: the median us between the tags
    const auto med = [&](const std::function<void()>& pre, const std::function<void()>& timed) {
        std::vector<double> v;
        for (int i = 0; i < p.warm + p.reps; ++i) {
            p.flush(1, i, nullptr);
            pre();
            sycl::event a = sx::submit_profiling_tag(q);
            timed();
            sycl::event b = sx::submit_profiling_tag(q);
            b.wait_and_throw();
            const auto ta = a.get_profiling_info<sycl::info::event_profiling::command_end>();
            const auto tb = b.get_profiling_info<sycl::info::event_profiling::command_end>();
            if (i >= p.warm) v.push_back((double) (int64_t) (tb - ta) / 1e3);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const auto none = [] {};
    std::printf("\nprefetch (set 0, 16 rows, a copy flush before every call; us, median of %d):\n", p.reps);
    std::printf("  read: cold %.1f | hot %.1f | pf, read %.1f | x1, read %.1f | pf, x1, read %.1f | x8, read %.1f | "
                "pf, x8, read %.1f\n",
                med(none, read16), med(read16, read16), med(pf, read16), med([&] { stream_read(1); }, read16),
                med([&] { pf(); stream_read(1); }, read16), med([&] { stream_read(8); }, read16),
                med([&] { pf(); stream_read(8); }, read16));
    std::printf("  launches: pf alone %.1f | x1 alone %.1f | pf + x1 %.1f | x8 alone %.1f | pf + x8 %.1f\n",
                med(none, pf), med(none, [&] { stream_read(1); }), med(none, [&] { pf(); stream_read(1); }),
                med(none, [&] { stream_read(8); }), med(none, [&] { pf(); stream_read(8); }));
    // the read's bits with and without the prefetch ahead of it (it reads R in place: R reset before each)
    float* R0 = static_cast<float*>(raw((size_t) MT * D * 4));
    q.memcpy(R0, p.R, (size_t) MT * D * 4).wait();
    const auto outs = [&](bool with) {
        q.memcpy(p.R, R0, (size_t) MT * D * 4);
        if (with) pf();
        read16();
        q.wait_and_throw();
        std::vector<float> o((size_t) MT * (D + LR + HC + HC + N));
        float* h = o.data();
        q.memcpy(h, p.R, (size_t) MT * D * 4);
        q.memcpy(h + (size_t) MT * D, p.lo, (size_t) MT * LR * 4);
        q.memcpy(h + (size_t) MT * (D + LR), p.rs, (size_t) MT * HC * 4);
        q.memcpy(h + (size_t) MT * (D + LR + HC), p.inj, (size_t) MT * HC * 4);
        q.memcpy(h + (size_t) MT * (D + LR + 2 * HC), p.mixed, (size_t) MT * N * 4);
        q.wait_and_throw();
        return o;
    };
    const auto o0 = outs(false), o1 = outs(true);
    const bool same = std::memcmp(o0.data(), o1.data(), o0.size() * 4) == 0;
    std::printf("  bits (R, lo, rs, inject_out, mixed) with the prefetch ahead: %s\n", same ? "identical" : "DIFFER");
    return same ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "hc_mode_probe: %s\n", e.what());
    return 1;
}
