// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/dmmq_parity.cpp - homelab K1: iq_mmq_decode (STRATA_BATCH_DENSE_XMX), the decode window's dense products on
// Xe2's XMX, against the host in FP64 (the device's int8 activations times their scales, the weights through ggml's
// dequantizer, as iq_mmq_parity) and against iq_mmq_grouped (the prompt tiles: the same operands, other sum order), at
// 1..16 rows on Qwen3.8-Flash-Next's dense shapes and types; then each product timed with the L2 flushed: the q8_1 GEMV
// (native_mmvq, as the engine runs it under 4 rows), iq_mmq_grouped and iq_mmq_decode, each with its quantize.
//
//     dmmq_parity [--rows 1,7,16] [--reps 25] [--no-check] [--no-time] [--ks-sweep]
//
// Exit 77 (skipped) where iq_mmq_decode does not run (no Xe2 XMX).  A check passes at rel < 1e-4 of the reference's
// largest magnitude (iq_mmq_parity's bound); a second launch must give the same bits (no atomics in the K split).
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/dmmq_contract.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "parity_device.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace P = strata::parity;
namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

struct XorShift32 {
    using result_type = uint32_t;
    uint32_t state;
    static constexpr result_type min() { return 1; }
    static constexpr result_type max() { return 0xFFFFFFFFu; }
    result_type operator()() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};
XorShift32& rng() {
    static XorShift32 g{20261024u};
    return g;
}

bool ggml_known(int t) { return t >= 0 && t < (int) GGML_TYPE_COUNT && ggml_get_type_traits((ggml_type) t)->to_float; }

// rows of `cols` values of GGUF type t: ggml's quantizer where it needs no importance matrix, else random block bytes
// whose fp16 scale (the block's first two bytes, and Q4_K / Q5_K's next two) is small and positive
std::vector<uint8_t> weights(int t, int64_t rows, int64_t cols) {
    const size_t rb = k::iq_mmq_row_bytes(t, cols);
    std::vector<uint8_t> w(rb * (size_t) rows);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    if (ggml_known(t) && !ggml_quantize_requires_imatrix((ggml_type) t) && t != GGML_TYPE_IQ3_XXS) {
        std::vector<float> f((size_t) (rows * cols));
        for (float& v : f) v = u(rng());
        ggml_quantize_chunk((ggml_type) t, f.data(), w.data(), 0, rows, cols, nullptr);
        return w;
    }
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& b : w) b = (uint8_t) byte(rng());
    const size_t bs = ggml_known(t) ? ggml_type_size((ggml_type) t) : k::iq_mmq_row_bytes(t, 64);   // Q2_0: 64 a block
    std::uniform_real_distribution<float> d(0.002f, 0.02f);
    for (size_t at = 0; at + 2 <= w.size(); at += bs) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(d(rng()));
        std::memcpy(&w[at], &h, 2);
    }
    return w;
}

struct Shape { const char* name; int64_t n_out, n_in; std::vector<int> types; };

sycl::queue& Q() { return strata::core::Runtime::get().compute(); }

// the median gap of profiling tags around f, the L2 flushed (a 128 MiB fill) before each
double time_us(float* flush, size_t flush_n, int reps, const std::function<void()>& f) {
    auto& q = Q();
    auto once = [&](int rep) {
        q.fill(flush, (float) (rep & 7), flush_n);
        sycl::event a = sx::submit_profiling_tag(q);
        f();
        sycl::event b = sx::submit_profiling_tag(q);
        b.wait_and_throw();
        return (double) (int64_t) (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                   a.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e3;
    };
    for (int i = 0; i < 20; ++i) once(i);
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) v.push_back(once(i));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) try {
    std::vector<int> rows_list{1, 2, 3, 7, 8, 9, 12, 16, 17, 24, 31, 32};
    int reps = 25;
    bool do_check = true, do_time = true, ks_sweep = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) rows_list = k::dmmq::parse_rows(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = k::dmmq::positive_int(argv[++i], std::numeric_limits<int>::max());
        else if (a == "--no-check") do_check = false;
        else if (a == "--no-time") do_time = false;
        else if (a == "--ks-sweep") ks_sweep = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (!do_check && !do_time) throw std::invalid_argument("neither checks nor timing requested");
    auto& q = Q();
    if (!k::iq_mmq_decode_ok(q, 23, 16, 256, 1)) {
        std::printf("dmmq_parity: iq_mmq_decode does not run on %s: skipped\n",
                    q.get_device().get_info<sycl::info::device::name>().c_str());
        return 77;
    }
    std::printf("dmmq_parity: %s, rows %d..%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                rows_list.front(), rows_list.back());
    // Flash-Next's dense products (spec-0-23: GDN qkv / z / ssm_out, QSA q|gate / k / o, the shared expert's three)
    const std::vector<Shape> shapes = {
        {"qkv", 10240, 2560, {23, 14, 12, 13}},
        {"z", 6144, 2560, {21, 14, 23, 12}},
        {"ssm_out", 2560, 6144, {23, 21, 12, 13, 14, k::kNativeQ6KRows}},
        {"q|gate", 12288, 2560, {23, 12}},
        {"o", 2560, 6144, {14, 13}},
        {"k|v", 512, 2560, {23, 14, 12}},
        {"shexp_gu", 640, 2560, {23, 21, 12, 13, 14}},
        {"shexp_down", 2560, 640, {20, 42}},
        {"hc_q8", 320, 10240, {8, k::kNativeQ8Rows}},
        {"hc_q8_up", 10240, 320, {8, k::kNativeQ8Rows}},
    };
    const int MT = k::dmmq::kMaxRows;
    P::Device base;
    const size_t flush_n = (size_t) 128 << 18;
    float* flush = base.alloc<float>(flush_n);
    int bad = 0, checks = 0;
    for (const Shape& sh : shapes) {
        for (int t : sh.types) {
            const int gt = t == k::kNativeQ6KRows ? 14 : t == k::kNativeQ8Rows ? 8 : t;   // the GGUF type of the bytes
            const int64_t N = sh.n_out, K = sh.n_in;
            if (!k::iq_mmq_decode_ok(q, t, N, K, MT)) {
                std::printf("  %-10s t%-5d not taken\n", sh.name, t);
                if (do_check) { ++checks; ++bad; }
                continue;
            }
            std::vector<uint8_t> w = weights(gt, N, K);
            std::vector<float> x((size_t) (MT * K));
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (float& v : x) v = nd(rng());
            P::Device dev;
            uint8_t* dw = dev.upload(w);
            if (t == k::kNativeQ6KRows) k::native_q6_k_to_rows(dw, (int) K, (int) N, P::stream());
            if (t == k::kNativeQ8Rows) k::native_q8_0_to_rows(dw, (int) K, (int) N, P::stream());
            float* dx = dev.upload(x);
            const size_t xq_bytes = k::iq_mmq_act_bytes(MT, K);
            uint8_t* dxq = dev.alloc<uint8_t>(xq_bytes);
            uint8_t* q81 = dev.alloc<uint8_t>(k::native_q8_1_bytes((int) K, MT));
            float* dy = dev.alloc<float>((size_t) (MT * N));
            float* dy2 = dev.alloc<float>((size_t) (MT * N));
            float* dm = dev.alloc<float>((size_t) (MT * N));
            std::vector<int32_t> bounds{0, 0}, ids(MT);
            for (int i = 0; i < MT; ++i) ids[(size_t) i] = i;
            int32_t* di = dev.upload(ids);
            int32_t* db = dev.alloc<int32_t>(2);
            const bool sums = k::iq_mmq_type_sums(t);
            const bool mmq_ok = N % 128 == 0;
            std::vector<float> wf;
            if (do_check && ggml_known(gt)) {   // the weights in FP32 by ggml (the original block layout)
                wf.resize((size_t) (N * K));
                const auto* tr = ggml_get_type_traits((ggml_type) gt);
                const size_t rb = k::iq_mmq_row_bytes(gt, K);
                for (int64_t o = 0; o < N; ++o) tr->to_float(w.data() + (size_t) o * rb, &wf[(size_t) (o * K)], K);
            }
            for (int T : rows_list) {
                bounds[1] = T;
                P::copy_in(db, bounds.data(), 2);
                k::iq_mmq_quantize(q, dx, nullptr, dxq, K, K, T, sums, false).wait();
                if (do_check) {
                    k::iq_mmq_decode(q, t, dw, N, K, dxq, T, dy, N).wait();
                    k::iq_mmq_decode(q, t, dw, N, K, dxq, T, dy2, N).wait();
                    if (mmq_ok) k::iq_mmq_grouped(q, t, dw, 0, N, K, 1, dxq, T, db, T, di, dm, N).wait();
                    std::vector<float> y((size_t) (T * N)), y2(y.size()), m(y.size());
                    std::vector<uint8_t> xq(xq_bytes);
                    P::copy_out(y.data(), dy, y.size());
                    P::copy_out(y2.data(), dy2, y2.size());
                    if (mmq_ok) P::copy_out(m.data(), dm, m.size());
                    P::copy_out(xq.data(), dxq, xq_bytes);
                    const bool same = std::memcmp(y.data(), y2.data(), y.size() * 4) == 0;
                    k::dmmq::Error fp64_error, mmq_error;
                    const auto* x8 = reinterpret_cast<const int8_t*>(xq.data());
                    const auto* xs = reinterpret_cast<const float*>(xq.data() + (size_t) (T * K));
                    for (int64_t r = 0; r < T; ++r)
                        for (int64_t o = 0; o < N; ++o) {
                            const double yv = y[(size_t) (r * N + o)];
                            if (mmq_ok) mmq_error.add(yv, m[(size_t) (r * N + o)]);
                            if (wf.empty()) continue;
                            double s = 0.0;
                            for (int64_t kk = 0; kk < K; ++kk)
                                s += (double) x8[r * K + kk] * xs[r * (K / 32) + kk / 32] * wf[(size_t) (o * K + kk)];
                            fp64_error.add(yv, s);
                        }
                    const double rel = fp64_error.relative();
                    const double rel_m = mmq_ok ? mmq_error.relative() : -1;
                    // FP64 is mandatory: an unavailable oracle is a missing check, not permission to pass against mmq alone.
                    const bool ok = same && fp64_error.ok(1e-4) && (!mmq_ok || mmq_error.ok(1e-4));
                    ++checks;
                    bad += !ok;
                    char a[32], b[32];
                    std::snprintf(a, sizeof a, rel < 0 ? "(no ggml)" : "%.2e", rel);
                    std::snprintf(b, sizeof b, rel_m < 0 ? "(n/a)" : "%.2e", rel_m);
                    std::printf("  check %-10s t%-5d %6lld x %6lld rows %2d: vs FP64 %s  vs mmq %s  rerun %s  %s\n",
                                sh.name, t, (long long) N, (long long) K, T, a, b, same ? "same bits" : "DIFFERENT",
                                ok ? "ok" : "*** FAIL ***");
                    std::fflush(stdout);
                }
            }
            if (!do_time) continue;
            k::native_quantize_q8_1(dx, q81, (int) K, MT, P::stream());
            q.wait();
            std::printf("  time %-10s t%-5d %6lld x %6lld (%.2f MB):", sh.name, t, (long long) N, (long long) K,
                        (double) w.size() / 1e6);
            for (int T : rows_list) {
                bounds[1] = T;
                P::copy_in(db, bounds.data(), 2);
                const double us_v = k::native_mmvq_supported(t)
                    ? time_us(flush, flush_n, reps, [&] {
                        k::native_quantize_q8_1(dx, q81, (int) K, T, P::stream());
                        k::native_mmvq(t, dw, q81, dy, (int) K, (int) N, T, P::stream());
                    }) : -1;
                const double us_m = mmq_ok ? time_us(flush, flush_n, reps, [&] {
                    k::iq_mmq_quantize(q, dx, nullptr, dxq, K, K, T, sums, false);
                    k::iq_mmq_grouped(q, t, dw, 0, N, K, 1, dxq, T, db, T, di, dm, N);
                }) : -1;
                const double us_x = time_us(flush, flush_n, reps, [&] {
                    k::iq_mmq_quantize(q, dx, nullptr, dxq, K, K, T, sums, false);
                    k::iq_mmq_decode(q, t, dw, N, K, dxq, T, dy, N);
                });
                std::printf("  r%d gemv %.1f mmq %.1f xmx %.1f", T, us_v, us_m, us_x);
                std::fflush(stdout);
            }
            std::printf("  | floor %.1f us at 540 GB/s\n", (double) w.size() / 540e3);
            if (ks_sweep) {
                std::printf("    ks sweep at %d rows:", rows_list.back());
                for (int ks : {1, 2, 4, 8, 16}) {
                    k::iq_mmq_decode_set_split(ks, 2048);
                    const double us = time_us(flush, flush_n, reps, [&] {
                        k::iq_mmq_decode(q, t, dw, N, K, dxq, rows_list.back(), dy, N);
                    });
                    std::printf("  ks%d %.1f", ks, us);
                }
                k::iq_mmq_decode_set_split(-1, -1);   // restore the caller's environment, not a different auto/SGS setting
                std::printf("\n");
            }
        }
    }
    // the shared expert whole (STRATA_SHEXP_XMX): gate, up, SwiGLU, down on iq_mmq_decode against the q8_1 GEMV path
    // (the same int8 forms of x; the SwiGLU's form may round a value apart where gate / up sum in another order, so the
    // bound is 1e-2 of the largest output), both raw and fully sigmoid-gated rows; then the raw products timed
    for (const auto& [gt_, ut_, dt_] : std::vector<std::tuple<int, int, int>>{{23, 21, 20}, {12, 13, 20}, {23, 21, 42}}) {
        const int64_t N = 2560, FF = 640;
        const int gt = gt_, ut = ut_, dt = dt_;
        if (!k::iq_mmq_decode_ok(q, gt, FF, N, MT) || !k::iq_mmq_decode_ok(q, ut, FF, N, MT) ||
            !k::iq_mmq_decode_ok(q, dt, N, FF, MT)) {
            std::printf("  shexp t%d/%d/%d not taken\n", gt, ut, dt);
            if (do_check) { ++checks; ++bad; }
            continue;
        }
        P::Device dev;
        k::NativeSharedWeights nw;
        nw.gate_type = gt; nw.up_type = ut; nw.down_type = dt;
        nw.gate_data = dev.upload(weights(gt, FF, N));
        nw.up_data = dev.upload(weights(ut, FF, N));
        nw.down_data = dev.upload(weights(dt, N, FF));
        nw.q8_1 = dev.alloc<uint8_t>(k::native_q8_1_bytes((int) N, MT));
        std::vector<float> x((size_t) (MT * N));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : x) v = nd(rng());
        float* dx = dev.upload(x);
        uint8_t* xq = dev.alloc<uint8_t>(k::native_q8_1_bytes((int) N, MT));
        float* sg = dev.alloc<float>((size_t) (MT * FF));
        float* su = dev.alloc<float>((size_t) (MT * FF));
        float* g = dev.alloc<float>(MT);
        float* o1 = dev.alloc<float>((size_t) (MT * N));
        float* o2 = dev.alloc<float>((size_t) (MT * N));
        float* o3 = dev.alloc<float>((size_t) (MT * N));
        std::vector<uint16_t> gate_input((size_t) N);
        for (uint16_t& v : gate_input) v = k::bf16_from_f32(nd(rng()) * 0.01f);
        uint16_t* dgi = dev.upload(gate_input);
        k::native_quantize_q8_1(dx, xq, (int) N, MT, P::stream());
        q.wait();
        k::shared_expert_set_native_bf16(true);
        auto run = [&](int xmx, int T, float* out, bool gated = false) {
            k::shared_expert_set_xmx(xmx);
            k::shared_expert_multi(T, dx, nullptr, nw, gated ? dgi : nullptr, sg, su, g, out, N, FF, P::stream(),
                                   !gated, !gated, xq, true);   // native_quantize_q8_1 above, not the expert quantizer
        };
        if (do_check)
            for (int T : rows_list)
                for (bool gated : {false, true}) {
                    run(0, T, o1, gated);
                    run(1, T, o2, gated);
                    run(1, T, o3, gated);
                    q.wait_and_throw();
                    std::vector<float> a((size_t) (T * N)), b(a.size()), c(a.size());
                    P::copy_out(a.data(), o1, a.size());
                    P::copy_out(b.data(), o2, b.size());
                    P::copy_out(c.data(), o3, c.size());
                    k::dmmq::Error error;
                    for (size_t i = 0; i < a.size(); ++i) error.add(b[i], a[i]);
                    const bool same = std::memcmp(b.data(), c.data(), b.size() * sizeof(float)) == 0;
                    const bool ok = same && error.ok(1e-2);
                    ++checks;
                    bad += !ok;
                    std::printf("  check shexp %s t%d/%d/%d rows %2d: vs gemv path %.2e  rerun %s  %s\n",
                                gated ? "whole" : "raw", gt, ut, dt, T, error.relative(), same ? "same bits" : "DIFFERENT",
                                ok ? "ok" : "*** FAIL ***");
                    std::fflush(stdout);
                }
        if (do_time) {
            std::printf("  time shexp t%d/%d/%d:", gt, ut, dt);
            for (int T : rows_list) {
                const double us_v = time_us(flush, flush_n, reps, [&] { run(0, T, o1); });
                const double us_x = time_us(flush, flush_n, reps, [&] { run(1, T, o2); });
                std::printf("  r%d gemv %.1f xmx %.1f", T, us_v, us_x);
                std::fflush(stdout);
            }
            std::printf("\n");
        }
        k::shared_expert_set_xmx(0);
    }
    std::printf("dmmq_parity: %d checks, %d failures\n", checks, bad);
    if (!do_check) { std::printf("TIMING ONLY (parity not checked)\n"); return 0; }
    const bool ok = k::dmmq::checks_ok(checks, bad);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "dmmq_parity: %s\n", e.what());
    return 1;
}
