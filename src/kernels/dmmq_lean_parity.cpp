// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/dmmq_lean_parity.cpp - homelab densefuse (STRATA_DMMQ_LEAN): LeanKernel, the decode window's dense
// products in fewer messages a block, against iq_mmq_decode's arithmetic bit for bit.
//
// Everywhere (the OpenCL CPU device included, STRATA_TEST_CPU_DEVICE=1): LeanKernel's body without matrix engines, one
// matrix and a pair (two matrices on one xq in one launch, as the shared expert's gate and up), against the host's
// emulation of iq_mmq_decode (iq_mmq_decode_reference: the same decoded blocks, int32 block sums, scale order and K
// slices), memcmp; a second launch the same bits.  On a GPU where iq_mmq_decode runs (Xe2's XMX) also: LeanKernel's XMX
// body against iq_mmq_decode (DecodeKernel) and against the host, the pair against two iq_mmq_decode launches,
// iq_mmq_decode itself under STRATA_DMMQ_LEAN's kernel bit against it without, and the shared expert's products with
// the pair bit against without (memcmp); then each product timed, DecodeKernel against LeanKernel, the L2 flushed.
//
//     dmmq_lean_parity [--rows 1,7,16] [--reps 25] [--quick] [--no-time]
//
// --quick: the small shapes only (every type, every decoder); the default adds Flash-Next's dense shapes and types.
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/dmmq_contract.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "parity_device.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace P = strata::parity;
namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

std::mt19937& rng() {
    static std::mt19937 g{20261026u};
    return g;
}

// A GGUF block's values a block and the byte offsets of its FP16 scales (ggml-common.h's layouts): random block
// bytes with those scales small and positive, so no decoded scale is a NaN or an infinity.
struct Layout { int qk; std::vector<int> half_at; };
Layout layout(int t) {
    switch (t) {
        case 6: case 8: case 20: return {32, {0}};
        case 42: return {64, {0}};
        case 11: return {256, {108}};
        case 12: case 13: return {256, {0, 2}};
        case 14: return {256, {208}};
        case 16: case 17: case 18: case 21: case 22: case 23: return {256, {0}};
        default: return {0, {}};
    }
}

std::vector<uint8_t> weights(int t, int64_t rows, int64_t cols) {
    const Layout l = layout(t);
    const size_t rb = k::iq_mmq_row_bytes(t, cols), bs = rb / (size_t) (cols / l.qk);
    std::vector<uint8_t> w(rb * (size_t) rows);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& b : w) b = (uint8_t) byte(rng());
    std::uniform_real_distribution<float> d(0.002f, 0.02f);
    for (size_t at = 0; at < w.size(); at += bs)
        for (int o : l.half_at) {
            const sycl::half h = d(rng());
            std::memcpy(&w[at + (size_t) o], &h, 2);
        }
    return w;
}

// iq_mmq_decode's activations for `rows` rows of `cols`: int8 values, a scale a block of 32, the scale times the
// block's integer sum (what iq_mmq_quantize writes)
std::vector<uint8_t> activations(int64_t rows, int64_t cols) {
    const int64_t nb = cols / 32;
    std::vector<uint8_t> xq(k::iq_mmq_act_bytes(rows, cols));
    auto* x8 = reinterpret_cast<int8_t*>(xq.data());
    auto* xs = reinterpret_cast<float*>(xq.data() + rows * cols);
    std::uniform_int_distribution<int> v(-127, 127);
    std::uniform_real_distribution<float> d(0.001f, 0.05f);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            int sum = 0;
            for (int j = 0; j < 32; ++j) sum += x8[r * cols + b * 32 + j] = (int8_t) v(rng());
            xs[r * nb + b] = d(rng());
            xs[rows * nb + r * nb + b] = xs[r * nb + b] * (float) sum;
        }
    return xq;
}

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
    for (int i = 0; i < 10; ++i) once(i);
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) v.push_back(once(i));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Case { const char* name; int64_t n_out, n_in; std::vector<int> types; bool pair; };

int checks = 0, bad = 0;
void verdict(const char* what, const Case& c, int t, int rows, bool ok) {
    ++checks;
    bad += !ok;
    std::printf("  %-26s %-10s t%-10d %6lld x %6lld rows %2d: %s\n", what, c.name, t, (long long) c.n_out,
                (long long) c.n_in, rows, ok ? "same bits" : "*** MISMATCH ***");
    std::fflush(stdout);
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main(int argc, char** argv) try {
    std::vector<int> rows_list{1, 2, 7, 8, 9, 15, 16, 17, 24, 32};
    int reps = 25;
    bool quick = false, do_time = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) rows_list = k::dmmq::parse_rows(argv[++i]);
        else if (a == "--reps" && i + 1 < argc)
            reps = k::dmmq::positive_int(argv[++i], std::numeric_limits<int>::max());
        else if (a == "--quick") quick = true;
        else if (a == "--no-time") do_time = false;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    // STRATA_DMMQ_LEAN's parse: off by default, bits kept, others dropped
    {
        const bool ok = k::dmmq::lean_bits(nullptr) == 0 && k::dmmq::lean_bits("") == 0 &&
                        k::dmmq::lean_bits("0") == 0 && k::dmmq::lean_bits("1") == 1 &&
                        k::dmmq::lean_bits("7") == 7 && k::dmmq::lean_bits("15") == 7 && k::dmmq::lean_bits("6") == 6;
        ++checks;
        bad += !ok;
        std::printf("  STRATA_DMMQ_LEAN parse: %s\n", ok ? "ok" : "*** FAIL ***");
    }
    auto& q = Q();
    const bool xmx = k::iq_mmq_decode_ok(q, 23, 16, 256, 1);
    std::printf("dmmq_lean_parity: %s, XMX body %s, rows", q.get_device().get_info<sycl::info::device::name>().c_str(),
                xmx ? "checked" : "not here (portable body only)");
    for (int r : rows_list) std::printf(" %d", r);
    std::printf("\n");
    // every decoder on small shapes (a scale per 16 or per 32, with and without a min term, the row-split layouts),
    // then Flash-Next's dense products as R8 serves them (dmmq_parity's list: GDN qkv / z / ssm_out, QSA q|gate / o /
    // k|v, the shared expert's gate|up (a pair) and down, the hc reads' Q8_0 rows)
    std::vector<Case> cases = {
        {"small", 64, 512,
         {6, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 42, k::kNativeQ6KRows, k::kNativeQ8Rows}, true},
        {"small_k", 48, 2048, {12, 14, 23, k::kNativeQ6KRows}, false},
    };
    if (!quick) {
        const std::vector<Case> model = {
            {"qkv", 10240, 2560, {23, k::kNativeQ6KRows, 12, 13}, false},
            {"z", 6144, 2560, {21, k::kNativeQ6KRows, 23, 12}, false},
            {"ssm_out", 2560, 6144, {23, 21, 12, 13, k::kNativeQ6KRows}, false},
            {"q|gate", 12288, 2560, {23, 12}, false},
            {"o", 2560, 6144, {k::kNativeQ6KRows, 13}, false},
            {"k|v", 512, 2560, {23, k::kNativeQ6KRows, 12}, false},
            {"shexp_gu", 640, 2560, {23, 21, 12, 13, k::kNativeQ6KRows}, true},
            {"shexp_down", 2560, 640, {20, 42}, false},
            {"hc_q8", 320, 10240, {k::kNativeQ8Rows}, false},
            {"hc_q8_up", 10240, 320, {k::kNativeQ8Rows}, false},
        };
        cases.insert(cases.end(), model.begin(), model.end());
    }
    const int MT = k::dmmq::kMaxRows;
    P::Device base;
    const size_t flush_n = (size_t) 128 << 18;
    float* flush = (xmx && do_time) ? base.alloc<float>(flush_n) : nullptr;
    for (const Case& c : cases) {
        for (int t : c.types) {
            const int64_t N = c.n_out, K = c.n_in;
            if (!k::dmmq::shape_ok(t, N, K, MT)) { verdict("shape not taken", c, t, MT, false); continue; }
            // the GGUF type of the bytes; the row-split layouts are made from them on the device, as the engine does
            const int gt = t == k::kNativeQ6KRows ? 14 : t == k::kNativeQ8Rows ? 8 : t;
            const int nw = c.pair ? 2 : 1;
            P::Device dev;
            std::vector<std::vector<uint8_t>> w(nw);
            const void* dw[2] = {nullptr, nullptr};
            for (int m = 0; m < nw; ++m) {
                w[m] = weights(gt, N, K);
                uint8_t* d = dev.upload(w[m]);
                if (t == k::kNativeQ6KRows) k::native_q6_k_to_rows(d, (int) K, (int) N, P::stream());
                if (t == k::kNativeQ8Rows) k::native_q8_0_to_rows(d, (int) K, (int) N, P::stream());
                P::sync();
                if (t != gt) P::copy_out(w[m].data(), d, w[m].size());   // the host's reference reads the same layout
                dw[m] = d;
            }
            float* dy[2] = {dev.alloc<float>((size_t) (MT * N)), dev.alloc<float>((size_t) (MT * N))};
            float* dz[2] = {dev.alloc<float>((size_t) (MT * N)), dev.alloc<float>((size_t) (MT * N))};
            for (int T : rows_list) {
                const std::vector<uint8_t> xq = activations(T, K);
                uint8_t* dxq = dev.upload(xq);
                std::vector<std::vector<float>> ref(nw, std::vector<float>((size_t) (T * N)));
                for (int m = 0; m < nw; ++m)
                    k::iq_mmq_decode_reference(t, w[m].data(), N, K, xq.data(), T, ref[m].data(), N);
                auto out = [&](float* const* d, int m) {
                    std::vector<float> y((size_t) (T * N));
                    P::copy_out(y.data(), d[m], y.size());
                    return y;
                };
                // the portable body: one matrix, again, and the pair
                k::iq_mmq_decode_lean_run(q, t, dw, 1, N, K, dxq, T, dy, N, false).wait_and_throw();
                k::iq_mmq_decode_lean_run(q, t, dw, 1, N, K, dxq, T, dz, N, false).wait_and_throw();
                const std::vector<float> p0 = out(dy, 0);
                verdict("portable vs host", c, t, T, same(p0, ref[0]));
                verdict("portable rerun", c, t, T, same(p0, out(dz, 0)));
                if (nw == 2) {
                    k::iq_mmq_decode_lean_run(q, t, dw, 2, N, K, dxq, T, dy, N, false).wait_and_throw();
                    verdict("portable pair vs host", c, t, T, same(out(dy, 0), ref[0]) && same(out(dy, 1), ref[1]));
                }
                if (!xmx || !k::iq_mmq_decode_ok(q, t, N, K, T)) continue;
                // the XMX body against DecodeKernel (iq_mmq_decode with the lean bits off) and the host
                k::iq_mmq_decode_set_lean(0);
                k::iq_mmq_decode(q, t, dw[0], N, K, dxq, T, dz[0], N).wait_and_throw();
                const std::vector<float> served = out(dz, 0);
                verdict("DecodeKernel vs host", c, t, T, same(served, ref[0]));
                k::iq_mmq_decode_lean_run(q, t, dw, 1, N, K, dxq, T, dy, N, true).wait_and_throw();
                verdict("lean XMX vs DecodeKernel", c, t, T, same(out(dy, 0), served));
                k::iq_mmq_decode_set_lean(k::dmmq::kLeanKernel);
                k::iq_mmq_decode(q, t, dw[0], N, K, dxq, T, dy[0], N).wait_and_throw();
                verdict("engine lean bit vs off", c, t, T, same(out(dy, 0), served));
                if (nw == 2) {
                    k::iq_mmq_decode_set_lean(0);
                    k::iq_mmq_decode(q, t, dw[1], N, K, dxq, T, dz[1], N).wait_and_throw();
                    k::iq_mmq_decode_set_lean(k::dmmq::kLeanPair);
                    const bool taken = k::iq_mmq_decode_pair(q, t, dw[0], dw[1], N, K, dxq, T, dy[0], dy[1], N);
                    q.wait_and_throw();
                    verdict("XMX pair vs two launches", c, t, T,
                            taken && same(out(dy, 0), served) && same(out(dy, 1), out(dz, 1)));
                }
                k::iq_mmq_decode_set_lean(-1);
            }
            if (!xmx || !do_time || !k::iq_mmq_decode_ok(q, t, N, K, MT)) continue;
            std::printf("  time %-10s t%-10d %6lld x %6lld (%.2f MB):", c.name, t, (long long) N, (long long) K,
                        (double) w[0].size() / 1e6);
            for (int T : rows_list) {
                const std::vector<uint8_t> xq = activations(T, K);
                uint8_t* dxq = dev.upload(xq);
                k::iq_mmq_decode_set_lean(0);
                const double us_d = time_us(flush, flush_n, reps, [&] {
                    k::iq_mmq_decode(q, t, dw[0], N, K, dxq, T, dy[0], N);
                });
                const double us_l = time_us(flush, flush_n, reps, [&] {
                    k::iq_mmq_decode_lean_run(q, t, dw, 1, N, K, dxq, T, dy, N, true);
                });
                std::printf("  r%d decode %.1f lean %.1f", T, us_d, us_l);
                if (nw == 2) {
                    const double us_2 = time_us(flush, flush_n, reps, [&] {
                        k::iq_mmq_decode(q, t, dw[0], N, K, dxq, T, dy[0], N);
                        k::iq_mmq_decode(q, t, dw[1], N, K, dxq, T, dy[1], N);
                    });
                    const double us_p = time_us(flush, flush_n, reps, [&] {
                        k::iq_mmq_decode_lean_run(q, t, dw, 2, N, K, dxq, T, dy, N, true);
                    });
                    std::printf(" two %.1f pair %.1f", us_2, us_p);
                }
                k::iq_mmq_decode_set_lean(-1);
                std::fflush(stdout);
            }
            std::printf("  | floor %.1f us at 608 GB/s\n", (double) w[0].size() / 608e3);
        }
    }
    // the shared expert whole on the decode products (STRATA_SHEXP_XMX): the pair bit against without, memcmp
    if (xmx) {
        const int64_t E = 2560, FF = 640;
        for (const auto& [gt, ut, dt] : std::vector<std::array<int, 3>>{{23, 23, 20}, {21, 21, 42}, {12, 13, 20}}) {
            if (!k::iq_mmq_decode_ok(q, gt, FF, E, MT) || !k::iq_mmq_decode_ok(q, ut, FF, E, MT) ||
                !k::iq_mmq_decode_ok(q, dt, E, FF, MT))
                continue;
            P::Device dev;
            k::NativeSharedWeights nw;
            nw.gate_type = gt; nw.up_type = ut; nw.down_type = dt;
            nw.gate_data = dev.upload(weights(gt, FF, E));
            nw.up_data = dev.upload(weights(ut, FF, E));
            nw.down_data = dev.upload(weights(dt, E, FF));
            nw.q8_1 = dev.alloc<uint8_t>(k::native_q8_1_bytes((int) E, MT));
            std::vector<float> x((size_t) (MT * E));
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (float& v : x) v = nd(rng());
            float* dx = dev.upload(x);
            float* sg = dev.alloc<float>((size_t) (MT * FF));
            float* su = dev.alloc<float>((size_t) (MT * FF));
            float* g = dev.alloc<float>(MT);
            float* o1 = dev.alloc<float>((size_t) (MT * E));
            float* o2 = dev.alloc<float>((size_t) (MT * E));
            k::shared_expert_set_native_bf16(true);
            k::shared_expert_set_xmx(1);
            const Case c{"shexp", E, FF, {}, true};
            for (int T : rows_list) {
                k::iq_mmq_decode_set_lean(0);
                k::shared_expert_multi(T, dx, nullptr, nw, nullptr, sg, su, g, o1, E, FF, P::stream(), true, true,
                                       nullptr, false);
                k::iq_mmq_decode_set_lean(k::dmmq::kLeanPair | k::dmmq::kLeanKernel);
                k::shared_expert_multi(T, dx, nullptr, nw, nullptr, sg, su, g, o2, E, FF, P::stream(), true, true,
                                       nullptr, false);
                q.wait_and_throw();
                std::vector<float> a((size_t) (T * E)), b(a.size());
                P::copy_out(a.data(), o1, a.size());
                P::copy_out(b.data(), o2, b.size());
                verdict(gt == ut ? "shexp lean+pair vs off" : "shexp lean (no pair) vs off", c, gt, T, same(a, b));
            }
            k::iq_mmq_decode_set_lean(-1);
            k::shared_expert_set_xmx(0);
        }
    }
    std::printf("dmmq_lean_parity: %d checks, %d failures\n", checks, bad);
    const bool ok = k::dmmq::checks_ok(checks, bad);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "dmmq_lean_parity: %s\n", e.what());
    return 1;
}
