// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/expert_xmx_parity.cpp - homelab: the routed experts' grouped product on XMX (STRATA_EXPERT_XMX,
// native_expert_grouped's kNativeExpertFastXmx) against the control (mask 0, 4 entries a pass), bitwise.
//
//     build/expert_xmx_parity [--pairs 18/20,17/42,18:21/20,...] [--rows 1,2,...,32] [--dims 2560,640] [--sets 1]
//                             [--bodies emulated,xmx] [--seed 353] [--gate-up] [--time 1,8,16,24,32] [--reps 25]
//
// Every expert type pair of Qwen3.8-Flash-Next's files (gate/up IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S; down IQ4_NL,
// Q2_0) and a mixed layer (gate:up/down), synthetic blobs (random bytes, every block's FP16 scale finite), random
// q8_1 activations, and three routings: random (K distinct of 512 a row), shared (every row the same K: groups of
// every row, more than one pass of 8 entries from 9 rows), mixed (rows drawing from a pool of 24).  Each plan is the
// engine's resident_plan.  For every pair, row count and routing the control and each body write into sentinel
// buffers, and are compared with memcmp: the output rows, the q8_1 h the down product reads, and with --gate-up the
// FP32 gate and up rows (the control writes them only off mode 8: run with STRATA_EXP_MODE=0).  Each candidate call
// must take the XMX path (native_expert_xmx_taken), and a call with no group must write nothing.
// Bodies: `emulated` (the lane emulation of the tiles: the CPU device's proof of the gather, the packed B layout, the
// parts and the float rebuild) and `xmx` (joint_matrix; only where the device has Intel's int8 tiles).
// --time: the control and each body timed at those rows (random routing, the L2 flushed by a fill; a GPU's check).
// Exit status: 0 when every check passed, else 1.
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/window_capacity.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

namespace {

sycl::queue& Q() { return strata::core::Runtime::get().compute(); }
void* CS() { return &Q(); }

template <class T> T* dev(size_t n) {
    auto& rt = strata::core::Runtime::get();
    void* p = sycl::malloc_device(std::max<size_t>(n * sizeof(T), 64) + 64, rt.device(), rt.context());
    if (p == nullptr) throw std::runtime_error("expert_xmx_parity: device allocation failed");
    return static_cast<T*>(p);
}
void release(void* p) { sycl::free(p, strata::core::Runtime::get().context()); }

int qk_of(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }
// a finite FP16 in [2^-10, 2^-6) with a random sign
uint16_t half_scale(std::mt19937& rng) { return (uint16_t) (((rng() & 1u) << 15) | ((5u + rng() % 4u) << 10) | (rng() & 0x3ffu)); }

std::vector<uint8_t> make_blob(const k::NativeExpertLayout& L, std::mt19937& rng) {
    std::vector<uint8_t> b(L.bytes);
    for (auto& v : b) v = (uint8_t) rng();
    auto fix = [&](size_t off, int64_t rows, size_t row_bytes, int t, int64_t n_in) {
        const int64_t nb = n_in / qk_of(t);
        const size_t bsz = row_bytes / (size_t) nb;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < nb; ++j) {
                const uint16_t h = half_scale(rng);
                std::memcpy(&b[off + (size_t) r * row_bytes + (size_t) j * bsz], &h, 2);
            }
    };
    fix(0, L.n_ff, L.gu_row, L.gu_type, L.n_embd);
    fix(L.up_off, L.n_ff, L.up_row, L.up_type, L.n_embd);
    fix(L.down_off, L.n_embd, L.d_row, L.d_type, L.n_ff);
    return b;
}

// q8_1 rows: random int8 values (the full -127..127), d a modest FP16 that differs block to block, s = d * sum
std::vector<uint8_t> make_xq(int rows, int64_t n, std::mt19937& rng) {
    const int64_t nb = n / 32;
    std::vector<uint8_t> x((size_t) rows * nb * 36);
    for (int r = 0; r < rows; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t* blk = &x[((size_t) r * nb + b) * 36];
            int sum = 0;
            for (int i = 0; i < 32; ++i) {
                const int8_t q = (int8_t) (rng() % 255 - 127);
                blk[4 + i] = (uint8_t) q;
                sum += q;
            }
            const float d = std::ldexp(1.0f + (float) (rng() % 1024) / 1024.0f, -7);
            const sycl::half hd(d), hs(d * (float) sum);
            std::memcpy(blk, &hd, 2);
            std::memcpy(blk + 2, &hs, 2);
        }
    return x;
}

struct Pair { int gu, up, dn; };

std::vector<int> list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::stoi(t));
    return v;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    // the model's pairs (gate/up = up type) and one mixed layer
    std::vector<Pair> pairs{{16, 16, 20}, {17, 17, 20}, {18, 18, 20}, {21, 21, 20}, {22, 22, 20}, {16, 16, 42},
                            {17, 17, 42}, {18, 18, 42}, {21, 21, 42}, {22, 22, 42}, {18, 21, 20}};
    std::vector<int> rows, time_rows;
    for (int t = 1; t <= k::kVerifyMaxT; ++t) rows.push_back(t);
    std::vector<std::string> bodies;
    int64_t N = 2560, FF = 640;
    int sets = 1, reps = 25, warm = 20;
    bool gate_up = false;
    uint32_t seed = 353;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--pairs") {   // gu/dn or gu:up/dn
            pairs.clear();
            std::stringstream ss(next());
            std::string t;
            while (std::getline(ss, t, ',')) {
                const size_t sl = t.find('/'), co = t.find(':');
                const int gu = std::stoi(t.substr(0, co != std::string::npos ? co : sl));
                const int up = co != std::string::npos ? std::stoi(t.substr(co + 1, sl - co - 1)) : gu;
                pairs.push_back({gu, up, std::stoi(t.substr(sl + 1))});
            }
        } else if (a == "--rows") rows = list(next());
        else if (a == "--dims") {
            const auto d = list(next());
            if (d.size() != 2) throw std::runtime_error("--dims N,FF");
            N = d[0];
            FF = d[1];
        } else if (a == "--sets") sets = std::stoi(next());
        else if (a == "--bodies") {
            std::stringstream ss(next());
            std::string t;
            while (std::getline(ss, t, ',')) bodies.push_back(t);
        } else if (a == "--seed") seed = (uint32_t) std::stoul(next());
        else if (a == "--gate-up") gate_up = true;
        else if (a == "--time") time_rows = list(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--warm") warm = std::stoi(next());
        else throw std::runtime_error("unknown argument " + a);
    }
    const int64_t NE = 512, K = 10;
    const int MT = std::max(*std::max_element(rows.begin(), rows.end()),
                            time_rows.empty() ? 1 : *std::max_element(time_rows.begin(), time_rows.end()));
    if (sets <= 0 || reps <= 0 || warm < 0 || pairs.empty() || rows.empty() ||
        std::any_of(rows.begin(), rows.end(), [](int n) { return n <= 0; }))
        throw std::runtime_error("need positive sets/reps/rows and pairs");
    if (MT > k::kVerifyMaxT || k::verify_max_t() < MT)
        throw std::runtime_error("rows past STRATA_BATCH_MAX_T: set STRATA_BATCH_MAX_T=32");
    if (gate_up) {
        const char* m = std::getenv("STRATA_EXP_MODE");
        if (m == nullptr || std::strcmp(m, "0") != 0)
            throw std::runtime_error("--gate-up compares the control's FP32 gate/up rows: run with STRATA_EXP_MODE=0");
    }
    auto& rt = strata::core::Runtime::get();
    // the bodies: the emulation everywhere, XMX where the device has the tiles (a layout's taken check with it)
    const k::NativeExpertLayout probe = k::native_expert_layout(18, 20, N, FF);
    k::native_expert_fast_override(k::kNativeExpertFastXmx);
    k::native_expert_xmx_body(k::NativeExpertXmxBody::automatic);
    const bool has_xmx = k::native_expert_xmx_taken(probe, K, CS());
    k::native_expert_fast_override(-1);
    if (bodies.empty()) {
        bodies.push_back("emulated");
        if (has_xmx) bodies.push_back("xmx");
    }
    std::printf("device: %s, %u compute units; XMX tiles: %s; dims %lld x %lld, %zu pairs, %zu row counts (max %d), "
                "%d sets, bodies:", rt.device().get_info<sycl::info::device::name>().c_str(),
                rt.device().get_info<sycl::info::device::max_compute_units>(), has_xmx ? "yes" : "no",
                (long long) N, (long long) FF, pairs.size(), rows.size(), MT, sets);
    for (const auto& b : bodies) std::printf(" %s", b.c_str());
    std::printf("%s\n", gate_up ? "; gate/up rows compared (STRATA_EXP_MODE=0)" : "");
    auto body_of = [](const std::string& b) {
        if (b == "emulated") return k::NativeExpertXmxBody::emulated;
        if (b == "xmx") return k::NativeExpertXmxBody::xmx;
        throw std::runtime_error("--bodies: emulated or xmx");
    };

    const int64_t capx = (int64_t) MT * K;
    const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const int64_t plan_i32 = ptr_off + 6 * capx + 16;
    std::mt19937 rng(seed);
    uint8_t* xq = dev<uint8_t>((size_t) MT * (N / 32) * 36);
    {
        const auto hx = make_xq(MT, N, rng);
        Q().memcpy(xq, hx.data(), hx.size()).wait();
    }
    int32_t* ids_sets = dev<int32_t>((size_t) sets * capx);
    int32_t* res_sets = dev<int32_t>((size_t) sets * NE);
    int32_t* plans = dev<int32_t>((size_t) sets * plan_i32);
    uint32_t* skip_scr = dev<uint32_t>(1);
    int32_t* zero4 = dev<int32_t>(4);
    Q().memset(zero4, 0, 4 * sizeof(int32_t)).wait();
    const size_t scr_bytes = k::native_expert_scratch_bytes(capx, FF);
    const size_t f = (size_t) capx * (size_t) FF * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* out0 = dev<float>((size_t) capx * N);
    float* out1 = dev<float>((size_t) capx * N);
    uint8_t* scr0 = dev<uint8_t>(scr_bytes);
    uint8_t* scr1 = dev<uint8_t>(scr_bytes);
    std::vector<float> h0((size_t) capx * N), h1((size_t) capx * N);
    std::vector<uint8_t> s0(scr_bytes), s1(scr_bytes);

    std::vector<double> entries_per(3, 0);
    auto routing = [&](int kind) {
        std::vector<int32_t> hid((size_t) sets * capx), hres((size_t) sets * NE, -1);
        entries_per[kind] = 0;
        for (int st = 0; st < sets; ++st) {
            std::vector<int32_t> all(NE), pool;
            for (int e = 0; e < NE; ++e) all[e] = e;
            std::shuffle(all.begin(), all.end(), rng);
            pool.assign(all.begin(), all.begin() + 24);
            std::vector<int32_t> pick;
            int nslot = 0;
            for (int t = 0; t < MT; ++t) {
                if (kind == 0 || (kind == 1 && t == 0)) {
                    std::shuffle(all.begin(), all.end(), rng);
                    pick.assign(all.begin(), all.begin() + K);
                } else if (kind == 2) {
                    std::shuffle(pool.begin(), pool.end(), rng);
                    pick.assign(pool.begin(), pool.begin() + K);
                }
                for (int j = 0; j < K; ++j) {
                    const int e = pick[j];
                    hid[(size_t) st * capx + t * K + j] = e;
                    if (hres[(size_t) st * NE + e] < 0) hres[(size_t) st * NE + e] = nslot++;
                }
            }
            entries_per[kind] += (double) capx / nslot / sets;
        }
        Q().memcpy(ids_sets, hid.data(), hid.size() * 4);
        Q().memcpy(res_sets, hres.data(), hres.size() * 4).wait();
    };
    const char* kind_name[3] = {"random", "shared", "mixed"};

    long long checks = 0;
    int fails = 0;
    for (const Pair& p : pairs) {
        if (!k::native_expert_supported(p.gu, p.dn, N, FF, p.up)) throw std::runtime_error("pair not supported");
        const k::NativeExpertLayout L = k::native_expert_layout(p.gu, p.dn, N, FF, p.up);
        uint8_t* pool = dev<uint8_t>(L.bytes * (size_t) capx);
        for (int64_t s = 0; s < capx; ++s) {
            const auto b = make_blob(L, rng);
            Q().memcpy(pool + (size_t) s * L.bytes, b.data(), b.size()).wait();
        }
        auto plan_for = [&](int T) {
            for (int st = 0; st < sets; ++st)
                k::resident_plan(ids_sets + (size_t) st * capx, (int) (T * K), (int) K, res_sets + (size_t) st * NE,
                                 (int) NE, pool, nullptr, (long long) L.bytes, plans + (size_t) st * plan_i32, capx,
                                 skip_scr, 1, CS());
        };
        // mask 0 + 4 entries: the control; the XMX bit with a forced body: the candidate
        auto call = [&](int T, int st, bool cand, k::NativeExpertXmxBody body, float* out, uint8_t* scr) {
            const int32_t* pl = plans + (size_t) st * plan_i32;
            const int32_t* p_start = pl + 4;
            const int32_t* p_dst = p_start + capx + 1;
            const int32_t* p_tok = p_dst + capx;
            const auto* p_ptr = reinterpret_cast<const unsigned long long*>(pl + ptr_off);
            const int64_t cap = (int64_t) T * K;
            k::native_expert_fast_override(cand ? k::kNativeExpertFastXmx : 0);
            k::native_expert_entries_override(cand ? -1 : 4);
            k::native_expert_xmx_body(body);
            const bool taken = k::native_expert_xmx_taken(L, cap, CS());
            k::native_expert_grouped(L, p_ptr, p_start, pl, p_dst, p_tok, cap, cap, xq, scr, out, CS(), cap);
            k::native_expert_xmx_body(k::NativeExpertXmxBody::automatic);
            k::native_expert_entries_override(-1);
            k::native_expert_fast_override(-1);
            return taken;
        };
        int pfails = 0;
        long long pchecks = 0;
        for (const std::string& bname : bodies) {
            const auto body = body_of(bname);
            for (int kind = 0; kind < 3; ++kind) {
                routing(kind);
                for (int T : rows) {
                    plan_for(T);
                    for (int st = 0; st < sets; ++st) {
                        const size_t used = (size_t) T * K * N;
                        Q().fill(out0, -7.25f, (size_t) capx * N);
                        Q().fill(out1, -7.25f, (size_t) capx * N);
                        Q().memset(scr0, 0xA5, scr_bytes);
                        Q().memset(scr1, 0xA5, scr_bytes);
                        Q().wait();
                        call(T, st, false, k::NativeExpertXmxBody::automatic, out0, scr0);
                        const bool taken = call(T, st, true, body, out1, scr1);
                        Q().memcpy(h0.data(), out0, h0.size() * 4);
                        Q().memcpy(h1.data(), out1, h1.size() * 4);
                        Q().memcpy(s0.data(), scr0, scr_bytes);
                        Q().memcpy(s1.data(), scr1, scr_bytes).wait();
                        auto fail = [&](const char* what, const std::string& detail) {
                            std::printf("FAIL %s t%d:%d/%d %s %s rows %d set %d%s\n", what, p.gu, p.up, p.dn,
                                        bname.c_str(), kind_name[kind], T, st, detail.c_str());
                            ++pfails;
                        };
                        ++pchecks;   // the candidate took the XMX path
                        if (!taken) fail("NOT-TAKEN", "");
                        ++pchecks;   // every used output row written, and the same bits
                        bool unwritten = false;
                        for (size_t i = 0; i < used && !unwritten; ++i) unwritten = h1[i] == -7.25f && h0[i] != -7.25f;
                        if (unwritten) fail("UNWRITTEN", "");
                        ++pchecks;
                        if (std::memcmp(h0.data(), h1.data(), used * 4) != 0) {
                            size_t first = 0, nd = 0;
                            for (size_t i = 0; i < used; ++i)
                                if (std::memcmp(&h0[i], &h1[i], 4) != 0 && nd++ == 0) first = i;
                            char b[160];
                            std::snprintf(b, sizeof b, ": %zu of %zu differ, first [%zu] %.9g vs %.9g", nd, used, first,
                                          h0[first], h1[first]);
                            fail("OUT", b);
                        }
                        ++pchecks;   // the q8_1 h the down product read
                        const size_t hq_used = (size_t) T * K * (size_t) (FF / 32) * 36;
                        if (std::memcmp(s0.data() + 3 * fa, s1.data() + 3 * fa, hq_used) != 0) fail("H-Q8_1", "");
                        if (gate_up) {
                            ++pchecks;   // the FP32 gate and up rows
                            const size_t gu_used = (size_t) T * K * (size_t) FF * 4;
                            if (std::memcmp(s0.data(), s1.data(), gu_used) != 0 ||
                                std::memcmp(s0.data() + fa, s1.data() + fa, gu_used) != 0)
                                fail("GATE-UP", "");
                        }
                    }
                }
            }
            // a call with no group (the verify window's PCIe call with no share, its 4-group grid): writes nothing
            for (int T : {rows.front(), rows.back()}) {
                plan_for(T);
                const int32_t* pl = plans;
                const int64_t cap = (int64_t) T * K;
                Q().fill(out1, -7.25f, (size_t) capx * N).wait();
                k::native_expert_fast_override(k::kNativeExpertFastXmx);
                k::native_expert_xmx_body(body);
                k::native_expert_grouped(L, reinterpret_cast<const unsigned long long*>(pl + ptr_off), pl + 4, zero4,
                                         pl + 4 + capx + 1, pl + 4 + 2 * capx + 1, cap, cap, xq, scr1, out1, CS(), 4);
                k::native_expert_xmx_body(k::NativeExpertXmxBody::automatic);
                k::native_expert_fast_override(-1);
                Q().memcpy(h1.data(), out1, h1.size() * 4).wait();
                ++pchecks;
                if (std::any_of(h1.begin(), h1.end(), [](float v) { return v != -7.25f; })) {
                    std::printf("FAIL EMPTY-WROTE t%d:%d/%d %s rows %d\n", p.gu, p.up, p.dn, bname.c_str(), T);
                    ++pfails;
                }
            }
        }
        std::printf("parity t%d:%d/%d: %s, %lld checks, %d failures (entries an expert at %d rows: random %.2f, "
                    "shared %.2f, mixed %.2f)\n", p.gu, p.up, p.dn, pfails ? "FAIL" : "ok", pchecks, pfails, MT,
                    entries_per[0], entries_per[1], entries_per[2]);
        std::fflush(stdout);
        checks += pchecks;
        fails += pfails;
        // ---- timing (a GPU's check): random routing, the L2 flushed, control then each body, median of reps
        if (!time_rows.empty()) {
            const size_t flush_n = (size_t) 128 << 18;
            float* flush = dev<float>(flush_n);
            routing(0);
            for (int T : time_rows) {
                plan_for(T);
                Q().wait();
                auto med = [&](bool cand, k::NativeExpertXmxBody body) {
                    std::vector<double> v;
                    for (int r = 0; r < warm + reps; ++r) {
                        Q().fill(flush, (float) (r & 7), flush_n);
                        sycl::event a = sx::submit_profiling_tag(Q());
                        call(T, r % sets, cand, body, out1, scr1);
                        sycl::event b = sx::submit_profiling_tag(Q());
                        b.wait_and_throw();
                        if (r >= warm)
                            v.push_back((double) (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                                  a.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e3);
                    }
                    return median(v);
                };
                const double c = med(false, k::NativeExpertXmxBody::automatic);
                std::printf("time t%d:%d/%d rows %2d: control %8.1f us", p.gu, p.up, p.dn, T, c);
                for (const std::string& bname : bodies) {
                    const double b = med(true, body_of(bname));
                    std::printf(" | %s %8.1f us (x%.3f)", bname.c_str(), b, c / b);
                }
                std::printf("\n");
                std::fflush(stdout);
            }
            release(flush);
        }
        release(pool);
    }
    std::printf("\nparity: %lld checks, %d failures\n", checks, fails);
    return fails > 0 ? 1 : 0;
}
