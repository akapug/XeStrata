// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/expert_grouped_parity.cpp - homelab: the routed experts' grouped product (native_expert_grouped) with its
// faster paths (STRATA_EXPERT_FAST, iq_kernels.hpp) against the control, bitwise, and both timed.
//
//     build/expert_grouped_parity [--pairs 17/20,16/42,...] [--rows 1,2,...,16] [--masks M,...] [--reps 25] [--rounds 3]
//                                 [--sets 8] [--seed 353] [--time 1,2,4,8,16] [--phases] [--parity-only]
//                                 [--fill-mod M] [--warm 50] [--time-kind 0|1|2]
// STRATA_GROUPED_ENTRIES=1|2|4|8 chooses the candidate tile; the reference always takes 4. Use --masks 0 to isolate
// tiling from the other fast paths, and --time-kind 0/1/2 for random/shared/mixed reuse controls.
//
// Synthetic expert blobs of the model's real shapes (2560 x 640, 512 experts, top-10): random bytes with every block's
// FP16 scale set finite, random q8_1 activations, routings random (K distinct of 512 a row, as decode routes),
// `shared` (every row the same K) and `mixed` (rows drawing from a small pool, groups of every size).  Each routing's
// plan comes from the engine's resident_plan.  For every pair and row count 1..16 and every routing, the control
// (mask 0, entries 4) and each candidate fast mask (default kNativeExpertFastAll) write into sentinel buffers; every
// output row is compared with memcmp.
// Then, at --time row counts, both are timed as batch_window_ubench times the experts (the L2 flushed by a fill,
// profiling tags around the call), interleaved in --rounds rounds; min and median of the medians per side, with the
// unique bytes a call must move (blob x distinct experts + activations) and the copy-kernel bandwidth floor.
// --fill-mod M: blob bytes random mod M with no scale fix (batch_window_ubench's fill_bytes is mod 0x3c = 60), to
// tell its synthetic weights' timing from random weights' (real GGUF bytes are near random).
// --phases: each side's launches timed apart (native_expert_set_phase_hook), at the --time row counts.
// Exit status: the number of mismatching (pair, rows, routing) cases (0 = parity).
#include "strata/core/runtime.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
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
    if (p == nullptr) throw std::runtime_error("expert_grouped_parity: device allocation failed");
    return static_cast<T*>(p);
}
void release(void* p) { sycl::free(p, strata::core::Runtime::get().context()); }

// values per block and bytes per block of the types the model's experts use (every block starts with its FP16 d)
int qk_of(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }

// a finite FP16 in [2^-10, 2^-6) with a random sign: the products stay far from overflow
uint16_t half_scale(std::mt19937& rng) { return (uint16_t) (((rng() & 1u) << 15) | ((5u + rng() % 4u) << 10) | (rng() & 0x3ffu)); }

int g_fill_mod = 0;   // --fill-mod: 0 = random bytes with finite scales
std::vector<uint8_t> make_blob(const k::NativeExpertLayout& L, std::mt19937& rng) {
    std::vector<uint8_t> b(L.bytes);
    for (auto& v : b) v = (uint8_t) (g_fill_mod > 0 ? rng() % (uint32_t) g_fill_mod : rng());
    if (g_fill_mod > 0) return b;
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

// q8_1 rows: random int8 values, d a modest FP16, s = d * sum (as the quantizer writes it)
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

struct Pair { int gu, dn; };

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<Pair> pairs{{17, 20}, {16, 42}, {17, 42}, {21, 42}, {22, 42}, {16, 20}, {18, 20}, {18, 42}, {21, 20}, {22, 20}};
    std::vector<int> rows, time_rows{1, 2, 4, 8, 16};
    for (int t = 1; t <= 16; ++t) rows.push_back(t);
    std::vector<int> masks;
    int reps = 25, rounds = 3, sets = 8, flush_mib = 128;
    int warm = 50;   // match the window ubench: its first 25-50 calls can still be first-use noise
    int time_kind = 0;   // --time-kind: 0 random, 1 shared, 2 mixed (the parity routings)
    bool phases = false;
    uint32_t seed = 353;
    auto list = [](const std::string& s) {
        std::vector<int> v;
        std::stringstream ss(s);
        std::string t;
        while (std::getline(ss, t, ',')) v.push_back(std::stoi(t));
        return v;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--pairs") {
            pairs.clear();
            std::stringstream ss(next());
            std::string t;
            while (std::getline(ss, t, ',')) pairs.push_back({std::stoi(t.substr(0, t.find('/'))), std::stoi(t.substr(t.find('/') + 1))});
        } else if (a == "--rows") rows = list(next());
        else if (a == "--time") time_rows = list(next());
        else if (a == "--masks") masks = list(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--warm") warm = std::stoi(next());
        else if (a == "--time-kind") time_kind = std::stoi(next());
        else if (a == "--rounds") rounds = std::stoi(next());
        else if (a == "--sets") sets = std::stoi(next());
        else if (a == "--seed") seed = (uint32_t) std::stoul(next());
        else if (a == "--phases") phases = true;
        else if (a == "--parity-only") time_rows.clear();
        else if (a == "--fill-mod") g_fill_mod = std::stoi(next());
        else throw std::runtime_error("unknown argument " + a);
    }
    if (masks.empty()) masks.push_back(k::kNativeExpertFastAll);
    if (warm < 0 || reps <= 0 || rounds <= 0 || sets <= 0 || flush_mib <= 0 ||
        time_kind < 0 || time_kind > 2 || pairs.empty() || rows.empty() ||
        std::any_of(rows.begin(), rows.end(), [](int n) { return n <= 0; }) ||
        std::any_of(time_rows.begin(), time_rows.end(), [](int n) { return n <= 0; }))
        throw std::runtime_error("need nonnegative --warm, positive reps/rounds/sets/flush, time-kind 0..2, pairs and positive rows");
    const char* entries_env = std::getenv("STRATA_GROUPED_ENTRIES");
    const int candidate_entries = entries_env != nullptr ? std::stoi(entries_env) : 4;
    k::native_expert_entries_override(candidate_entries);   // reject unsupported candidate tiles before allocating
    k::native_expert_entries_override(-1);
    std::printf("entries: control 4, candidate %d; warm %d; fill-mod %d (0: finite random scales)\n",
                candidate_entries, warm, g_fill_mod);
    const int64_t N = 2560, FF = 640, NE = 512, K = 10;
    const int MT = std::max(*std::max_element(rows.begin(), rows.end()),
                            time_rows.empty() ? 1 : *std::max_element(time_rows.begin(), time_rows.end()));
    if (MT > k::kVerifyMaxT || k::verify_max_t() < MT) throw std::runtime_error("rows past STRATA_BATCH_MAX_T: set it to 16");
    const int64_t capx = (int64_t) MT * K;
    const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const int64_t plan_i32 = ptr_off + 6 * capx + 16;
    auto& rt = strata::core::Runtime::get();
    std::printf("device: %s, %u compute units, max %u MHz; %zu fast masks (control 0); pairs %zu, rows %zu, sets %d\n",
                rt.device().get_info<sycl::info::device::name>().c_str(),
                rt.device().get_info<sycl::info::device::max_compute_units>(),
                rt.device().get_info<sycl::info::device::max_clock_frequency>(), masks.size(), pairs.size(), rows.size(), sets);

    std::mt19937 rng(seed);
    const size_t flush_n = (size_t) flush_mib << 18;
    float* flush = dev<float>(flush_n);
    // the copy kernel's bandwidth (read + write), as batch_window_ubench's floor
    double bw = 0, bw_read = 0;
    if (!time_rows.empty()) {   // --parity-only: no timing (a device without profiling tags, the CPU)
        float* y = dev<float>(flush_n);
        // random values: the B70 compresses device memory, and a constant buffer copies far above DRAM speed
        Q().parallel_for(sycl::range<1>(flush_n), [=](sycl::id<1> i) {
            uint32_t v = (uint32_t) i[0] * 2654435761u;
            v ^= v >> 15; v *= 0x2c1b3c6du; v ^= v >> 12;
            flush[i] = (float) (v & 0xffffff);
        }).wait();
        auto* x4 = reinterpret_cast<sycl::float4*>(flush);
        auto* y4 = reinterpret_cast<sycl::float4*>(y);
        for (int r = 0; r < 20; ++r) {
            sycl::event a = sx::submit_profiling_tag(Q());
            Q().parallel_for(sycl::range<1>(flush_n / 4), [=](sycl::id<1> i) { y4[i] = x4[i]; });
            sycl::event b = sx::submit_profiling_tag(Q());
            b.wait_and_throw();
            const double us = (double) (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                        a.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e3;
            bw = std::max(bw, 2.0 * flush_n * 4 / us / 1e3);
        }
        for (int r = 0; r < 20; ++r) {   // read-only: what the experts' weight stream can reach
            sycl::event a = sx::submit_profiling_tag(Q());
            Q().parallel_for(sycl::range<1>(flush_n / 4), [=](sycl::id<1> i) {
                const sycl::float4 v = x4[i];
                if (v.x() == -1.0f) y4[i] = v;
            });
            sycl::event b = sx::submit_profiling_tag(Q());
            b.wait_and_throw();
            const double us = (double) (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                        a.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e3;
            bw_read = std::max(bw_read, 1.0 * flush_n * 4 / us / 1e3);
        }
        release(y);
    }
    if (!time_rows.empty())
        std::printf("copy kernel: %.1f GB/s (read + write; the floor below); read-only %.1f GB/s\n", bw, bw_read);

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
    float* out0 = dev<float>((size_t) capx * N);
    float* out1 = dev<float>((size_t) capx * N);
    std::vector<float> h0((size_t) capx * N), h1((size_t) capx * N);

    // routings: kind 0 random (K distinct of NE a row), 1 shared (every row the same K), 2 mixed (rows draw K
    // distinct from a pool of 24: groups of 1..16 entries); each set its own slots in the pool of capx blobs
    std::vector<double> distinct(3, 0);
    auto routing = [&](int kind) {
        std::vector<int32_t> hid((size_t) sets * capx), hres((size_t) sets * NE, -1);
        distinct[kind] = 0;
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
            distinct[kind] += (double) nslot / sets;
        }
        Q().memcpy(ids_sets, hid.data(), hid.size() * 4);
        Q().memcpy(res_sets, hres.data(), hres.size() * 4).wait();
    };
    const char* kind_name[3] = {"random", "shared", "mixed"};

    int fails = 0;
    long long checked = 0;
    struct Side { int mask, entries; double min, med; };
    struct Row { std::vector<Side> side; double bytes = 0; };
    std::map<std::pair<int, int>, std::map<int, Row>> table;
    for (const Pair& p : pairs) {
        if (!k::native_expert_supported(p.gu, p.dn, N, FF)) throw std::runtime_error("pair not supported");
        const k::NativeExpertLayout L = k::native_expert_layout(p.gu, p.dn, N, FF);
        uint8_t* pool = dev<uint8_t>(L.bytes * (size_t) capx);
        for (int64_t s = 0; s < capx; ++s) {
            const auto b = make_blob(L, rng);
            Q().memcpy(pool + (size_t) s * L.bytes, b.data(), b.size()).wait();   // b must live until the copy completes
        }
        Q().wait();
        void* scr = dev<uint8_t>(k::native_expert_scratch_bytes(capx, FF));
        auto plan_for = [&](int T) {
            for (int st = 0; st < sets; ++st)
                k::resident_plan(ids_sets + (size_t) st * capx, (int) (T * K), (int) K, res_sets + (size_t) st * NE, (int) NE,
                                 pool, nullptr, (long long) L.bytes, plans + (size_t) st * plan_i32, capx, skip_scr, 1, CS());
        };
        auto call = [&](int T, int st, int m, float* out, int entries = -1) {
            const int32_t* pl = plans + (size_t) st * plan_i32;
            const int32_t* p_start = pl + 4;
            const int32_t* p_dst = p_start + capx + 1;
            const int32_t* p_tok = p_dst + capx;
            const auto* p_ptr = reinterpret_cast<const unsigned long long*>(pl + ptr_off);
            const int64_t cap = (int64_t) T * K;
            k::native_expert_fast_override(m);
            k::native_expert_entries_override(entries);
            k::native_expert_grouped(L, p_ptr, p_start, pl, p_dst, p_tok, cap, cap, xq, scr, out, CS(), cap);
            k::native_expert_entries_override(-1);
            k::native_expert_fast_override(-1);
        };
        // ---- parity
        int pfails = 0;
        for (int kind = 0; kind < 3; ++kind) {
            routing(kind);
            for (int T : rows) {
                plan_for(T);
                for (int st = 0; st < std::min(sets, 3); ++st) {
                    Q().fill(out0, -7.25f, (size_t) capx * N);
                    call(T, st, 0, out0, 4);
                    Q().memcpy(h0.data(), out0, h0.size() * 4).wait();
                    const size_t used = (size_t) T * K * N;
                    for (int m : masks) {
                        Q().fill(out1, -7.25f, (size_t) capx * N);
                        call(T, st, m, out1);
                        Q().memcpy(h1.data(), out1, h1.size() * 4).wait();
                        ++checked;
                        if (std::memcmp(h0.data(), h1.data(), used * 4) != 0) {
                            size_t first = 0, nd = 0;
                            for (size_t i = 0; i < used; ++i)
                                if (std::memcmp(&h0[i], &h1[i], 4) != 0) { if (nd++ == 0) first = i; }
                            std::printf("MISMATCH mask %d t%d/%d %s rows %d set %d: %zu of %zu values differ, first [%zu] %.9g vs %.9g\n",
                                        m, p.gu, p.dn, kind_name[kind], T, st, nd, used, first, h0[first], h1[first]);
                            ++pfails;
                        }
                        bool sentinel = false;   // every used row written (no -7.25 left)
                        for (size_t i = 0; i < used && !sentinel; ++i) sentinel = h1[i] == -7.25f && h0[i] != -7.25f;
                        if (sentinel) {
                            std::printf("UNWRITTEN mask %d t%d/%d %s rows %d set %d\n", m, p.gu, p.dn, kind_name[kind], T, st);
                            ++pfails;
                        }
                    }
                }
            }
        }
        // the verify window's PCIe call with no group (its 4-group grid, the plan's count 0): it writes nothing, so a
        // window that can never have a PCIe share may leave it out (STRATA_SKIP_NO_PCIE_SHARE)
        for (int T : rows) {
            constexpr int64_t kPcieGroupRows = 4;   // Verifier::kPcieGroupRows
            const int32_t* pl = plans;
            const int64_t cap = (int64_t) T * K;
            for (int m : {0, masks.front()}) {
                Q().fill(out1, -7.25f, (size_t) capx * N);
                k::native_expert_fast_override(m);
                k::native_expert_grouped(L, reinterpret_cast<const unsigned long long*>(pl + ptr_off), pl + 4, zero4,
                                         pl + 4 + capx + 1, pl + 4 + 2 * capx + 1, cap, cap, xq, scr, out1, CS(),
                                         kPcieGroupRows);
                k::native_expert_fast_override(-1);
                Q().memcpy(h1.data(), out1, h1.size() * 4).wait();
                ++checked;
                const bool touched = std::any_of(h1.begin(), h1.end(), [](float v) { return v != -7.25f; });
                if (touched) {
                    std::printf("EMPTY-PCIE-WROTE mask %d t%d/%d rows %d\n", m, p.gu, p.dn, T);
                    ++pfails;
                }
            }
        }
        fails += pfails;
        std::printf("parity t%d/%d: %s (%zu row counts x 3 routings x %d sets x %zu masks; empty PCIe call)\n", p.gu,
                    p.dn, pfails ? "FAIL" : "ok", rows.size(), std::min(sets, 3), masks.size());
        std::fflush(stdout);
        // ---- timing: a selected parity routing, interleaved rounds
        if (!time_rows.empty()) {
            routing(time_kind);
            for (int T : time_rows) {
                plan_for(T);
                Q().wait();
                auto once = [&](int m, int entries, int rep) {
                    Q().fill(flush, (float) (rep & 7), flush_n);
                    sycl::event a = sx::submit_profiling_tag(Q());
                    call(T, rep % sets, m, out1, entries);
                    sycl::event b = sx::submit_profiling_tag(Q());
                    b.wait_and_throw();
                    return (double) (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                     a.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e3;
                };
                auto med = [&](int m, int entries) {
                    for (int w = 0; w < warm; ++w) once(m, entries, w);
                    std::vector<double> v;
                    for (int r = 0; r < reps; ++r) v.push_back(once(m, entries, warm + r));
                    return median(v);
                };
                std::vector<std::pair<int, int>> all{{0, 4}};
                for (int m : masks) all.emplace_back(m, candidate_entries);
                std::vector<std::vector<double>> v(all.size());
                for (int r = 0; r < rounds; ++r)
                    for (size_t m = 0; m < all.size(); ++m) v[m].push_back(med(all[m].first, all[m].second));
                // distinct experts at T rows: measured from the plan
                int32_t ng = 0;
                Q().memcpy(&ng, plans, 4).wait();
                Row& row = table[{p.gu, p.dn}][T];
                row.bytes = (double) L.bytes * ng + (double) T * K * (N * 4.0 + 64);
                for (size_t m = 0; m < all.size(); ++m)
                    row.side.push_back({all[m].first, all[m].second, *std::min_element(v[m].begin(), v[m].end()), median(v[m])});
            }
            if (phases) {
                for (int T : time_rows) {
                    plan_for(T);
                    std::vector<std::pair<int, int>> all{{0, 4}};
                    for (int m : masks) all.emplace_back(m, candidate_entries);
                    for (const auto& [m, entries] : all) {
                        std::map<int, std::vector<double>> ph;
                        static std::vector<std::pair<int, sycl::event>>* tags = nullptr;
                        std::vector<std::pair<int, sycl::event>> local;
                        tags = &local;
                        k::native_expert_set_phase_hook([](void* s, int phase) {
                            tags->push_back({phase, sx::submit_profiling_tag(*static_cast<sycl::queue*>(s))});
                        });
                        for (int r = 0; r < reps + warm; ++r) {
                            local.clear();
                            Q().fill(flush, (float) (r & 7), flush_n);
                            sycl::event a = sx::submit_profiling_tag(Q());
                            call(T, r % sets, m, out1, entries);
                            sycl::event b = sx::submit_profiling_tag(Q());
                            b.wait_and_throw();
                            auto t_of = [](const sycl::event& e) {
                                return (double) e.get_profiling_info<sycl::info::event_profiling::command_end>() / 1e3;
                            };
                            double prev = t_of(a);
                            if (r < warm) continue;
                            for (auto& [phase, e] : local) {
                                ph[phase].push_back(t_of(e) - prev);
                                prev = t_of(e);
                            }
                            ph[99].push_back(t_of(b) - prev);
                        }
                        k::native_expert_set_phase_hook(nullptr);
                        std::printf("phases t%d/%d rows %d mask %d entries %d:", p.gu, p.dn, T, m, entries);
                        for (auto& [phase, v] : ph) std::printf("  [%d] %.1f", phase, median(v));
                        std::printf(" us\n");
                    }
                }
            }
        }
        release(pool);
        release(scr);
    }
    if (!table.empty()) {
        std::printf("\nTIMING (%s routing; us per call, min and median over %d rounds of %d-rep medians; GB/s and x floor "
                    "from the min; floor = unique bytes / %.0f GB/s, the copy kernel; read-only %.0f GB/s)\n",
                    kind_name[time_kind], rounds, reps, bw, bw_read);
        std::printf("%-8s %4s %8s %5s %7s | %8s %8s %7s %6s %6s\n", "pair", "rows", "MB", "mask", "entries", "min", "median", "GB/s", "xfl",
                    "speed");
        for (auto& [pr, m] : table)
            for (auto& [T, r] : m) {
                const double fl = r.bytes / (bw * 1e3);
                for (const Side& sd : r.side)
                    std::printf("t%d/%-5d %4d %8.2f %5d %7d | %8.1f %8.1f %7.1f %6.2f %6.3f\n", pr.first, pr.second, T,
                                r.bytes / 1e6, sd.mask, sd.entries, sd.min, sd.med, r.bytes / sd.min / 1e3, sd.min / fl,
                                r.side[0].min / sd.min);
            }
    }
    std::printf("\nparity: %lld cases checked, %d failures\n", checked, fails);
    return fails;
}
