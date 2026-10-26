// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/gdn_slots_parity.cpp - STRATA_BATCH_GDN_FORM: every form of gdn_step_norm_slots against the per-slot
// loop of gdn_step_norm_multi (the engine's path without STRATA_BATCH_GDN_SLOTS, each call in the form gdn_step_tune
// picks, or STRATA_GDN_STEP's), bitwise (memcmp of the outputs, their q8_1 image and the committed states), and the
// time of each, at the model's shape (state 128, 16 key heads, 48 value heads) for 1..16 slots of one row each.
// Each time is a round's mean over `reps` launches, the forms' rounds taken in turns; the fastest and the median
// round are printed (the card may be serving).
//
//     build/gdn_slots_parity [reps] [rounds]
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {
namespace P = strata::parity;
namespace k = strata::kernels;
template <typename T> T* up(const std::vector<T>& h) {
    T* p = static_cast<T*>(P::alloc_bytes(h.size() * sizeof(T) + 64));
    P::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    P::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}
const char* const kFormName[] = {"plain", "split", "large", "block", "blk+lg", "blk16"};
}  // namespace

int main(int argc, char** argv) try {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20, rounds = argc > 2 ? std::atoi(argv[2]) : 7;
    constexpr int S = 128, HK = 16, HV = 48, C = 2 * S * HK + S * HV, VD = S * HV, R = k::kVerifyMaxT;
    const size_t n_state = (size_t) S * S * HV;
    const int forms = k::gdn_slots_forms();
    void* s = P::stream();
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<float> st0((size_t) R * n_state), h((size_t) R * C), gate((size_t) R * HV), beta(gate.size()),
        z((size_t) R * VD), gamma(S);
    for (auto& v : st0) v = 0.05f * nd(rng);
    for (auto& v : h) v = 0.09f * nd(rng);   // about the l2-normed rows' scale
    for (auto& v : gate) v = -0.5f * ud(rng);
    for (auto& v : beta) v = ud(rng);
    for (auto& v : z) v = nd(rng);
    for (auto& v : gamma) v = 1.f + 0.1f * nd(rng);
    float *dh = up(h), *dg = up(gate), *db = up(beta), *dz = up(z), *dgm = up(gamma);
    int32_t* keep = up(std::vector<int32_t>{1});
    float* dst = static_cast<float*>(P::alloc_bytes(st0.size() * 4));
    float* dy = static_cast<float*>(P::alloc_bytes((size_t) R * VD * 4));
    uint8_t* dq = static_cast<uint8_t*>(P::alloc_bytes((size_t) R * VD / 32 * 36));
    k::gdn_step_tune(C, HK, HV, s);   // the per-slot calls' form, as the engine picks it
    // the large register file forms (2 large, 4 block+large) build for the GPU only: the OpenCL CPU device refuses
    // -ze-opt-large-register-file (pad_rows_parity skips them alike)
    const bool cpu = std::getenv("STRATA_TEST_CPU_DEVICE") != nullptr;
    const auto skipped = [&](int f) { return cpu && (f == 2 || f == 4); };
    for (int f = 0; f < forms; ++f)
        if (skipped(f)) std::printf("form %s: skipped on the CPU device (large register file)\n", kFormName[f]);
    int failures = 0, checks = 0;
    for (int n = 1; n <= R; ++n) {
        k::GdnSlotTable tab;
        tab.n = n;
        for (int j = 0; j < n; ++j) { tab.state[j] = dst + (size_t) j * n_state; tab.first[j] = j; tab.ntok[j] = 1; }
        // run -1: the per-slot loop; run f >= 0: gdn_step_norm_slots in form f
        const auto run = [&](int f, bool commit) {
            if (f < 0) {
                for (int j = 0; j < n; ++j)
                    k::gdn_step_norm_multi(tab.state[j], dh + (size_t) j * C, C, dg + (size_t) j * HV, db + (size_t) j * HV,
                                           dz + (size_t) j * VD, dgm, 1e-6f, dy + (size_t) j * VD, HK, HV, 1,
                                           commit ? keep : nullptr, s, 0, dq + (size_t) j * VD / 32 * 36);
                return;
            }
            k::gdn_slots_set_form(f);
            k::gdn_step_norm_slots(tab, dh, C, dg, db, dz, dgm, 1e-6f, dy, HK, HV, commit, s, 0, dq);
        };
        for (bool commit : {false, true}) {
            std::vector<float> y0, st_0;
            std::vector<uint8_t> yq0;
            for (int f = -1; f < forms; ++f) {
                if (skipped(f)) continue;
                P::copy_bytes_in(dst, st0.data(), st0.size() * 4);
                run(f, commit);
                P::sync();
                const auto y = down(dy, (size_t) n * VD);
                const auto yq = down(dq, (size_t) n * VD / 32 * 36);
                const auto st = down(dst, (size_t) n * n_state);
                if (f < 0) { y0 = y; yq0 = yq; st_0 = st; continue; }
                ++checks;
                const bool ok = !std::memcmp(y0.data(), y.data(), y.size() * 4) && !std::memcmp(yq0.data(), yq.data(), yq.size()) &&
                                !std::memcmp(st_0.data(), st.data(), st.size() * 4);
                if (!ok) { ++failures; std::printf("FAIL slots %d commit %d form %s\n", n, (int) commit, kFormName[f]); }
            }
        }
        if (n == 1 || n == 4 || n == 8 || n == 16) {
            std::vector<std::vector<double>> us((size_t) forms + 1);
            for (int round = 0; round <= rounds; ++round)
                for (int f = -1; f < forms; ++f) {
                    if (skipped(f)) continue;
                    run(f, true);
                    P::sync();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) run(f, true);
                    P::sync();
                    const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                    if (round > 0) us[(size_t) f + 1].push_back(t);   // round 0: the warm-up
                }
            std::printf("slots %2d (us min/median, GB/s of state at the min):\n", n);
            for (int f = -1; f < forms; ++f) {
                auto& v = us[(size_t) f + 1];
                if (v.empty()) continue;
                std::sort(v.begin(), v.end());
                std::printf("  %-8s %7.1f %7.1f  %4.0f\n", f < 0 ? "per-slot" : kFormName[f], v[0], v[v.size() / 2],
                            2.0 * n * n_state * 4 / v[0] / 1e3);
            }
        }
    }
    std::printf("gdn_slots_parity: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "gdn_slots_parity: %s\n", e.what());
    return 1;
}
