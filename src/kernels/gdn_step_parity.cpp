// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/gdn_step_parity.cpp - STRATA_GDN_BLOCK_GRF: the per-slot recurrence (gdn_step_norm_multi in each of its
// three forms) and the state commit (gdn_step_norm_multi's commit, gdn_commit_layers) with block loads/stores and the
// large register file, against today's kernels (the plain form, the setting off), bitwise (memcmp of the outputs, their
// q8_1 image and the states) for windows of 1..16 tokens in every way the engine calls them; then the time of each at
// the model's shape (state 128, 16 key heads, 48 value heads), one token for each of 1, 4, 8 and 16 slots, the slots'
// calls one after another as the per-slot loop runs them.  Each time is the fastest and the median of `rounds` rounds of
// `reps` loops, the variants' rounds in turns (the card may be serving).
//
//     build/gdn_step_parity [reps] [rounds]
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
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
const char* const kFormName[] = {"plain", "split", "prefetch"};
constexpr int kForms = 3;
}  // namespace

int main(int argc, char** argv) try {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20, rounds = argc > 2 ? std::atoi(argv[2]) : 7;
    constexpr int S = 128, HK = 16, HV = 48, C = 2 * S * HK + S * HV, VD = S * HV, R = k::kVerifyMaxT;
    constexpr int LAYERS = 3;   // gdn_commit_layers' check
    const size_t n_state = (size_t) S * S * HV, layer = n_state + (size_t) 3 * C;   // a layer: its state, its conv history
    void* s = P::stream();
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    std::vector<float> st0((size_t) R * layer), h((size_t) R * C * LAYERS), gate((size_t) R * HV * LAYERS),
        beta(gate.size()), z((size_t) R * VD), gamma(S), y0((size_t) R * VD), qkv((size_t) R * C * LAYERS);
    for (auto& v : st0) v = 0.05f * nd(rng);
    for (auto& v : h) v = 0.09f * nd(rng);   // about the l2-normed rows' scale
    for (auto& v : gate) v = -0.5f * ud(rng);
    for (auto& v : beta) v = ud(rng);
    for (auto& v : z) v = nd(rng);
    for (auto& v : gamma) v = 1.f + 0.1f * nd(rng);
    for (auto& v : y0) v = nd(rng);   // the rows a call does not write keep these
    for (auto& v : qkv) v = nd(rng);
    float *dh = up(h), *dg = up(gate), *db = up(beta), *dz = up(z), *dgm = up(gamma), *dqkv = up(qkv);
    int32_t* keep = up(std::vector<int32_t>{1});
    float* dst = static_cast<float*>(P::alloc_bytes(st0.size() * 4));
    float* dy = static_cast<float*>(P::alloc_bytes(y0.size() * 4));
    const size_t nq = (size_t) R * VD / 32 * 36;
    uint8_t* dq = static_cast<uint8_t*>(P::alloc_bytes(nq));
    const std::vector<uint8_t> q0(nq, 0x5a);
    int failures = 0, checks = 0;
    struct Out { std::vector<float> y, st; std::vector<uint8_t> yq; };
    const auto reset = [&] {
        P::copy_bytes_in(dst, st0.data(), st0.size() * 4);
        P::copy_bytes_in(dy, y0.data(), y0.size() * 4);
        P::copy_bytes_in(dq, q0.data(), nq);
    };
    const auto read = [&] { P::sync(); return Out{down(dy, y0.size()), down(dst, st0.size()), down(dq, nq)}; };
    const auto same = [](const Out& a, const Out& b) {
        return !std::memcmp(a.y.data(), b.y.data(), a.y.size() * 4) && !std::memcmp(a.st.data(), b.st.data(), a.st.size() * 4) &&
               !std::memcmp(a.yq.data(), b.yq.data(), a.yq.size());
    };
    // the engine's calls of a T-token window: (n_keep, t_out_begin, q8_1 image) - verify, verify without the image,
    // a commit that keeps fewer rows than it runs, a replay (outputs from the middle), the state-only commit
    for (int T = 1; T <= R; ++T) {
        const int kept = std::max(1, T - 1);
        const struct { int keep, tob; bool q8; const char* what; } calls[] = {
            {0, 0, true, "verify"}, {0, 0, false, "verify-noq8"}, {kept, 0, true, "commit"},
            {T, T / 2, true, "replay"}, {kept, T, false, "state-commit"}};
        for (const auto& c : calls) {
            if (c.keep) P::copy_bytes_in(keep, &c.keep, 4);
            const auto run = [&](int form, bool block_grf) {
                k::gdn_step_set_form(form);
                k::gdn_step_set_block_grf(block_grf);
                reset();
                k::gdn_step_norm_multi(dst, dh, C, dg, db, dz, dgm, 1e-6f, dy, HK, HV, T, c.keep ? keep : nullptr, s,
                                       c.tob, c.q8 ? dq : nullptr);
                return read();
            };
            const Out ref = run(0, false);
            for (int form = 0; form < kForms; ++form)
                for (bool block_grf : {false, true}) {
                    if (form == 0 && !block_grf) continue;
                    ++checks;
                    if (!same(ref, run(form, block_grf))) {
                        ++failures;
                        std::printf("FAIL T %d %s form %s block_grf %d\n", T, c.what, kFormName[form], (int) block_grf);
                    }
                }
        }
        // gdn_commit_layers: LAYERS layers' conv histories and states in one launch each
        P::copy_bytes_in(keep, &kept, 4);
        Out lay[2];
        for (bool block_grf : {false, true}) {
            k::gdn_step_set_block_grf(block_grf);
            reset();
            if (!k::gdn_commit_layers(dst, (int64_t) layer, (int64_t) n_state, dqkv, (int64_t) R * C, dh, (int64_t) R * C,
                                      dg, db, (int64_t) R * HV, C, HK, HV, T, LAYERS, keep, s))
                throw std::runtime_error("gdn_commit_layers took the per-layer loop");
            lay[block_grf] = read();
        }
        ++checks;
        if (!same(lay[0], lay[1])) { ++failures; std::printf("FAIL T %d commit_layers\n", T); }
    }
    // the times: one token a slot, the slots' calls in turn (the per-slot loop's commit), and the state-only commit
    const int one = 1;
    P::copy_bytes_in(keep, &one, 4);
    for (int n : {1, 4, 8, 16}) {
        struct Variant { std::string name; int form; bool block_grf, state_only; };
        std::vector<Variant> vs;
        for (int form = 0; form < kForms; ++form)
            for (bool b : {false, true}) vs.push_back({std::string(kFormName[form]) + (b ? "+blk" : ""), form, b, false});
        for (bool b : {false, true}) vs.push_back({std::string("commit") + (b ? "+blk" : ""), 0, b, true});
        const auto loop = [&](const Variant& v) {
            for (int j = 0; j < n; ++j)
                k::gdn_step_norm_multi(dst + (size_t) j * layer, dh + (size_t) j * C, C, dg + (size_t) j * HV,
                                       db + (size_t) j * HV, dz + (size_t) j * VD, dgm, 1e-6f, dy + (size_t) j * VD, HK, HV,
                                       1, keep, s, v.state_only ? 1 : 0, v.state_only ? nullptr : dq + (size_t) j * VD / 32 * 36);
        };
        std::vector<std::vector<double>> us(vs.size());
        for (int round = 0; round <= rounds; ++round)
            for (size_t i = 0; i < vs.size(); ++i) {
                k::gdn_step_set_form(vs[i].form);
                k::gdn_step_set_block_grf(vs[i].block_grf);
                loop(vs[i]);
                P::sync();
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r) loop(vs[i]);
                P::sync();
                const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                if (round > 0) us[i].push_back(t);   // round 0: the warm-up
            }
        std::printf("slots %2d (us min/median, GB/s of state at the min):\n", n);
        for (size_t i = 0; i < vs.size(); ++i) {
            auto& v = us[i];
            std::sort(v.begin(), v.end());
            std::printf("  %-14s %7.1f %7.1f  %4.0f\n", vs[i].name.c_str(), v[0], v[v.size() / 2], 2.0 * n * n_state * 4 / v[0] / 1e3);
        }
    }
    std::printf("gdn_step_parity: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "gdn_step_parity: %s\n", e.what());
    return 1;
}
