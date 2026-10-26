// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Synthetic prefill MoE combine + HC write/norm against the existing two launches, exact bytes.
#include "strata/prefill/kernels.hpp"
#include "../kernels/parity_device.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace P = strata::parity;
namespace pf = strata::prefill;

namespace {
constexpr int N = 2560, HC = 4, K = 10;
constexpr float EPS = 1e-6f;
constexpr std::array<int, 4> kRows{1, 3, 16, 65};

// Separate allocations are required: the caller's shared HC region is deliberately ineligible.
template<class T> bool equal(const T* a, const T* b, size_t n) {
    std::vector<T> x(n), y(n);
    P::copy_out(x.data(), a, n);
    P::copy_out(y.data(), b, n);
    return std::memcmp(x.data(), y.data(), n * sizeof(T)) == 0;
}
}

int main() {
    std::mt19937 rng(353);  // NOLINT(bugprone-random-generator-seed): reproducible synthetic rows.
    std::uniform_real_distribution<float> value(-2.0f, 2.0f);
    int cases = 0, failed = 0;
    for (int T : {0, -1}) {
        pf::moe_combine_gr_write_norm_rs(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        HC, nullptr, EPS, nullptr, nullptr, T, nullptr);
        ++cases;
        std::printf("combine_hc T=%d: empty launch returned without a queue\n", T);
    }
    for (int T : kRows) for (int extra : {0, 3}) for (bool residual : {false, true}) {
        const int stride = HC + extra;
        const size_t tn = (size_t) T * N, rows = tn * HC;
        std::vector<float> Dm(tn * K), weights((size_t) T * K), shared(tn), sg((size_t) T),
                           R(rows), inj((size_t) T * stride), norm(N * HC);
        std::vector<int32_t> slots((size_t) T * K);
        for (auto* v : {&Dm, &weights, &shared, &sg, &R, &inj, &norm})
            for (auto& x : *v) x = value(rng);
        for (size_t i = 0; i < slots.size(); ++i) {
            slots[i] = (int32_t) ((i * 17 + 1) % slots.size());
            weights[i] = i % 7 ? weights[i] * 0.1f : 0.0f;
        }
        for (int t = 0; t < T; ++t) {
            sg[(size_t) t] = t % 3 ? sg[(size_t) t] : t % 2 ? -80.0f : 80.0f;
            for (int c = 0; c < HC; ++c) {
                inj[(size_t) t * stride + c] *= (float) (1 << (c * 2));
                for (int n = 0; n < N; ++n) R[((size_t) t * HC + c) * N + n] *= (float) (1 << c);
            }
        }
        P::Device d;
        float* dm = d.upload(Dm);
        int32_t* slot = d.upload(slots);
        float* w = d.upload(weights);
        float* sh = d.upload(shared);
        float* gate = d.upload(sg);
        float* inject = d.upload(inj);
        float* wn = d.upload(norm);
        float* a = d.upload(R);
        float* b = d.upload(R);
        float* bo = d.alloc<float>(tn);
        float* rsa = d.alloc<float>((size_t) T * HC);
        float* rsb = d.alloc<float>((size_t) T * HC);
        uint16_t* ha = d.alloc<uint16_t>(rows);
        uint16_t* hb = d.alloc<uint16_t>(rows);
        uint16_t* la = residual ? d.alloc<uint16_t>(rows) : nullptr;
        uint16_t* lb = residual ? d.alloc<uint16_t>(rows) : nullptr;
        pf::moe_combine(dm, slot, w, sh, gate, bo, T, P::stream());
        pf::gr_write_norm_rs(a, bo, inject, stride, wn, EPS, rsa, ha, T, P::stream(), la);
        pf::moe_combine_gr_write_norm_rs(dm, slot, w, sh, gate, b, inject, stride, wn, EPS,
                                        rsb, hb, T, P::stream(), lb);
        P::sync();
        const bool r = equal(a, b, rows), rs = equal(rsa, rsb, (size_t) T * HC),
                   hi = equal(ha, hb, rows), lo = !residual || equal(la, lb, rows);
        ++cases;
        failed += !(r && rs && hi && lo);
        std::printf("combine_hc T=%d stride=%d residual=%d: R=%d rs=%d BF16=%d lo=%d\n",
                    T, stride, (int) residual, (int) r, (int) rs, (int) hi, (int) lo);
    }
    std::printf("combine_hc_parity: %d cases, %d failures\n", cases, failed);
    return failed ? 1 : 0;
}
