// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Model-free device qualification, not a speed benchmark. Run a separate
// process per chunk shape; the production configuration is cached per device.
#include "strata/prefill/gdn_chunk.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace P = strata::parity;
namespace k = strata::prefill;
constexpr int S = 128, HK = 16, GUARD = 16;
constexpr float SENTINEL = 0.125f;
constexpr uint16_t HALF_SENTINEL = 0x3555;
constexpr double FP32_TOL = 5e-5, FP16_TOL = 2e-3;

void select(const char* chunk) {
#if defined(_WIN32)
    if (_putenv_s("STRATA_GDN_CHUNK", chunk ? chunk : "") != 0) throw std::runtime_error("cannot set chunk config");
#else
    if ((chunk ? setenv("STRATA_GDN_CHUNK", chunk, 1) : unsetenv("STRATA_GDN_CHUNK")) != 0)
        throw std::runtime_error("cannot set chunk config");
#endif
}

struct Result {
    std::vector<float> state, y;
    std::vector<uint16_t> y16;
};

double error(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<double>::infinity();
        e = std::max(e, std::fabs((double) a[i] - b[i]) / (1.0 + std::fabs((double) a[i])));
    }
    return e;
}

double half_error(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    std::vector<float> x(a.size()), y(b.size());
    std::transform(a.begin(), a.end(), x.begin(), strata::kernels::f32_from_f16);
    std::transform(b.begin(), b.end(), y.begin(), strata::kernels::f32_from_f16);
    return error(x, y);
}

bool guards(const Result& r, size_t state, size_t y) {
    for (int i = 0; i < GUARD; ++i) {
        const size_t j = (size_t) i;
        if (r.state[j] != SENTINEL || r.state[GUARD + state + j] != SENTINEL ||
            r.y[j] != SENTINEL || r.y[GUARD + y + j] != SENTINEL ||
            r.y16[j] != HALF_SENTINEL || r.y16[GUARD + y + j] != HALF_SENTINEL) return false;
    }
    return true;
}

int run(int hv, int tokens, int regime, bool silu, const std::string& selected, int& checks) {
    const int c = (2 * HK + hv) * S, width = hv * S;
    const size_t ns = (size_t) S * hv * S, ny = (size_t) tokens * width;
    std::vector<float> initial(ns + 2 * GUARD, SENTINEL), y0(ny + 2 * GUARD, SENTINEL);
    std::vector<uint16_t> y160(ny + 2 * GUARD, HALF_SENTINEL);
    std::vector<float> h((size_t) std::max(tokens, 1) * c), gate((size_t) std::max(tokens, 1) * hv), beta(gate.size()),
        z((size_t) std::max(tokens, 1) * width), gamma(S);
    for (size_t i = 0; i < ns; ++i) initial[GUARD + i] = 0.1f * std::sin((float) (i + 1) * 0.013f);
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < c; ++i)
            h[(size_t) t * c + i] = std::sin((float) (t * 3 + i * 7) * 0.019f) / std::sqrt((float) S);
        for (int head = 0; head < hv; ++head) {
            gate[(size_t) t * hv + head] = regime == 1 ? (t % 3 == 0 ? -1000.0f : 0.0f)
                                                       : regime == 2 ? -1e-8f : -0.03f * (1 + (head + t) % 5);
            beta[(size_t) t * hv + head] = regime == 1 ? (float) (t % 2) : regime == 2 ? 1.0f : 0.6f;
        }
    }
    for (size_t i = 0; i < z.size(); ++i) z[i] = std::sin((float) i * 0.11f) * (regime == 1 ? 80.0f : 2.0f);
    for (int i = 0; i < S; ++i) gamma[i] = 0.9f + 0.1f * std::sin((float) i);
    P::Device device;
    float* state = device.upload(initial);
    float* y = device.upload(y0);
    uint16_t* y16 = device.upload(y160);
    const float *dh = device.upload(h), *dg = device.upload(gate), *db = device.upload(beta),
                *dz = device.upload(z), *dgm = device.upload(gamma);
    void* stream = P::stream();
    const auto call = [&](int begin, int count) {
        k::gdn_recurrence(state + GUARD, dh + (size_t) begin * c, dg + (size_t) begin * hv, db + (size_t) begin * hv,
                          dz + (size_t) begin * width, dgm, 1e-6f, y + GUARD + (size_t) begin * width,
                          y16 + GUARD + (size_t) begin * width, count, stream, HK, hv, silu);
    };
    const auto execute = [&](bool chunk, bool split) {
        select(chunk ? selected.c_str() : nullptr);
        P::copy_in(state, initial.data(), initial.size());
        P::copy_in(y, y0.data(), y0.size());
        P::copy_in(y16, y160.data(), y160.size());
        if (split && tokens > 1) {
            call(0, tokens / 2);
            P::sync();  // an actual caller boundary; no state reset
            call(tokens / 2, tokens - tokens / 2);
        } else {
            call(0, tokens);
        }
        Result r{initial, y0, y160};
        P::copy_out(r.state.data(), state, r.state.size());
        P::copy_out(r.y.data(), y, r.y.size());
        P::copy_out(r.y16.data(), y16, r.y16.size());
        return r;
    };
    const Result legacy = execute(false, false);
    int failures = 0;
    for (const bool split : {false, true}) {
        const Result chunk = execute(true, split);
        const double se = error(legacy.state, chunk.state), ye = error(legacy.y, chunk.y), he = half_error(legacy.y16, chunk.y16);
        const bool ok = se <= FP32_TOL && ye <= FP32_TOL && he <= FP16_TOL && guards(chunk, ns, ny) && guards(legacy, ns, ny);
        ++checks;
        if (!ok) ++failures;
        std::printf("%s chunk=%s hv=%d T=%d regime=%d silu=%d split=%d state=%.3e raw=%.3e fp16=%.3e\n",
                    ok ? "PASS" : "FAIL", selected.c_str(), hv, tokens, regime, (int) silu, (int) split, se, ye, he);
    }
    return failures;
}
}  // namespace

int main(int argc, char** argv) try {
    const char* setting = argc == 2 ? argv[1] : "1";
    const int chunk = k::gdn_chunk_request(setting);
    if (argc > 2 || chunk == 0) { std::fprintf(stderr, "usage: gdn_chunk_parity [1|16|32|64]\n"); return 2; }
    const std::string selected = std::to_string(chunk);
    select(selected.c_str());
    void* stream = P::stream();
    for (const int hv : {32, 48})
        if (k::gdn_recurrence_chunk_size(stream, HK, hv) != chunk) {
            std::printf("SKIP: chunk %d unavailable for %d/%d heads; legacy fallback is NOT a chunk parity pass\n", chunk, HK, hv);
            return 77;
        }
    int failures = 0, checks = 0;
    for (const int hv : {32, 48})
        for (const int t : {0, 1, chunk - 1, chunk, chunk + 1, 2 * chunk + 1})
            for (int regime = 0; regime < 3; ++regime)
                for (const bool silu : {false, true}) failures += run(hv, t, regime, silu, selected, checks);
    std::printf("gdn_chunk_parity: %d checks, %d failures; device numerical parity only, no speed claim\n", checks, failures);
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "gdn_chunk_parity: %s\n", e.what());
    return 1;
}
