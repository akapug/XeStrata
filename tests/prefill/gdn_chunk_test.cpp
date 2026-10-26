// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/prefill/gdn_chunk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
using strata::prefill::GdnChunk;

struct Fixture {
    int s, hk, hv, tokens, ld;
    std::vector<double> h, a, beta, initial;
    Fixture(int width, int keys, int values, int count, int regime)
        : s(width), hk(keys), hv(values), tokens(count), ld((2 * hk + hv) * s + 7),
          h((size_t) (tokens + 1) * ld, std::numeric_limits<double>::quiet_NaN()),
          a((size_t) tokens * hv), beta(a.size()), initial((size_t) s * hv * s) {
        for (size_t i = 0; i < initial.size(); ++i)
            initial[i] = regime == 3 ? 0.0 : 0.2 * std::sin((double) (i + 1) * 0.13);
        for (int t = 0; t < tokens; ++t) {
            for (int i = 0; i < (2 * hk + hv) * s; ++i)
                h[(size_t) t * ld + i] = std::sin((double) (1 + t * 11 + i * 7) * 0.31) / std::sqrt((double) s);
            for (int head = 0; head < hv; ++head) {
                const size_t at = (size_t) t * hv + head;
                const double gate = regime == 1 ? (t % 3 == 0 ? -1000.0 : 0.0)
                                               : regime == 2 ? -1e-9 : -0.02 * (1 + (t + head) % 7);
                // Match the kernel boundary: gate has already been computed;
                // exp is not clamped and can produce a literal zero.
                a[at] = std::exp(gate);
                beta[at] = regime == 1 ? (double) (t % 2) : regime == 3 ? 0.0 : 0.3 + 0.1 * (head % 4);
            }
        }
    }
};

struct Result { std::vector<double> state, y; };

// Independent token recurrence in double, transcribed from GdnRecLane rather
// than from WY. Four fma chains per dot, modulo pairing, decay BEFORE update.
Result token(const Fixture& f, bool wrong_decay = false, bool wrong_heads = false, bool zero_initial = false) {
    Result r{f.initial, std::vector<double>((size_t) f.tokens * f.hv * f.s)};
    if (zero_initial) std::fill(r.state.begin(), r.state.end(), 0.0);
    for (int t = 0; t < f.tokens; ++t)
        for (int head = 0; head < f.hv; ++head) {
            const int kh = wrong_heads ? head / (f.hv / f.hk) : head % f.hk;
            const double* q = f.h.data() + (size_t) t * f.ld + kh * f.s;
            const double* k = q + f.hk * f.s;
            const double a = f.a[(size_t) t * f.hv + head], b = f.beta[(size_t) t * f.hv + head];
            for (int c = 0; c < f.s; ++c) {
                double p[4] = {};
                for (int row = 0; row < f.s / 4; ++row)
                    for (int g = 0; g < 4; ++g) {
                        const int j = g * (f.s / 4) + row;
                        p[g] = std::fma(r.state[((size_t) j * f.hv + head) * f.s + c], k[j], p[g]);
                    }
                const double v = f.h[(size_t) t * f.ld + 2 * f.hk * f.s + head * f.s + c];
                const double d = (v - (wrong_decay ? 1.0 : a) * (p[0] + p[1] + p[2] + p[3])) * b;
                double o[4] = {};
                for (int row = 0; row < f.s / 4; ++row)
                    for (int g = 0; g < 4; ++g) {
                        const int j = g * (f.s / 4) + row;
                        double& x = r.state[((size_t) j * f.hv + head) * f.s + c];
                        x = wrong_decay ? a * std::fma(k[j], d, x) : std::fma(a, x, k[j] * d);
                        o[g] = std::fma(x, q[j], o[g]);
                    }
                r.y[((size_t) t * f.hv + head) * f.s + c] = (o[0] + o[1] + o[2] + o[3]) / std::sqrt((double) f.s);
            }
        }
    return r;
}

template <int Chunk, int S, class R> Result chunk(const Fixture& f, int split = 0) {
    using W = GdnChunk<Chunk, S>;
    const auto fma = [](R a, R b, R c) { return std::fma(a, b, c); };
    std::vector<R> state(f.initial.begin(), f.initial.end()), h(f.h.begin(), f.h.end());
    Result result{{}, std::vector<double>((size_t) f.tokens * f.hv * S)};
    // A split models two independent recurrence calls (cancellation is checked
    // at the caller boundary), preserving the first call's final state.
    for (int call = 0; call < (split > 0 ? 2 : 1); ++call) {
        const int begin = call == 0 ? 0 : split;
        const int end = split > 0 && call == 0 ? split : f.tokens;
        for (int t = begin; t < end; t += Chunk) {
            const int n = std::min(Chunk, end - t);
            for (int head = 0; head < f.hv; ++head) {
                const R* q = h.data() + (size_t) t * f.ld + (head % f.hk) * S;
                const R* k = q + f.hk * S;
                R a[Chunk], beta[Chunk], lower[Chunk * Chunk], output[Chunk * Chunk];
                for (int i = 0; i < n; ++i) {
                    a[i] = (R) f.a[(size_t) (t + i) * f.hv + head];
                    beta[i] = (R) f.beta[(size_t) (t + i) * f.hv + head];
                }
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) W::factor(i, j, q, k, a, beta, lower, output, fma, f.ld);
                for (int c = 0; c < S; ++c) {
                    R st[S], d[Chunk];
                    for (int row = 0; row < S; ++row) st[row] = state[((size_t) row * f.hv + head) * S + c];
                    for (int i = 0; i < n; ++i) {
                        const R v = h[(size_t) (t + i) * f.ld + 2 * f.hk * S + head * S + c];
                        d[i] = W::delta(i, k, a, beta, lower, v, st, d, fma, f.ld);
                        result.y[((size_t) (t + i) * f.hv + head) * S + c] =
                            W::output(i, q, a, output, st, d, fma, f.ld) / std::sqrt((R) S);
                    }
                    for (int row = 0; row < S; ++row)
                        state[((size_t) row * f.hv + head) * S + c] = W::carry(row, n, k, a, st[row], d, fma, f.ld);
                }
            }
        }
    }
    result.state.assign(state.begin(), state.end());
    return result;
}

double error(const Result& a, const Result& b) {
    double worst = 0;
    for (const auto pair : {std::pair{&a.state, &b.state}, std::pair{&a.y, &b.y}})
        for (size_t i = 0; i < pair.first->size(); ++i) {
            const double x = (*pair.first)[i], y = (*pair.second)[i];
            if (!std::isfinite(x) || !std::isfinite(y)) return std::numeric_limits<double>::infinity();
            worst = std::max(worst, std::fabs(x - y) / (1.0 + std::fabs(x)));
        }
    return worst;
}

int bad = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); ++bad; }
}

template <int Chunk, int S> void parity(int hk, int hv, int tokens, int regime) {
    const Fixture f(S, hk, hv, tokens, regime);
    const Result want = token(f);
    const double e64 = error(want, chunk<Chunk, S, double>(f));
    const double e32 = error(want, chunk<Chunk, S, float>(f));
    check(e64 < 1e-11, "double chunk algebra vs independent token recurrence");
    check(e32 < 3e-6, "FP32 chunk algebra vs independent token recurrence");
    if (tokens > 1) {
        check(error(want, chunk<Chunk, S, double>(f, tokens / 2)) < 1e-11, "nonzero state carry across caller boundary");
        check(error(want, chunk<Chunk, S, float>(f, tokens / 2)) < 3e-6, "FP32 state carry across caller boundary");
    }
    std::printf("chunk=%d S=%d heads=%d/%d T=%d regime=%d double=%.3e float=%.3e\n",
                Chunk, S, hk, hv, tokens, regime, e64, e32);
}
}  // namespace

int main() {
    using namespace strata::prefill;
    const size_t bytes = gdn_chunk_local_bytes(GDN_CHUNK_DEFAULT, 128);
    check(gdn_chunk_fits(GDN_CHUNK_DEFAULT, 128, bytes, 128), "named default fits exact local-memory boundary");
    check(!gdn_chunk_fits(64, 128, bytes - 1, 128), "insufficient local memory falls back");
    check(!gdn_chunk_fits(64, 128, 65536, 127), "insufficient work-group size falls back");
    check(!gdn_chunk_fits(17, 128, 65536, 128), "unsupported chunk falls back");
    check(!gdn_chunk_fits(0, 128, 65536, 128), "disabled chunk falls back");
    for (const char* v : {"", "0", "17", "-1", "64x", "9999999999999999999999999"})
        check(gdn_chunk_request(v) == 0, "invalid or disabled config falls back");
    check(gdn_chunk_request(nullptr) == 0, "unset config leaves legacy dispatch unchanged");
    check(gdn_chunk_request("1") == GDN_CHUNK_DEFAULT, "explicit opt-in uses named default");
    check(gdn_chunk_request("16") == 16 && gdn_chunk_request("32") == 32 && gdn_chunk_request("64") == 64,
          "supported shapes selected without rebuild");
    for (const int t : {0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129})
        for (int regime = 0; regime < 4; ++regime) {
            parity<16, 8>(2, 4, t, regime);
            parity<32, 8>(2, 4, t, regime);
            parity<64, 8>(2, 4, t, regime);
        }
    // Actual engine state width and both supported head geometries; a tail and
    // more than one chunk, without a device, a model or the SYCL runtime.
    parity<64, 128>(16, 32, 65, 0);
    parity<64, 128>(16, 48, 65, 1);
    const Fixture f(8, 2, 4, 17, 0);
    const Result right = token(f);
    check(error(right, token(f, true)) > 1e-4, "fixture falsifies decay-after-update");
    check(error(right, token(f, false, true)) > 1e-4, "fixture falsifies interleaved head pairing");
    check(error(right, token(f, false, false, true)) > 1e-4, "fixture falsifies omitted nonzero initial state");
    const double zero[2] = {0.0, 1.0};
    check(GdnChunk<16, 8>::decay(zero, 0, 1) == 0.0, "zero decay remains zero without 0/0 cumulative ratio");
    std::printf("%s: host mathematical parity only; no GPU parity or speed claim\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
