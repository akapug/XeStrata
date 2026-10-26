// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstddef>

namespace strata::prefill {

// Portable chunk-WY algebra, shared by the host falsifier and the device backend.
// S_t = a_t S_(t-1) + k_t d_t, d_t = beta_t (v_t - a_t k_t^T S_(t-1)).
// Q/K: [n,S] with caller-supplied row stride; factors: [n,n] with row
// stride Chunk; state: one contiguous column.
// The triangular system is (I + L)d = beta*v - beta*P*k*S_initial:
// L_ij = beta_i prod(a_[j+1..i]) k_i^T k_j, j<i.
// A_ij = prod(a_[j+1..i]) q_i^T k_j, j<=i.
// Direct products NEVER divide by a cumulative decay: exp(gate) can underflow
// to zero. No clamp, inverse exp, low-precision conversion or state reset here.
// This is algebraic parity, not bit parity with token-ordered state updates.
inline constexpr int GDN_CHUNK_DEFAULT = 64;

inline constexpr bool gdn_chunk_supported(int chunk) {
    return chunk == 16 || chunk == 32 || chunk == GDN_CHUNK_DEFAULT;
}

// STRATA_GDN_CHUNK: unset/0 disables; 1 selects the named default; an explicit
// supported shape selects that shape. Invalid text must never enable a path.
inline constexpr int gdn_chunk_request(const char* value) {
    if (value == nullptr || value[0] == '\0') return 0;
    int n = 0;
    for (const char* p = value; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9' || n > GDN_CHUNK_DEFAULT) return 0;
        n = n * 10 + (*p - '0');
    }
    if (n == 1) return GDN_CHUNK_DEFAULT;
    return gdn_chunk_supported(n) ? n : 0;
}

inline constexpr size_t gdn_chunk_local_bytes(int chunk, int state_size) {
    return state_size > 0 ? (2 * (size_t) chunk * chunk + 2 * (size_t) chunk) * sizeof(float) : 0;
}

inline constexpr bool gdn_chunk_fits(int chunk, int state_size, size_t local_bytes, size_t group_size) {
    return gdn_chunk_supported(chunk) && state_size > 0 && group_size >= (size_t) state_size &&
           gdn_chunk_local_bytes(chunk, state_size) <= local_bytes;
}

template <int Chunk, int S> struct GdnChunk {
    static_assert(S > 0 && S % 4 == 0);
    static_assert(gdn_chunk_supported(Chunk));

    template <class R> static R decay(const R* a, int first, int last) {
        R p = 1;
        for (int t = first; t <= last; ++t) p *= a[t];
        return p;
    }

    // The token kernel's four independent chains and left-to-right final sum.
    template <class R, class F> static R dot(const R* a, const R* b, F fma) {
        R p[4] = {0, 0, 0, 0};
        for (int r = 0; r < S / 4; ++r)
            for (int j = 0; j < 4; ++j) p[j] = fma(a[j * (S / 4) + r], b[j * (S / 4) + r], p[j]);
        return p[0] + p[1] + p[2] + p[3];
    }

    // One cell of the two factors; parallel across cells on the device. These
    // FP32 factors are the extension boundary for a future matrix-engine path.
    template <class R, class F> static void factor(int i, int j, const R* q, const R* k, const R* a,
                                                  const R* beta, R* lower, R* output, F fma, int ld = S) {
        const size_t at = (size_t) i * Chunk + j;
        lower[at] = 0;
        output[at] = 0;
        if (j > i) return;
        const R p = decay(a, j + 1, i);
        if (j < i) lower[at] = beta[i] * p * dot(k + i * ld, k + j * ld, fma);
        output[at] = p * dot(q + i * ld, k + j * ld, fma);
    }

    template <class R, class F> static R delta(int i, const R* k, const R* a, const R* beta,
                                              const R* lower, R v, const R* state, const R* d, F fma, int ld = S) {
        R x = (v - decay(a, 0, i) * dot(k + i * ld, state, fma)) * beta[i];
        for (int j = 0; j < i; ++j) x = fma(-lower[(size_t) i * Chunk + j], d[j], x);
        return x;
    }

    // Unscaled output; the caller preserves 1/sqrt(S), the existing RMS norm,
    // z gate (sigmoid or SiLU), gamma and FP16 conversion at their boundaries.
    template <class R, class F> static R output(int i, const R* q, const R* a, const R* factor,
                                               const R* state, const R* d, F fma, int ld = S) {
        R x = decay(a, 0, i) * dot(q + i * ld, state, fma);
        for (int j = 0; j <= i; ++j) x = fma(factor[(size_t) i * Chunk + j], d[j], x);
        return x;
    }

    template <class R, class F> static R carry(int row, int n, const R* k, const R* a,
                                              R state, const R* d, F fma, int ld = S) {
        R x = decay(a, 0, n - 1) * state;
        for (int j = 0; j < n; ++j) x = fma(decay(a, j + 1, n - 1) * k[j * ld + row], d[j], x);
        return x;
    }
};

}  // namespace strata::prefill
