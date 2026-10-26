// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/mmvq_rows_parity.cpp - STRATA_MMVQ_ROWS and STRATA_MMVQ_TILE_COLS: native_mmvq with 2 and 4 output rows a
// sub-group (or work-group) and 16 columns a pass against the established layout (1 row, 8 columns a pass), bitwise
// (memcmp), and the time of each.
//
//   * the model's dense shapes and formats: 2560-wide rows (qkv, z, q, the head: the sub-group layout) and 6144-wide
//     rows (ssm_out, attn_output: the work-group layout), Q4_K, Q5_K, Q6_K and IQ4_XS;
//   * columns 1..16 (past 8 the product runs in tiles of 8), a row count that is no multiple of 8 (the tail rows);
//   * the time per call of each layout at 1, 2, 4, 8 and 16 columns.
//
//     build/mmvq_rows_parity [reps]
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "parity_device.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

namespace P = strata::parity;
namespace k = strata::kernels;

struct Fmt { int type; const char* name; int bsz; std::vector<int> halves; };   // a 256-value block, its FP16 offsets
const Fmt kFmts[] = {{12, "Q4_K", 144, {0, 2}}, {13, "Q5_K", 176, {0, 2}}, {14, "Q6_K", 210, {208}},
                     {23, "IQ4_XS", 136, {0}}};
struct Shape { int n_in, n_out; };
const Shape kShapes[] = {{2560, 10240 + 40}, {6144, 2560 + 24}, {2560, 65536}};

// random blocks with small finite scales (0.01 in FP16) where the format keeps one
std::vector<uint8_t> blocks(const Fmt& f, size_t n, std::mt19937& rng) {
    std::vector<uint8_t> w(n * (size_t) f.bsz);
    for (auto& b : w) b = (uint8_t) rng();
    for (size_t i = 0; i < n; ++i)
        for (int o : f.halves) { w[i * f.bsz + o] = 0x1f; w[i * f.bsz + o + 1] = 0x21; }
    return w;
}

}  // namespace

int main(int argc, char** argv) try {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    void* s = P::stream();
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    int failures = 0, checks = 0;
    for (const Fmt& f : kFmts) {
        for (const Shape& sh : kShapes) {
            if (sh.n_out > 20000 && f.type != 13) continue;   // the head's shape: its own format only
            const size_t nb = (size_t) sh.n_out * (sh.n_in / 256);
            const std::vector<uint8_t> w = blocks(f, nb, rng);
            void* dw = P::alloc_bytes(w.size());
            P::copy_bytes_in(dw, w.data(), w.size());
            constexpr int kCols = 16;
            std::vector<float> x((size_t) kCols * sh.n_in);
            for (auto& v : x) v = nd(rng);
            float* dx = static_cast<float*>(P::alloc_bytes(x.size() * 4));
            P::copy_bytes_in(dx, x.data(), x.size() * 4);
            void* xq = P::alloc_bytes((size_t) kCols * sh.n_in / 32 * 36);
            k::quantize_q8_1_rows(dx, kCols, sh.n_in, xq, s);
            float* dy = static_cast<float*>(P::alloc_bytes((size_t) kCols * sh.n_out * 4));
            std::vector<float> ref((size_t) kCols * sh.n_out), got(ref.size());
            for (int nc = 1; nc <= kCols; ++nc) {
                k::native_mmvq_set_rows(1, 1);
                k::native_mmvq_set_tile_cols(8);
                k::native_mmvq(f.type, dw, xq, dy, sh.n_in, sh.n_out, nc, s);
                P::sync();
                P::copy_bytes_out(ref.data(), dy, (size_t) nc * sh.n_out * 4);
                for (int tile : {8, 16})
                    for (int r : {1, 2, 4}) {
                        if (tile == 8 && r == 1) continue;   // the reference
                        k::native_mmvq_set_rows(r, r);
                        k::native_mmvq_set_tile_cols(tile);
                        P::copy_bytes_in(dy, std::vector<float>((size_t) nc * sh.n_out, -1.f).data(), (size_t) nc * sh.n_out * 4);
                        k::native_mmvq(f.type, dw, xq, dy, sh.n_in, sh.n_out, nc, s);
                        P::sync();
                        P::copy_bytes_out(got.data(), dy, (size_t) nc * sh.n_out * 4);
                        ++checks;
                        if (std::memcmp(ref.data(), got.data(), (size_t) nc * sh.n_out * 4) != 0) {
                            ++failures;
                            std::printf("FAIL %-6s %d x %d cols %d rows %d tile %d\n", f.name, sh.n_out, sh.n_in, nc, r, tile);
                        }
                    }
            }
            // time per call: (tile, rows) = (8, 1) the established layout, then the candidates
            const int combos[][2] = {{8, 1}, {8, 2}, {16, 1}, {16, 2}, {16, 4}};
            for (int nc : {1, 2, 4, 8, 16}) {
                std::printf("%-6s %6d x %4d cols %2d:", f.name, sh.n_out, sh.n_in, nc);
                for (const auto& c : combos) {
                    k::native_mmvq_set_tile_cols(c[0]);
                    k::native_mmvq_set_rows(c[1], c[1]);
                    k::native_mmvq(f.type, dw, xq, dy, sh.n_in, sh.n_out, nc, s);
                    P::sync();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) k::native_mmvq(f.type, dw, xq, dy, sh.n_in, sh.n_out, nc, s);
                    P::sync();
                    const double us =
                        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                    std::printf("  t%d/r%d %7.1f", c[0], c[1], us);
                }
                std::printf(" us\n");
            }
            k::native_mmvq_set_tile_cols(8);
            k::native_mmvq_set_rows(1, 1);
        }
    }
    std::printf("mmvq_rows_parity: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mmvq_rows_parity: %s\n", e.what());
    return 1;
}
