// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/mmvq_slm_parity.cpp - STRATA_MMVQ_SLM: native_mmvq (and through it iq_mmvq) and native_mmvq_pair with
// the window's activations staged in local memory against the established kernels, bitwise (memcmp), and the time of
// each.
//
//   * the dense products of the window: ssm_out and attn_output (6144-wide rows: the work-group layout), the shared
//     expert's gate and up (2560 x 640, also as the pair) and down (640 x 2560), and the 2560-wide qkv rows (the
//     sub-group layout), in the formats the model gives them: Q4_K, Q5_K, Q6_K (GGUF order and the split rows),
//     IQ4_XS, IQ3_S (iq_mmvq), IQ4_NL, Q8_0 and Q2_0, each where its block divides the row;
//   * columns 1..16, a row count that is no multiple of the work-group's rows (the tail), 1, 2 and 4 rows a sub-group,
//     1, 4 and 8 sub-groups a work-group, local-memory budgets of one and several rounds, and activations 16-byte and
//     only 4-byte aligned (the copy's two paths);
//   * the time per call at 1, 4, 8 and 16 columns: the established layout as the window runs it (16 columns a pass, 2
//     rows a sub-group and a work-group) and the staged one at each rows x sub-groups.
//
//     build/mmvq_slm_parity [reps] [--time-only | --parity-only | --parity-quick]
//
// --parity-quick (the OpenCL CPU device, STRATA_TEST_CPU_DEVICE=1, where a call takes milliseconds): parity only, at
// 1, 2, 5, 8, 9, 13 and 16 columns, without the 10280-row qkv shape.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"

#include "parity_device.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace P = strata::parity;
namespace k = strata::kernels;

// a format: its type, name, block bytes and values, and its FP16 scale offsets (set to a small finite value), the
// type native_mmvq is called with (the split-row layouts rewrite GGUF blocks in place first)
struct Fmt { int type; const char* name; int bsz, qk; std::vector<int> halves; int call; };
const Fmt kFmts[] = {
    {12, "Q4_K", 144, 256, {0, 2}, 12},          {13, "Q5_K", 176, 256, {0, 2}, 13},
    {14, "Q6_K", 210, 256, {208}, 14},           {14, "Q6_Krow", 210, 256, {208}, k::kNativeQ6KRows},
    {23, "IQ4_XS", 136, 256, {0}, 23},           {21, "IQ3_S", 110, 256, {0}, 21},
    {20, "IQ4_NL", 18, 32, {0}, 20},             {8, "Q8_0", 34, 32, {0}, 8},
    {42, "Q2_0", 18, 64, {0}, 42},
};
struct Shape { const char* name; int n_in, n_out; bool pair; };
const Shape kShapes[] = {{"ssm_out", 6144, 2560 + 24, false},
                         {"shexp_gu", 2560, 640, true},
                         {"shexp_down", 640, 2560, false},
                         {"qkv", 2560, 10240 + 40, false}};

std::vector<uint8_t> blocks(const Fmt& f, size_t n, std::mt19937& rng) {
    std::vector<uint8_t> w(n * (size_t) f.bsz);
    for (auto& b : w) b = (uint8_t) rng();
    for (size_t i = 0; i < n; ++i)
        for (int o : f.halves) { w[i * f.bsz + o] = 0x1f; w[i * f.bsz + o + 1] = 0x21; }
    return w;
}

k::MmvqSlm off() { return k::MmvqSlm{}; }
k::MmvqSlm on(int rows, int sgs, int kb) {
    k::MmvqSlm c;
    c.on = true;
    c.rows = rows;
    c.sgs = sgs;
    c.kb = kb;
    c.rules = {{k::MmvqRole::any, -1, 1, 16, rows, sgs}};
    return c;
}

// The rules (STRATA_MMVQ_SLM_ROLES' parser, the default table's picks by role, type and columns, the overrides, the
// role scope): host-only, so they run before any kernel.  Returns the failures.
int check_rules(int& checks) {
    int failures = 0;
    const auto expect = [&](const char* what, bool ok) {
        ++checks;
        if (!ok) { ++failures; std::printf("FAIL rules: %s\n", what); }
    };
    const auto throws = [&](const char* what, const std::function<void()>& f) {
        bool threw = false;
        try { f(); } catch (const std::invalid_argument&) { threw = true; }
        expect(what, threw);
    };
    const k::MmvqSlm saved = k::native_mmvq_slm();
    k::MmvqSlm c = saved;
    c.on = true;
    c.rows = c.sgs = 0;
    if (!std::getenv("STRATA_MMVQ_SLM_ROLES")) {   // the default table's measured verdicts
        k::native_mmvq_set_slm(c);
        const auto pick = [&](k::MmvqRole r, int type, int nc) {
            k::MmvqRoleScope role(r);
            return k::native_mmvq_slm_pick(type, nc);
        };
        const auto is = [](const std::optional<k::MmvqSlmPick>& p, int rows, int sgs) {
            return p && p->rows == rows && p->sgs == sgs;
        };
        expect("an unnamed role stays established", !pick(k::MmvqRole::other, 12, 16));
        expect("qkv Q4_K from 4 columns stays established", !pick(k::MmvqRole::qkv, 12, 4));
        expect("qkv Q4_K at 1 column stages r4/s4", is(pick(k::MmvqRole::qkv, 12, 1), 4, 4));
        expect("ssm_out Q4_K at 16 columns stages r1/s16", is(pick(k::MmvqRole::ssm_out, 12, 16), 1, 16));
        expect("ssm_out Q4_K at 8 columns stays established", !pick(k::MmvqRole::ssm_out, 12, 8));
        expect("ssm_out Q6_K rows at 16 columns stages r2/s16", is(pick(k::MmvqRole::ssm_out, k::kNativeQ6KRows, 16), 2, 16));
        expect("ssm_out IQ4_XS stays established", !pick(k::MmvqRole::ssm_out, 23, 16));
        expect("ssm_out Q8_0 at 3 columns stages r2/s16", is(pick(k::MmvqRole::ssm_out, 8, 3), 2, 16));
        expect("shexp_down IQ4_NL stays established", !pick(k::MmvqRole::shexp_down, 20, 16));
        expect("shexp_gu Q5_K at 4 columns stages r1/s8", is(pick(k::MmvqRole::shexp_gu, 13, 4), 1, 8));
        c.rows = 2;
        k::native_mmvq_set_slm(c);
        expect("the rows override wins", is(pick(k::MmvqRole::ssm_out, 12, 16), 2, 16));
        c.rows = 0;
        c.on = false;
        k::native_mmvq_set_slm(c);
        expect("off picks nothing", !pick(k::MmvqRole::ssm_out, 12, 16));
    }
    {
        k::MmvqRoleScope a(k::MmvqRole::qkv);
        {
            k::MmvqRoleScope b(k::MmvqRole::ssm_out);
            expect("an inner scope names its role", k::native_mmvq_role() == k::MmvqRole::ssm_out);
        }
        expect("a scope restores the outer role", k::native_mmvq_role() == k::MmvqRole::qkv);
    }
    expect("no scope: other", k::native_mmvq_role() == k::MmvqRole::other);
    const auto any = k::native_mmvq_slm_rules("*:*:1-16:2:8,qkv:q8rows:1-1:4:4,");
    expect("the parser reads wildcards and names", any.size() == 2 && any[0].role == k::MmvqRole::any &&
                                                       any[0].type == -1 && any[0].max_cols == 16 &&
                                                       any[1].type == k::kNativeQ8Rows && any[1].rows == 4);
    expect("an empty table has no rules", k::native_mmvq_slm_rules("").empty());
    throws("a rule without a column range", [] { k::native_mmvq_slm_rules("ssm_out:12:16:1:16"); });
    throws("an unknown role", [] { k::native_mmvq_slm_rules("ssm_in:12:1-2:1:8"); });
    throws("an unknown type", [] { k::native_mmvq_slm_rules("ssm_out:x:1-2:1:8"); });
    throws("an empty rule", [] { k::native_mmvq_slm_rules("ssm_out:12:1-2:1:8,,qkv:8:1-1:2:4"); });
    throws("rows 3", [&] { k::MmvqSlm b = c; b.rules = k::native_mmvq_slm_rules("ssm_out:12:1-2:3:8"); k::native_mmvq_set_slm(b); });
    throws("17 columns", [&] { k::MmvqSlm b = c; b.rules = k::native_mmvq_slm_rules("ssm_out:12:1-17:1:8"); k::native_mmvq_set_slm(b); });
    throws("min above max", [&] { k::MmvqSlm b = c; b.rules = k::native_mmvq_slm_rules("ssm_out:12:5-4:1:8"); k::native_mmvq_set_slm(b); });
    k::native_mmvq_set_slm(saved);
    std::printf("rules: %d checks, %d failures\n", checks, failures);
    return failures;
}

// --flush MiB: every timed call follows a write of that many MiB (the engine's products read cold weights: the layers
// between two calls of one product evict them), and the flush alone, timed the same way, is subtracted
void* g_flush = nullptr;
size_t g_flush_bytes = 0;
double g_flush_us = 0;
void flush(int i) {
    if (g_flush) strata::core::Runtime::get().stream(P::stream()).memset(g_flush, i & 0xff, g_flush_bytes);
}
double time_us(int reps, const std::function<void()>& f) {
    f();
    P::sync();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        flush(i);
        f();
    }
    P::sync();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps - g_flush_us;
}
// the window's role for a shape (the rules' key: MmvqRoleScope)
k::MmvqRole role_of(const std::string& shape) {
    return shape == "ssm_out" ? k::MmvqRole::ssm_out : shape == "shexp_gu" ? k::MmvqRole::shexp_gu
         : shape == "shexp_down" ? k::MmvqRole::shexp_down : shape == "qkv" ? k::MmvqRole::qkv : k::MmvqRole::other;
}
// the process's rule table (STRATA_MMVQ_SLM_ROLES or the default), read before any test replaces it, switched on: the
// engine's STRATA_MMVQ_SLM=1 with no rows or sub-groups override
k::MmvqSlm g_roles;
k::MmvqSlm roles_on() {
    k::MmvqSlm c = g_roles;
    c.on = true;
    c.rows = c.sgs = 0;
    return c;
}

}  // namespace

int main(int argc, char** argv) try {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
    g_roles = k::native_mmvq_slm();
    const auto has = [&](const char* o) {
        for (int i = 2; i < argc; ++i)
            if (std::string(argv[i]) == o) return i;
        return 0;
    };
    const bool time_only = has("--time-only");
    const bool quick = has("--parity-quick");
    const bool parity_only = quick || has("--parity-only");
    if (const int i = has("--flush"); i && i + 1 < argc) {
        g_flush_bytes = (size_t) std::atoi(argv[i + 1]) << 20;
        g_flush = P::alloc_bytes(g_flush_bytes);
        g_flush_us = time_us(reps, [] {});
        std::printf("flush %zu MiB between calls: %.1f us, subtracted\n", g_flush_bytes >> 20, g_flush_us);
    }
    auto skip_cols = [&](int nc) { return quick && nc != 1 && nc != 2 && nc != 5 && nc != 8 && nc != 9 && nc != 13 && nc != 16; };
    void* s = P::stream();
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    int checks = 0;
    int failures = check_rules(checks);
    if (has("--rules-only")) {
        std::printf("mmvq_slm_parity: %d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    }
    constexpr int kCols = 16;
    const int configs[][3] = {{1, 8, 36}, {2, 8, 36}, {4, 8, 36}, {2, 1, 36}, {2, 4, 20}, {1, 4, 64}, {4, 1, 8}};
    for (const Fmt& f : kFmts) {
        for (const Shape& sh : kShapes) {
            if (sh.n_in % f.qk != 0 || (quick && sh.n_out > 10000)) continue;
            if (f.call == k::kNativeQ6KRows && !k::native_q6_k_rows_ok(sh.n_in)) continue;
            const int n_mats = sh.pair ? 2 : 1;
            const size_t nb = (size_t) sh.n_out * (sh.n_in / f.qk);
            void* dw[2] = {};
            for (int m = 0; m < n_mats; ++m) {
                const std::vector<uint8_t> w = blocks(f, nb, rng);
                dw[m] = P::alloc_bytes(w.size());
                P::copy_bytes_in(dw[m], w.data(), w.size());
                if (f.call == k::kNativeQ6KRows) k::native_q6_k_to_rows(dw[m], sh.n_in, sh.n_out, s);
            }
            std::vector<float> x((size_t) kCols * sh.n_in);
            for (auto& v : x) v = nd(rng);
            float* dx = static_cast<float*>(P::alloc_bytes(x.size() * 4));
            P::copy_bytes_in(dx, x.data(), x.size() * 4);
            const size_t xq_bytes = (size_t) kCols * sh.n_in / 32 * 36;
            void* xq = P::alloc_bytes(xq_bytes);
            k::quantize_q8_1_rows(dx, kCols, sh.n_in, xq, s);
            // the same activations 4 bytes in: only the copy's word path applies
            std::vector<uint8_t> h(xq_bytes);
            P::copy_bytes_out(h.data(), xq, xq_bytes);
            void* xq4 = static_cast<uint8_t*>(P::alloc_bytes(xq_bytes + 16)) + 4;
            P::copy_bytes_in(xq4, h.data(), xq_bytes);
            const size_t ybytes = (size_t) kCols * sh.n_out * 4;
            float* dy[2] = {static_cast<float*>(P::alloc_bytes(ybytes)), static_cast<float*>(P::alloc_bytes(ybytes))};
            std::vector<float> ref[2] = {std::vector<float>((size_t) kCols * sh.n_out),
                                         std::vector<float>((size_t) kCols * sh.n_out)};
            std::vector<float> got((size_t) kCols * sh.n_out);
            auto run = [&](const void* xa, int nc, bool pair) {
                if (pair && k::native_mmvq_pair(f.call, dw[0], dw[1], xa, dy[0], dy[1], sh.n_in, sh.n_out, nc, s)) return;
                for (int m = 0; m < n_mats; ++m) k::native_mmvq(f.call, dw[m], xa, dy[m], sh.n_in, sh.n_out, nc, s);
            };
            auto check = [&](const char* what, int nc, const k::MmvqSlm& c, bool aligned, bool pair) {
                const size_t bytes = (size_t) nc * sh.n_out * 4;
                for (int m = 0; m < n_mats; ++m)
                    P::copy_bytes_in(dy[m], std::vector<float>((size_t) nc * sh.n_out, -1.f).data(), bytes);
                run(aligned ? xq : xq4, nc, pair);
                P::sync();
                for (int m = 0; m < n_mats; ++m) {
                    P::copy_bytes_out(got.data(), dy[m], bytes);
                    ++checks;
                    if (std::memcmp(ref[m].data(), got.data(), bytes) != 0) {
                        ++failures;
                        std::printf("FAIL %-7s %-10s %s cols %2d rows %d sgs %d kb %d %s mat %d\n", f.name, sh.name, what,
                                    nc, c.rows, c.sgs, c.kb, aligned ? "a16" : "a4", m);
                    }
                }
            };
            if (!time_only)
                for (int nc = 1; nc <= kCols; ++nc) {
                    if (skip_cols(nc)) continue;
                    // the reference: the established layout, 8 columns a pass, 1 row, separate calls
                    k::native_mmvq_set_slm(off());
                    k::native_mmvq_set_rows(1, 1);
                    k::native_mmvq_set_tile_cols(8);
                    run(xq, nc, false);
                    P::sync();
                    for (int m = 0; m < n_mats; ++m) P::copy_bytes_out(ref[m].data(), dy[m], (size_t) nc * sh.n_out * 4);
                    for (const auto& c : configs) {
                        const k::MmvqSlm cfg = on(c[0], c[1], c[2]);
                        k::native_mmvq_set_slm(cfg);
                        check("slm", nc, cfg, true, false);
                        if (c[0] == 2 && c[1] == 8) check("slm", nc, cfg, false, false);
                        if (sh.pair) check("pair", nc, cfg, true, true);
                    }
                    k::native_mmvq_set_slm(off());
                }
            if (!time_only) {
                std::printf("parity %-7s %-10s %5d x %4d: %d checks, %d failures so far\n", f.name, sh.name, sh.n_out,
                            sh.n_in, checks, failures);
                std::fflush(stdout);
            }
            // time per call: the established layout as the window runs it (16 a pass, 2 rows), then the staged one
            for (int nc : {1, 4, 8, 16}) {
                if (parity_only) break;
                std::printf("%-7s %-10s %5d x %4d cols %2d: ", f.name, sh.name, sh.n_out, sh.n_in, nc);
                k::native_mmvq_set_slm(off());
                k::native_mmvq_set_tile_cols(8);
                k::native_mmvq_set_rows(1, 1);
                std::printf(" b8r1 %7.1f", time_us(reps, [&] { run(xq, nc, sh.pair); }));
                k::native_mmvq_set_tile_cols(16);
                k::native_mmvq_set_rows(2, 2);
                std::printf(" base %7.1f", time_us(reps, [&] { run(xq, nc, sh.pair); }));
                {
                    k::MmvqRoleScope role(role_of(sh.name));
                    k::native_mmvq_set_slm(roles_on());
                    std::printf(" roles %7.1f", time_us(reps, [&] { run(xq, nc, sh.pair); }));
                }
                for (int rows : {1, 2, 4})
                    for (int sgs : {4, 8, 16}) {
                        k::native_mmvq_set_slm(on(rows, sgs, 36));
                        std::printf("  r%d/s%-2d %6.1f", rows, sgs, time_us(reps, [&] { run(xq, nc, sh.pair); }));
                    }
                std::printf(" us\n");
            }
            k::native_mmvq_set_slm(off());
            k::native_mmvq_set_tile_cols(8);
            k::native_mmvq_set_rows(1, 1);
        }
    }
    // the shared expert end to end (STRATA_SHEXP_TILE=16 with the staged products against 8-row tiles of the
    // established ones): its unscaled rows (the combine scales them), byte for byte, and the time per window
    {
        const int N = 2560, FF = 640;
        const int triples[][3] = {{23, 21, 20}, {12, 13, 20}, {12, 12, 42}, {23, 14, 42}, {21, 23, 20}, {13, 12, 42}};
        auto fmt_of = [](int type) -> const Fmt& {
            for (const Fmt& f : kFmts)
                if (f.type == type && (type != 14 || f.call == k::kNativeQ6KRows)) return f;
            throw std::runtime_error("no format " + std::to_string(type));
        };
        std::vector<float> x((size_t) kCols * N);
        for (auto& v : x) v = nd(rng);
        float* dx = static_cast<float*>(P::alloc_bytes(x.size() * 4));
        P::copy_bytes_in(dx, x.data(), x.size() * 4);
        void* xq = P::alloc_bytes((size_t) kCols * N / 32 * 36);
        k::quantize_q8_1_rows(dx, kCols, N, xq, s);
        auto* wgi = static_cast<uint16_t*>(P::alloc_bytes((size_t) N * 2));
        float* sg = static_cast<float*>(P::alloc_bytes((size_t) kCols * FF * 4));
        float* su = static_cast<float*>(P::alloc_bytes((size_t) kCols * FF * 4));
        float* g = static_cast<float*>(P::alloc_bytes((size_t) kCols * 4 + 16));
        float* out = static_cast<float*>(P::alloc_bytes((size_t) kCols * N * 4));
        void* sq = P::alloc_bytes((size_t) kCols * N / 32 * 36);
        k::shared_expert_set_native_bf16(true);
        for (const auto& t : triples) {
            k::NativeSharedWeights nw;
            void* wm[3];
            const int dims[3][2] = {{N, FF}, {N, FF}, {FF, N}};
            int calls[3];
            for (int m = 0; m < 3; ++m) {
                const Fmt& f = fmt_of(t[m]);
                const std::vector<uint8_t> w = blocks(f, (size_t) dims[m][1] * (dims[m][0] / f.qk), rng);
                wm[m] = P::alloc_bytes(w.size());
                P::copy_bytes_in(wm[m], w.data(), w.size());
                if (f.call == k::kNativeQ6KRows) k::native_q6_k_to_rows(wm[m], dims[m][0], dims[m][1], s);
                calls[m] = f.call;
            }
            nw.gate_type = calls[0]; nw.gate_data = wm[0];
            nw.up_type = calls[1]; nw.up_data = wm[1];
            nw.down_type = calls[2]; nw.down_data = wm[2];
            nw.q8_1 = sq;
            auto call = [&](int T) { k::shared_expert_multi(T, dx, nullptr, nw, wgi, sg, su, g, out, N, FF, s, true, true, xq); };
            auto establish = [&] {
                k::native_mmvq_set_slm(off());
                k::shared_expert_set_tile(8);
                k::native_mmvq_set_tile_cols(16);
                k::native_mmvq_set_rows(2, 2);
            };
            std::vector<float> ref((size_t) kCols * N), got(ref.size());
            if (!time_only)
                for (int T = 1; T <= kCols; ++T) {
                    if (skip_cols(T)) continue;
                    establish();
                    call(T);
                    P::sync();
                    P::copy_bytes_out(ref.data(), out, (size_t) T * N * 4);
                    for (const auto& c : {on(2, 8, 36), on(1, 4, 36), on(4, 8, 36)})
                        for (int tile : {8, 16}) {
                            k::native_mmvq_set_slm(c);
                            k::shared_expert_set_tile(tile);
                            P::copy_bytes_in(out, std::vector<float>((size_t) T * N, -1.f).data(), (size_t) T * N * 4);
                            call(T);
                            P::sync();
                            P::copy_bytes_out(got.data(), out, (size_t) T * N * 4);
                            ++checks;
                            if (std::memcmp(ref.data(), got.data(), (size_t) T * N * 4) != 0) {
                                ++failures;
                                std::printf("FAIL shared expert t%d/%d/%d rows %d tile %d slm rows %d sgs %d\n", t[0], t[1],
                                            t[2], T, tile, c.rows, c.sgs);
                            }
                        }
                }
            if (!time_only) {
                std::printf("parity shared expert t%d/%d/%d: %d checks, %d failures so far\n", t[0], t[1], t[2], checks,
                            failures);
                std::fflush(stdout);
            }
            for (int T : {1, 4, 8, 16}) {
                if (parity_only) break;
                std::printf("shared expert t%d/%d/%d rows %2d: ", t[0], t[1], t[2], T);
                establish();
                std::printf(" base %7.1f", time_us(reps, [&] { call(T); }));
                k::native_mmvq_set_slm(roles_on());
                k::shared_expert_set_tile(16);
                std::printf(" roles %7.1f", time_us(reps, [&] { call(T); }));
                for (int rows : {1, 2, 4})
                    for (int sgs : {4, 8}) {
                        k::native_mmvq_set_slm(on(rows, sgs, 36));
                        k::shared_expert_set_tile(16);
                        std::printf("  r%d/s%d %6.1f", rows, sgs, time_us(reps, [&] { call(T); }));
                    }
                std::printf(" us\n");
            }
        }
        k::native_mmvq_set_slm(off());
        k::shared_expert_set_tile(8);
    }
    // the shared expert's SwiGLU + q8_1 at 16 rows is its two 8-row calls, byte for byte (STRATA_SHEXP_TILE=16)
    {
        const int n_ff = 640;
        std::vector<float> g((size_t) kCols * n_ff), u(g.size());
        for (auto& v : g) v = nd(rng);
        for (auto& v : u) v = nd(rng);
        float* dg = static_cast<float*>(P::alloc_bytes(g.size() * 4));
        float* du = static_cast<float*>(P::alloc_bytes(u.size() * 4));
        P::copy_bytes_in(dg, g.data(), g.size() * 4);
        P::copy_bytes_in(du, u.data(), u.size() * 4);
        const size_t qb = (size_t) kCols * n_ff / 32 * 36;
        void* q16 = P::alloc_bytes(qb);
        void* q8 = P::alloc_bytes(qb);
        k::native_swiglu_q8_1(dg, du, q16, n_ff, kCols, s);
        k::native_swiglu_q8_1(dg, du, q8, n_ff, 8, s);
        k::native_swiglu_q8_1(dg + 8 * n_ff, du + 8 * n_ff, static_cast<uint8_t*>(q8) + qb / 2, n_ff, 8, s);
        P::sync();
        std::vector<uint8_t> a(qb), b(qb);
        P::copy_bytes_out(a.data(), q16, qb);
        P::copy_bytes_out(b.data(), q8, qb);
        ++checks;
        if (std::memcmp(a.data(), b.data(), qb) != 0) {
            ++failures;
            std::printf("FAIL swiglu_q8_1 16 rows against 2 x 8\n");
        }
    }
    std::printf("mmvq_slm_parity: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mmvq_slm_parity: %s\n", e.what());
    return 1;
}
