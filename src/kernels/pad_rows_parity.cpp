// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/pad_rows_parity.cpp - homelab pad rows (window_prefill::pad_row, STRATA_BATCH_PAD_KEEPS_IDLE): an idle
// slot's row in a group window must leave that slot's conversation byte for byte.  N windows of four one-row slots,
// two of them pad rows (keep 0, K/V cell = the slot's history length, indexer position -1), through every kernel a
// batch window or its commit runs on a slot's state:
//   - the GDN conv history and recurrent state: gdn_conv_l2_slots + gdn_step_norm_slots with the window's own commit
//     (STRATA_BATCH_SELF_COMMIT + STRATA_BATCH_GDN_SLOTS) in every form, and the commit graph's gdn_conv_commit +
//     gdn_step_norm_multi with n_keep 0;
//   - the PLE history: ple_history_advance with keep 0;
//   - the K/V cells (fp16, INT8, Q4_0, K8V4) by the one-launch appends and the per-row ones, from the window's
//     per-row cell array (stride 1) in place of the steps;
//   - the indexer tail, dead key, pooled keys and block position (the one-launch append with its snapshot, the
//     per-row append), position -1.
// Each pad slot ends memcmp-identical to where it started (K/V: every cell below its history length; the one cell
// at it is exactly a single append there), the active slots end bitwise as a run without pads (keep null / the
// steps), and a run that keeps the pad rows does change the pad slots (the check sees a wipe).  Synthetic, no model.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace P = strata::parity;
namespace k = strata::kernels;

int g_fail = 0, g_checks = 0;
std::mt19937 rng(353);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.

template <typename T> T* dalloc(size_t n) { return static_cast<T*>(P::alloc_bytes(n * sizeof(T) + 64)); }
template <typename T> T* dput(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    P::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}
template <typename T> std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n);
    P::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}
template <typename T> void put(T* d, const std::vector<T>& h) { P::copy_bytes_in(d, h.data(), h.size() * sizeof(T)); }
std::vector<float> gauss(size_t n, float scale) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng) * scale;
    return v;
}
std::vector<uint8_t> noise(size_t n) {
    std::uniform_int_distribution<int> ud(0, 255);
    std::vector<uint8_t> v(n);
    for (uint8_t& x : v) x = (uint8_t) ud(rng);
    return v;
}
void expect(bool ok, const char* what, int slot) {
    ++g_checks;
    if (ok) return;
    ++g_fail;
    std::printf("    *** %s (slot %d) ***\n", what, slot);
}
template <typename T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

constexpr int NS = 4, W = 3;                  // slots (one row each), windows
constexpr int kPadOf[NS] = {0, 1, 0, 1};      // slots 1 and 3 idle: pad rows

// ---- GDN: the window's own commit (every form) and the commit graph's kernels
void gdn() {
    constexpr int S = 128, HK = 16, HV = 48, C = 2 * S * HK + S * HV, VD = S * HV;
    const size_t n_state = (size_t) S * S * HV, n_slot = n_state + (size_t) C * 3;
    void* s = P::stream();
    std::vector<std::vector<float>> init(NS);
    for (auto& v : init) {
        v = gauss(n_slot, 0.05f);
        for (size_t i = n_state; i < n_slot; ++i) v[i] *= 4.f;   // the conv history at the inputs' scale
    }
    std::vector<float*> qkv(W), gate(W), beta(W), z(W);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    for (int w = 0; w < W; ++w) {
        qkv[w] = dput(gauss((size_t) NS * C, 0.2f));
        std::vector<float> g((size_t) NS * HV), b((size_t) NS * HV);
        for (auto& x : g) x = -0.5f * ud(rng);
        for (auto& x : b) x = ud(rng);
        gate[w] = dput(g); beta[w] = dput(b);
        z[w] = dput(gauss((size_t) NS * VD, 1.f));
    }
    const float* conv_w = dput(gauss((size_t) C * 4, 0.3f));
    std::vector<float> gm(S);
    for (auto& x : gm) x = 1.f + 0.1f * gauss(1, 1.f)[0];
    const float* gamma = dput(gm);
    std::vector<int32_t> hk(NS);
    for (int j = 0; j < NS; ++j) hk[(size_t) j] = kPadOf[j] ? 0 : 1;
    const int32_t* keep = dput(hk);
    float* h = dalloc<float>((size_t) NS * C);
    float* y = dalloc<float>((size_t) NS * VD);
    float* st[NS];
    for (int j = 0; j < NS; ++j) st[j] = dalloc<float>(n_slot);
    const auto table = [&](const int32_t* kp) {
        k::GdnSlotTable t;
        t.n = NS;
        t.keep = kp;
        for (int j = 0; j < NS; ++j) { t.state[j] = st[j]; t.hist[j] = st[j] + n_state; t.first[j] = j; t.ntok[j] = 1; }
        return t;
    };
    // W windows from the initial states; the slots' end states and every window's outputs
    const auto run = [&](int form, const int32_t* kp, std::vector<std::vector<float>>& end,
                         std::vector<std::vector<float>>& ys) {
        for (int j = 0; j < NS; ++j) put(st[j], init[(size_t) j]);
        k::gdn_slots_set_form(form);
        const k::GdnSlotTable t = table(kp);
        ys.clear();
        for (int w = 0; w < W; ++w) {
            k::gdn_conv_l2_slots(t, qkv[w], conv_w, h, C, 2 * HK, 1e-6f, s, true);
            k::gdn_step_norm_slots(t, h, C, gate[w], beta[w], z[w], gamma, 1e-6f, y, HK, HV, true, s);
            P::sync();
            ys.push_back(fetch(y, (size_t) NS * VD));
        }
        end.assign(NS, {});
        for (int j = 0; j < NS; ++j) end[(size_t) j] = fetch(st[j], n_slot);
    };
    // the large register file forms (2 large, 4 block+large) build for the GPU only: the OpenCL CPU device refuses
    // -ze-opt-large-register-file; their keep test is the same line of code as the other forms' (GdnSlotsPlain)
    const bool cpu = std::getenv("STRATA_TEST_CPU_DEVICE") != nullptr;
    for (int f = 0; f < k::gdn_slots_forms(); ++f) {
        if (cpu && (f == 2 || f == 4)) {
            std::printf("  GDN slots form %d: skipped on the CPU device (large register file)\n", f);
            continue;
        }
        std::vector<std::vector<float>> a, b, ya, yb;
        run(f, keep, a, ya);
        run(f, nullptr, b, yb);
        for (int j = 0; j < NS; ++j) {
            if (kPadOf[j]) {
                expect(same(a[(size_t) j], init[(size_t) j]), "GDN slots: a pad row changed its slot's state or conv history", j);
                expect(!same(b[(size_t) j], init[(size_t) j]), "GDN slots: the kept row did not write (the check is blind)", j);
                continue;
            }
            expect(same(a[(size_t) j], b[(size_t) j]), "GDN slots: an active slot differs beside pad rows", j);
            bool ys = true;
            for (int w = 0; w < W; ++w)
                ys = ys && std::memcmp(ya[(size_t) w].data() + (size_t) j * VD, yb[(size_t) w].data() + (size_t) j * VD,
                                       (size_t) VD * 4) == 0;
            expect(ys, "GDN slots: an active slot's outputs differ beside pad rows", j);
        }
        std::printf("  GDN slots form %d: %d windows, pads kept their states\n", f, W);
    }
    // the commit graph (a window without the self-commit): its kernels keep nothing for n_keep 0
    const int32_t* zero = dput(std::vector<int32_t>{0});
    const int32_t* one = dput(std::vector<int32_t>{1});
    for (const int32_t* kp : {zero, one}) {
        put(st[0], init[0]);
        k::gdn_conv_commit(st[0] + n_state, qkv[0], C, kp, s);
        k::gdn_step_norm_multi(st[0], h, C, gate[0], beta[0], z[0], gamma, 1e-6f, y, HK, HV, 1, kp, s);
        P::sync();
        const bool kept = same(fetch(st[0], n_slot), init[0]);
        expect(kp == zero ? kept : !kept, kp == zero ? "GDN commit: n_keep 0 changed the state" : "GDN commit: n_keep 1 wrote nothing", 0);
    }
}

// ---- PLE: the window's history advance with keep 0
void ple() {
    void* s = P::stream();
    const size_t HS = (size_t) k::NG_HIST * k::NG_HC_DIM;
    const std::vector<float> init = gauss(HS, 1.f);
    float* hist = dput(init);
    float* ref = dput(init);
    const float* norm = dput(gauss((size_t) k::NG_HC_DIM, 1.f));
    const int32_t* zero = dput(std::vector<int32_t>{0});
    const int32_t* one = dput(std::vector<int32_t>{1});
    for (int w = 0; w < W; ++w) k::ple_history_advance(hist, norm, s, zero);
    P::sync();
    expect(same(fetch(hist, HS), init), "PLE: a pad row advanced its slot's history", 1);
    for (int w = 0; w < W; ++w) { k::ple_history_advance(hist, norm, s, one); k::ple_history_advance(ref, norm, s); }
    P::sync();
    const std::vector<float> got = fetch(hist, HS);
    expect(same(got, fetch(ref, HS)) && !same(got, init), "PLE: keep 1 is not the plain advance", 0);
    std::printf("  PLE: %d pad windows kept the history\n", W);
}

// ---- K/V: every format, one launch and per row, from the per-row cell array
void kv() {
    void* cs = P::stream();
    const k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, PG = (int) s.page_size, HD = H * D;
    const int cells = 64, pages = cells / PG, G8 = D / k::KV_Q8_GROUP;
    const int held[NS] = {6, 9, 0, 13};   // history lengths; a pad row's cell is its slot's length
    // the pools of one slot: the bytes of each, and what one cell of one head takes in it
    struct Pool { size_t bytes, unit; };
    const Pool pool[] = {{(size_t) cells * HD * 2, (size_t) D * 2}, {(size_t) cells * HD * 2, (size_t) D * 2},   // fp16 K, V
                         {(size_t) cells * HD, (size_t) D}, {(size_t) cells * HD, (size_t) D},                     // int8 K, V
                         {(size_t) cells * H * G8 * 2, (size_t) G8 * 2}, {(size_t) cells * H * G8 * 2, (size_t) G8 * 2},
                         {(size_t) cells * H * 144, 144}, {(size_t) cells * H * 144, 144}};                        // q4_0 K, V
    constexpr int NP = 8;
    std::vector<int32_t> tab_h[NS];
    int32_t* tab_d[NS];
    std::vector<uint8_t> init[NS][NP];
    for (int j = 0; j < NS; ++j) {
        tab_h[j].resize((size_t) pages);
        for (int p = 0; p < pages; ++p) tab_h[j][(size_t) p] = (p * 5 + 3 + j) % pages;   // a permuted page table
        tab_d[j] = dput(tab_h[j]);
        for (int q = 0; q < NP; ++q) init[j][q] = noise(pool[q].bytes);
    }
    std::vector<const float*> kc(W), vc(W);
    std::vector<int32_t*> apos(W), steps(W);
    for (int w = 0; w < W; ++w) {
        kc[w] = dput(gauss((size_t) NS * HD, 1.f));
        vc[w] = dput(gauss((size_t) NS * HD, 2.f));
        std::vector<int32_t> a(NS), st((size_t) NS * k::kStepCount);
        for (int j = 0; j < NS; ++j) {
            a[(size_t) j] = held[j] + (kPadOf[j] ? 0 : w);
            k::qsa_step_fill(st.data() + (size_t) j * k::kStepCount, a[(size_t) j], s);
        }
        apos[w] = dput(a);
        steps[w] = dput(st);
    }
    // one run: W windows into fresh copies of the pools; mode 0 one launch from the cell array, 1 one launch from the
    // steps, 2 per row from the cell array, 3 per row from the steps
    const auto run = [&](int fmt, int mode, std::vector<uint8_t> (&out)[NS][NP]) {
        uint8_t* d[NS][NP];
        for (int j = 0; j < NS; ++j)
            for (int q = 0; q < NP; ++q) d[j][q] = dput(init[j][q]);
        for (int w = 0; w < W; ++w) {
            const bool cell_array = mode == 0 || mode == 2;
            const int32_t* pos = cell_array ? apos[w] : steps[w];
            const int stride = cell_array ? 1 : k::kStepCount;
            if (mode < 2) {
                k::KvAppendSlots t, vt;
                for (int j = 0; j < NS; ++j) {
                    t.page_table[j] = vt.page_table[j] = tab_d[j];
                    if (fmt == 0) { t.k[j] = d[j][0]; t.v[j] = d[j][1]; }
                    else if (fmt == 1) {
                        t.k[j] = d[j][2]; t.v[j] = d[j][3];
                        t.k_scale[j] = (uint16_t*) d[j][4]; t.v_scale[j] = (uint16_t*) d[j][5];
                    } else if (fmt == 2) { t.k[j] = d[j][6]; t.v[j] = d[j][7]; }
                    else { t.k[j] = d[j][2]; t.k_scale[j] = (uint16_t*) d[j][4]; vt.k[j] = d[j][7]; }
                }
                if (fmt == 0) k::kv_append_slots(t, pos, stride, kc[w], vc[w], HD, NS, s, cs);
                else if (fmt == 1) k::kv_append_q8_slots(t, pos, stride, kc[w], vc[w], HD, NS, 2, s, cs);
                else if (fmt == 2) k::kv_append_q4_slots(t, pos, stride, NS, kc[w], vc[w], 2, s, cs);
                else {
                    k::kv_append_q8_slots(t, pos, stride, kc[w], kc[w], HD, NS, 1, s, cs);
                    k::kv_append_q4_slots(vt, pos, stride, NS, vc[w], vc[w], 1, s, cs);
                }
            } else
                for (int j = 0; j < NS; ++j) {
                    const int32_t* p = pos + (size_t) j * stride;
                    const float* kj = kc[w] + (size_t) j * HD;
                    const float* vj = vc[w] + (size_t) j * HD;
                    auto* ks = (uint16_t*) d[j][4];
                    auto* vs = (uint16_t*) d[j][5];
                    if (fmt == 0) k::kv_append_step((uint16_t*) d[j][0], (uint16_t*) d[j][1], tab_d[j], p, kj, vj, s, cs);
                    else if (fmt == 1)
                        k::kv_append_q8_step((int8_t*) d[j][2], (int8_t*) d[j][3], ks, vs, tab_d[j], p, kj, vj, s, cs);
                    else if (fmt == 2) k::kv_append_q4_step(d[j][6], d[j][7], tab_d[j], p, kj, vj, s, cs);
                    else {
                        k::kv_append_q8_step((int8_t*) d[j][2], (int8_t*) d[j][2], ks, ks, tab_d[j], p, kj, kj, s, cs);
                        k::kv_append_q4_step(d[j][7], d[j][7], tab_d[j], p, vj, vj, s, cs);
                    }
                }
        }
        P::sync();
        for (int j = 0; j < NS; ++j)
            for (int q = 0; q < NP; ++q) out[j][q] = fetch(d[j][q], pool[q].bytes);
    };
    // every cell below `n` of slot j's pool q as it was: row (page * H + head) * PG + cell % PG of `unit` bytes
    const auto history_kept = [&](int j, int q, const std::vector<uint8_t>& got, int n) {
        for (int c = 0; c < n; ++c)
            for (int hh = 0; hh < H; ++hh) {
                const size_t row = ((size_t) tab_h[j][(size_t) (c / PG)] * H + hh) * PG + c % PG;
                if (std::memcmp(got.data() + row * pool[q].unit, init[j][q].data() + row * pool[q].unit, pool[q].unit))
                    return false;
            }
        return true;
    };
    static const char* const kFmt[] = {"fp16", "int8", "q4_0", "k8v4"};
    for (int fmt = 0; fmt < 4; ++fmt) {
        std::vector<uint8_t> r[4][NS][NP];
        for (int mode = 0; mode < 4; ++mode) run(fmt, mode, r[mode]);
        // the pad cell alone: the initial pools with one append at the slot's length (the last window's K/V)
        std::vector<uint8_t> one[NS][NP];
        {
            uint8_t* d[NS][NP];
            for (int j = 0; j < NS; ++j)
                for (int q = 0; q < NP; ++q) d[j][q] = dput(init[j][q]);
            for (int j = 0; j < NS; ++j) {
                if (!kPadOf[j]) continue;
                const int32_t* p = apos[W - 1] + j;
                const float* kj = kc[W - 1] + (size_t) j * HD;
                const float* vj = vc[W - 1] + (size_t) j * HD;
                auto* ks = (uint16_t*) d[j][4];
                auto* vs = (uint16_t*) d[j][5];
                if (fmt == 0) k::kv_append_step((uint16_t*) d[j][0], (uint16_t*) d[j][1], tab_d[j], p, kj, vj, s, cs);
                else if (fmt == 1) k::kv_append_q8_step((int8_t*) d[j][2], (int8_t*) d[j][3], ks, vs, tab_d[j], p, kj, vj, s, cs);
                else if (fmt == 2) k::kv_append_q4_step(d[j][6], d[j][7], tab_d[j], p, kj, vj, s, cs);
                else {
                    k::kv_append_q8_step((int8_t*) d[j][2], (int8_t*) d[j][2], ks, ks, tab_d[j], p, kj, kj, s, cs);
                    k::kv_append_q4_step(d[j][7], d[j][7], tab_d[j], p, vj, vj, s, cs);
                }
            }
            P::sync();
            for (int j = 0; j < NS; ++j)
                for (int q = 0; q < NP; ++q) one[j][q] = fetch(d[j][q], pool[q].bytes);
        }
        for (int j = 0; j < NS; ++j)
            for (int q = 0; q < NP; ++q) {
                expect(same(r[0][j][q], r[1][j][q]), "K/V one launch: the cell array is not the steps", j);
                expect(same(r[2][j][q], r[3][j][q]), "K/V per row: the cell array is not the steps", j);
                expect(same(r[0][j][q], r[2][j][q]), "K/V: one launch is not the per-row appends", j);
                if (!kPadOf[j]) continue;
                expect(history_kept(j, q, r[0][j][q], held[j]), "K/V: a pad row wrote below its slot's history", j);
                expect(same(r[0][j][q], one[j][q]), "K/V: a pad row wrote more than its one cell past the history", j);
            }
        std::printf("  K/V %s: %d windows, pads wrote only the cell past their history\n", kFmt[fmt], W);
    }
}

// ---- the indexer: position -1 appends nothing (one launch with its snapshot, and per row)
void indexer() {
    void* cs = P::stream();
    const k::QsaShapes s = k::qsa_real_shapes();
    const int ID = (int) s.idx_dim, R4 = (int) s.idx_block, cells = 64;
    const size_t TS = (size_t) (R4 - 1) * ID, PR = (size_t) (cells / R4 + 1) * ID;
    const int held[NS] = {6, 9, 0, 13};   // slot 0 crosses a block end (cell 7), slot 2 starts at cell 0
    std::vector<float> t0[NS], d0[NS], p0[NS];
    for (int j = 0; j < NS; ++j) { t0[j] = gauss(TS, 1.f); d0[j] = gauss((size_t) ID, 1.f); p0[j] = gauss(PR, 1.f); }
    const std::vector<int32_t> b0{4};
    std::vector<const float*> raw(W);
    std::vector<int32_t*> ipos(W), steps(W);
    for (int w = 0; w < W; ++w) {
        raw[w] = dput(gauss((size_t) NS * ID, 2.f));
        std::vector<int32_t> a(NS), st((size_t) NS * k::kStepCount);
        for (int j = 0; j < NS; ++j) {
            a[(size_t) j] = kPadOf[j] ? -1 : held[j] + w;
            k::qsa_step_fill(st.data() + (size_t) j * k::kStepCount, held[j] + w, s);
        }
        ipos[w] = dput(a);
        steps[w] = dput(st);
    }
    const float* gamma = dput(gauss((size_t) ID, 1.f));
    const k::RopeScaling sc;
    struct Idx { float *snap, *tail, *dead, *pooled; int32_t* bpos; };
    const auto fresh = [&](Idx (&x)[NS]) {
        for (int j = 0; j < NS; ++j)
            x[j] = {dalloc<float>(TS), dput(t0[j]), dput(d0[j]), dput(p0[j]), dput(b0)};
    };
    const auto kept = [&](const Idx& x, int j) {
        return same(fetch(x.tail, TS), t0[j]) && same(fetch(x.dead, (size_t) ID), d0[j]) &&
               same(fetch(x.pooled, PR), p0[j]) && same(fetch(x.bpos, 1), b0);
    };
    const auto equal = [&](const Idx& x, const Idx& y) {
        return same(fetch(x.tail, TS), fetch(y.tail, TS)) && same(fetch(x.dead, (size_t) ID), fetch(y.dead, (size_t) ID)) &&
               same(fetch(x.pooled, PR), fetch(y.pooled, PR)) && same(fetch(x.bpos, 1), fetch(y.bpos, 1));
    };
    Idx a[NS], b[NS], c[NS];
    fresh(a); fresh(b); fresh(c);
    for (int w = 0; w < W; ++w) {
        k::QsaIndexerSlots t;
        for (int j = 0; j < NS; ++j) {
            const int g = t.n++;
            t.snap[g] = a[j].snap; t.tail[g] = a[j].tail; t.dead[g] = a[j].dead; t.pooled[g] = a[j].pooled;
            t.block_pos[g] = a[j].bpos; t.first[g] = j; t.ntok[g] = 1; t.max_cells[g] = cells;
        }
        k::native_qsa_indexer_append_slots(raw[w], ipos[w], 1, 0, gamma, k::qsa_rms_eps(), t, s, sc, cs);
        for (int j = 0; j < NS; ++j) {
            // per row from the window's array (pads: -1), and the reference: the active rows' steps only
            const k::QsaIndexerBuffers ib{b[j].tail, b[j].dead, b[j].pooled, b[j].bpos};
            k::native_qsa_indexer_append(raw[w] + (size_t) j * ID, ipos[w] + j, 0, gamma, k::qsa_rms_eps(), ib, s, cells,
                                         sc, cs);
            if (kPadOf[j]) continue;
            const k::QsaIndexerBuffers ic{c[j].tail, c[j].dead, c[j].pooled, c[j].bpos};
            k::native_qsa_indexer_append(raw[w] + (size_t) j * ID, steps[w] + (size_t) j * k::kStepCount + k::kStepPos, 0,
                                         gamma, k::qsa_rms_eps(), ic, s, cells, sc, cs);
        }
    }
    P::sync();
    for (int j = 0; j < NS; ++j) {
        if (kPadOf[j]) {
            expect(kept(a[j], j), "indexer (one launch): a pad row changed its slot's tail, dead key, pooled keys or block", j);
            expect(same(fetch(a[j].snap, TS), t0[j]), "indexer: a pad slot's snapshot (its commit's restore) is not its tail", j);
            expect(kept(b[j], j), "indexer (per row): a pad row changed its slot's indexer", j);
            continue;
        }
        expect(equal(a[j], c[j]), "indexer (one launch): an active slot differs from its steps' appends", j);
        expect(equal(b[j], c[j]), "indexer (per row): an active slot differs from its steps' appends", j);
        expect(!kept(c[j], j), "indexer: the active appends wrote nothing (the check is blind)", j);
    }
    std::printf("  indexer: %d windows, pads kept tail, dead key, pooled keys and block position\n", W);
}
}  // namespace

int main() try {
    std::printf("pad_rows_parity: %d slots, slots 1 and 3 idle (pad rows), %d windows\n", NS, W);
    gdn();
    ple();
    kv();
    indexer();
    std::printf("pad_rows_parity: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "pad_rows_parity: %s\n", e.what());
    return 1;
}
