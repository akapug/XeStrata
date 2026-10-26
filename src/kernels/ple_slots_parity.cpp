// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/ple_slots_parity.cpp - homelab STRATA_BATCH_PLE_SLOTS: native_ple_postops_slots against the per-row loop
// it replaces in a batch window (Verifier::record_window, layer 1): for each row, ple_block with the batched key and
// value projections (pre_key / pre_value) and --native-ple-postops, ple_history_advance with the row's keep flag, then
// a copy of the history (the commit's snapshot).  Compared bitwise (memcmp): every row's result (the residual streams),
// every slot's history after the window and every row's snapshot.
//
// Layouts: one row a slot; slots of 2 and 3 rows (--batch-mtp's contiguous rows); pad rows (keep 0) alone and inside a
// run; 1..32 rows.  Synthetic weights and inputs at the model's geometry (2560 x 4 streams, 9 history rows).  Then the
// time of both paths at 16 and 32 one-row slots (eager launches; the engine replays them in a graph).
//
//     build/ple_slots_parity [reps 5]
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

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

constexpr int N = k::NG_N_EMBD, D = k::NG_HC_DIM, HS = k::NG_HIST * k::NG_HC_DIM, R = k::PleSlots::kMax;
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
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * 4);
}
std::vector<float> gauss(size_t n, float mean, float scale) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = mean + nd(rng) * scale;
    return v;
}

// a window: its rows' slots (runs of equal ids are one slot's contiguous rows) and keep flags
struct Layout {
    std::string name;
    std::vector<int> slot;
    std::vector<int32_t> keep;
};

std::vector<Layout> layouts() {
    std::vector<Layout> out;
    for (int T : {1, 2, 7, 16, 17, 24, 32}) {
        Layout one{"one row a slot, " + std::to_string(T) + " rows", {}, {}};
        for (int t = 0; t < T; ++t) { one.slot.push_back(t); one.keep.push_back(1); }
        out.push_back(one);
        Layout pads{"pad rows, " + std::to_string(T) + " rows", one.slot, {}};
        for (int t = 0; t < T; ++t) pads.keep.push_back(t % 3 == 1 ? 0 : 1);
        out.push_back(pads);
    }
    // runs of 2 and 3 rows of one slot, a pad inside a run, a run at the end; 32 rows over 13 slots
    Layout runs{"runs of 1-3 rows, a pad inside a run, 32 rows", {}, {}};
    const int lens[] = {2, 1, 3, 3, 1, 2, 4, 1, 3, 2, 3, 5, 2};
    for (int j = 0, t = 0; j < 13; ++j)
        for (int i = 0; i < lens[j]; ++i, ++t) {
            runs.slot.push_back(j);
            runs.keep.push_back(j == 3 && i == 1 ? 0 : (j == 8 && i == 0 ? 0 : 1));
        }
    out.push_back(runs);
    // one slot's rows (a solo window's shape): native_ple_postops_batch_snap is checked against the loop too
    for (int T : {5, 12, 32}) {
        Layout one{"one slot, " + std::to_string(T) + " rows", std::vector<int>((size_t) T, 0),
                   std::vector<int32_t>((size_t) T, 1)};
        out.push_back(one);
    }
    return out;
}

struct Fixture {
    k::PleWeights w;
    float *key0, *hidden0, *value, *hist0;   // the window's inputs (key0, hidden0: copied before each run)
    float *key, *hidden, *hist, *snap;       // what a run writes
    float *qn, *gated, *gate, *norm, *emb;
    void* scratch;
    int32_t* keep;
};

// the per-row loop of Verifier::record_window (layer 1 of a batch window, the batched projections, native post-ops)
void per_row(const Fixture& f, const Layout& L, void* s, std::vector<float>& snaps) {
    const int T = (int) L.slot.size();
    for (int t = 0; t < T; ++t) {
        k::PleWeights pw = f.w;
        pw.pre_key = f.key + (size_t) t * D;
        pw.pre_value = f.value + (size_t) t * N;
        k::PleOut po;
        po.normalized = f.norm;
        po.result = f.hidden + (size_t) t * D;
        float* hist = f.hist + (size_t) L.slot[(size_t) t] * HS;
        k::ple_block(f.emb, f.hidden + (size_t) t * D, hist, pw, po, f.scratch, s);
        k::ple_history_advance(hist, f.norm, s, f.keep + t);
        const auto h = fetch(hist, HS);   // the snapshot copy after the row
        std::memcpy(snaps.data() + (size_t) t * HS, h.data(), (size_t) HS * 4);
    }
}

k::PleSlots table(const Fixture& f, const Layout& L) {
    k::PleSlots tab;
    const int T = (int) L.slot.size();
    for (int t = 0; t < T;) {
        const int first = t;
        while (t < T && L.slot[(size_t) t] == L.slot[(size_t) first]) ++t;
        tab.hist[tab.n] = f.hist + (size_t) L.slot[(size_t) first] * HS;
        tab.first[tab.n] = first;
        tab.ntok[tab.n] = t - first;
        ++tab.n;
    }
    return tab;
}

void slots(const Fixture& f, const Layout& L, void* s) {
    const int T = (int) L.slot.size();
    k::native_ple_postops_slots(f.key, f.hidden, f.value, table(f, L), f.keep, f.w, f.qn, f.gated, f.gate, T, f.snap,
                                s);
}

void reset(const Fixture& f, int T, int slots) {
    P::sync();
    P::copy_bytes_in(f.key, fetch(f.key0, (size_t) T * D).data(), (size_t) T * D * 4);
    P::copy_bytes_in(f.hidden, fetch(f.hidden0, (size_t) T * D).data(), (size_t) T * D * 4);
    P::copy_bytes_in(f.hist, fetch(f.hist0, (size_t) slots * HS).data(), (size_t) slots * HS * 4);
}

}  // namespace

int main(int argc, char** argv) try {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    void* s = P::stream();
    k::ple_set_native_postops(true);
    Fixture f{};
    std::vector<uint16_t> taps((size_t) 4 * D);
    for (auto& v : taps) v = k::f16_from_f32(std::normal_distribution<float>(0.f, 0.5f)(rng));
    f.w.norm_key = dput(gauss(D, 1.f, 0.1f));
    f.w.norm_query = dput(gauss(D, 1.f, 0.1f));
    f.w.norm_conv = dput(gauss(D, 1.f, 0.1f));
    f.w.conv1d_f16 = dput(taps);
    f.w.value_bf16 = dalloc<uint16_t>(1);   // not read: the rows' values come projected (pre_value)
    f.key0 = dput(gauss((size_t) R * D, 0.f, 1.f));
    f.hidden0 = dput(gauss((size_t) R * D, 0.f, 1.f));
    f.value = dput(gauss((size_t) R * N, 0.f, 1.f));
    f.hist0 = dput(gauss((size_t) R * HS, 0.f, 1.f));
    f.key = dalloc<float>((size_t) R * D);
    f.hidden = dalloc<float>((size_t) R * D);
    f.hist = dalloc<float>((size_t) R * HS);
    f.snap = dalloc<float>((size_t) R * HS);
    f.qn = dalloc<float>((size_t) R * D);
    f.gated = dalloc<float>((size_t) R * D);
    f.gate = dalloc<float>((size_t) R * 4);
    f.norm = dalloc<float>(D);
    f.emb = dalloc<float>(N);
    f.scratch = P::alloc_bytes((size_t) k::ple_block_scratch_bytes() + 64);
    f.keep = dalloc<int32_t>(R);

    int checks = 0, failures = 0;
    for (const Layout& L : layouts()) {
        const int T = (int) L.slot.size(), S = L.slot.back() + 1;
        put(f.keep, L.keep);
        reset(f, T, S);
        std::vector<float> snap_ref((size_t) T * HS);
        per_row(f, L, s, snap_ref);
        P::sync();
        const auto out_ref = fetch(f.hidden, (size_t) T * D), hist_ref = fetch(f.hist, (size_t) S * HS);
        reset(f, T, S);
        slots(f, L, s);
        P::sync();
        const auto out = fetch(f.hidden, (size_t) T * D), hist = fetch(f.hist, (size_t) S * HS),
                   snap = fetch(f.snap, (size_t) T * HS);
        const bool same_out = same(out, out_ref), same_hist = same(hist, hist_ref), same_snap = same(snap, snap_ref);
        // the window did something: a kept row moved its slot's history, every row's result left its input
        const auto hist0 = fetch(f.hist0, (size_t) S * HS), hidden0 = fetch(f.hidden0, (size_t) T * D);
        bool moved = std::memcmp(out.data(), hidden0.data(), out.size() * 4) != 0, any_keep = false;
        for (int32_t kf : L.keep) any_keep = any_keep || kf != 0;
        if (any_keep) moved = moved && std::memcmp(hist.data(), hist0.data(), hist.size() * 4) != 0;
        checks += 4;
        const int bad = !same_out + !same_hist + !same_snap + !moved;
        failures += bad;
        std::printf("%-48s results %s, histories %s, snapshots %s, %s\n", L.name.c_str(), same_out ? "same" : "DIFFER",
                    same_hist ? "same" : "DIFFER", same_snap ? "same" : "DIFFER", moved ? "moved" : "DID NOT MOVE");
        if (S == 1) {   // the solo window's batch form (STRATA_DF_PLE) over the same rows
            reset(f, T, S);
            k::native_ple_postops_batch_snap(f.key, f.hidden, f.value, f.hist, f.w, f.qn, f.gated, f.gate, T, f.snap,
                                             s);
            P::sync();
            const bool b_out = same(fetch(f.hidden, (size_t) T * D), out_ref);
            const bool b_hist = same(fetch(f.hist, HS), hist_ref);
            const bool b_snap = same(fetch(f.snap, (size_t) T * HS), snap_ref);
            checks += 3;
            failures += !b_out + !b_hist + !b_snap;
            std::printf("%-48s results %s, histories %s, snapshots %s\n", "  native_ple_postops_batch_snap",
                        b_out ? "same" : "DIFFER", b_hist ? "same" : "DIFFER", b_snap ? "same" : "DIFFER");
        }
    }

    for (int T : {16, 32}) {
        Layout L{"", {}, {}};
        for (int t = 0; t < T; ++t) { L.slot.push_back(t); L.keep.push_back(1); }
        put(f.keep, L.keep);
        double us[2] = {};
        for (int path = 0; path < 2; ++path) {
            for (int i = 0; i <= reps; ++i) {
                reset(f, T, T);
                const auto t0 = std::chrono::steady_clock::now();
                if (path == 0) {   // the loop's launches, the snapshot a device copy (the engine's copy_from_mapped)
                    for (int t = 0; t < T; ++t) {
                        k::PleWeights pw = f.w;
                        pw.pre_key = f.key + (size_t) t * D;
                        pw.pre_value = f.value + (size_t) t * N;
                        k::PleOut po;
                        po.normalized = f.norm;
                        po.result = f.hidden + (size_t) t * D;
                        float* hist = f.hist + (size_t) t * HS;
                        k::ple_block(f.emb, f.hidden + (size_t) t * D, hist, pw, po, f.scratch, s);
                        k::ple_history_advance(hist, f.norm, s, f.keep + t);
                        strata::core::Runtime::get().stream(s).memcpy(f.snap + (size_t) t * HS, hist, (size_t) HS * 4);
                    }
                } else {
                    slots(f, L, s);
                }
                P::sync();
                const auto t1 = std::chrono::steady_clock::now();
                const double d = std::chrono::duration<double, std::micro>(t1 - t0).count();
                if (i > 0) us[path] += d / reps;   // call 0: the warm-up (kernel builds)
            }
        }
        std::printf("%d rows, eager, mean of %d: per-row loop %.1f us, slots %.1f us\n", T, reps, us[0], us[1]);
    }
    std::printf("ple_slots_parity: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "ple_slots_parity: %s\n", e.what());
    return 1;
}
