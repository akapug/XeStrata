// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/kv_stream_slots_parity.cpp - STRATA_BATCH_QSA_STREAMED: a batch window's streamed rows in one launch
// (the `_slots` K/V appends with each row's host copy, kv_stream_resolve_slots) against the per-row calls the engine
// makes without it (kv_append_*_step with the host copy, kv_stream_resolve one row at a time), bitwise.
//
// Each of R rows has two copies of a streamed layer (host copy, a VRAM slot pool small enough to evict constantly,
// its own residency map), A for the per-row calls and B for the window calls.  Rows start at different lengths;
// every round appends one cell to every row and, every few rounds, resolves a selection shaped like the indexer's per
// row.  The resolve's miss list is filled by an atomic counter, so which slot a missed block lands in (and from there
// the later evictions) differs run to run in the per-row calls themselves; what must hold after every resolve, in
// both copies, is the residency contract: every selected block resident, page table and slot_block inverse to each
// other, each resident slot's bytes its block's host bytes, no overflow, the same lookups.  The appends are
// deterministic: the host copies must match bitwise at the end.  INT8, FP16, Q4_0 and K8V4, at R = 1, 7 and 16 rows.
//
//     build/kv_stream_slots_parity [rounds]
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace k = strata::kernels;
namespace P = strata::parity;

namespace {
int g_fail = 0, g_checks = 0;

template <typename T> T* dalloc(size_t n) {
    T* p = static_cast<T*>(P::alloc_bytes(n * sizeof(T) + 64));
    std::vector<unsigned char> z(n * sizeof(T) + 64, 0);
    P::copy_bytes_in(p, z.data(), z.size());
    return p;
}
template <typename T> T* halloc(size_t n) {   // host USM: kernels address it directly
    void* h = P::alloc_host_bytes(n * sizeof(T) + 64);
    std::memset(h, 0, n * sizeof(T) + 64);
    return (T*) h;
}

// One pool set: the per-format arrays and their byte sizes, so the copies compare array by array.
struct Pools {
    k::KvHostPools p;
    std::vector<std::pair<void*, size_t>> arrays;
    bool host = false;
    template <typename T> T* take(size_t n) {
        T* x = host ? halloc<T>(n) : dalloc<T>(n);
        arrays.emplace_back(x, n * sizeof(T));
        return x;
    }
    void alloc(int64_t pages, const k::QsaShapes& s, int fmt, bool on_host) {
        host = on_host;
        const size_t rows = (size_t) pages * s.n_head_kv * s.page_size, q4 = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
        if (fmt == k::kKvHybrid) {
            p.k_q = take<int8_t>(rows * s.head_dim); p.k_scale = take<uint16_t>(rows * 4); p.v_q4 = take<uint8_t>(q4);
        } else if (fmt == k::kKvQ4) {
            p.k_q4 = take<uint8_t>(q4); p.v_q4 = take<uint8_t>(q4);
        } else if (fmt == k::kKvInt8) {
            p.k_q = take<int8_t>(rows * s.head_dim); p.v_q = take<int8_t>(rows * s.head_dim);
            p.k_scale = take<uint16_t>(rows * 4); p.v_scale = take<uint16_t>(rows * 4);
        } else {
            p.k_pool = take<uint16_t>(rows * s.head_dim); p.v_pool = take<uint16_t>(rows * s.head_dim);
        }
    }
    k::QsaAttnPools attn(const int32_t* table) const {
        k::QsaAttnPools a;
        a.k_pool = p.k_pool; a.v_pool = p.v_pool; a.k_q = p.k_q; a.v_q = p.v_q; a.k_scale = p.k_scale;
        a.v_scale = p.v_scale; a.k_q4 = p.k_q4; a.v_q4 = p.v_q4; a.page_table = table;
        return a;
    }
    std::vector<unsigned char> bytes(size_t i) const {
        std::vector<unsigned char> b(arrays[i].second);
        if (host) std::memcpy(b.data(), arrays[i].first, b.size());
        else P::copy_bytes_out(b.data(), arrays[i].first, b.size());
        return b;
    }
};

struct Layer {   // one row's streamed layer
    Pools slots, host;
    k::KvStreamMap m;
    void alloc(int64_t n_blocks, int64_t n_slots, const k::QsaShapes& s, int fmt) {
        slots.alloc(n_slots, s, fmt, false);
        host.alloc(n_blocks, s, fmt, true);
        m.page_table = dalloc<int32_t>(n_blocks);
        m.slot_block = dalloc<int32_t>(n_slots); m.slot_stamp = dalloc<int32_t>(n_slots); m.slot_ref = dalloc<int32_t>(n_slots);
        m.miss_block = dalloc<int32_t>(n_slots); m.miss_slot = dalloc<int32_t>(n_slots);
        m.ctl = dalloc<int32_t>(k::kKvCtlInts);
        m.n_blocks = n_blocks; m.n_slots = n_slots;
        k::kv_stream_reset(m, nullptr);
    }
};

std::vector<int32_t> down(const int32_t* d, size_t n) {
    std::vector<int32_t> h(n);
    P::copy_bytes_out(h.data(), d, n * 4);
    return h;
}

// the engine's one-row append (layer.cpp / verify.cpp's per-row loop), host copy included
void append_row(Layer& L, const int32_t* step, const float* kc, const float* vc, const k::QsaShapes& s, int fmt) {
    const k::KvHostPools& h = L.host.p;
    const k::KvHostPools& v = L.slots.p;
    const int32_t* tab = L.m.page_table;
    if (fmt == k::kKvHybrid) {
        const k::KvHostPools hk = k::kv_hybrid_k_half(h), hv = k::kv_hybrid_v_half(h);
        k::kv_append_q8_step(v.k_q, v.k_q, v.k_scale, v.k_scale, tab, step, kc, kc, s, nullptr, &hk);
        k::kv_append_q4_step(v.v_q4, v.v_q4, tab, step, vc, vc, s, nullptr, &hv);
    } else if (fmt == k::kKvQ4) k::kv_append_q4_step(v.k_q4, v.v_q4, tab, step, kc, vc, s, nullptr, &h);
    else if (fmt == k::kKvInt8) k::kv_append_q8_step(v.k_q, v.v_q, v.k_scale, v.v_scale, tab, step, kc, vc, s, nullptr, &h);
    else k::kv_append_step(v.k_pool, v.v_pool, tab, step, kc, vc, s, nullptr, &h);
}

// verify.cpp's window append: the table as the engine fills it, then the format's _slots call(s)
void append_rows(std::vector<Layer>& B, int R, const int32_t* steps, const float* kc, const float* vc, int per,
                 const k::QsaShapes& s, int fmt) {
    k::KvAppendSlots t;
    for (int r = 0; r < R; ++r) {
        const k::KvHostPools& v = B[r].slots.p;
        const k::KvHostPools& h = B[r].host.p;
        t.page_table[r] = B[r].m.page_table;
        if (fmt == k::kKvHybrid) {
            t.k[r] = v.k_q; t.k_scale[r] = v.k_scale; t.v[r] = v.v_q4;
            t.host_k[r] = h.k_q; t.host_k_scale[r] = h.k_scale; t.host_v[r] = h.v_q4;
        } else if (fmt == k::kKvQ4) {
            t.k[r] = v.k_q4; t.v[r] = v.v_q4; t.host_k[r] = h.k_q4; t.host_v[r] = h.v_q4;
        } else if (fmt == k::kKvInt8) {
            t.k[r] = v.k_q; t.v[r] = v.v_q; t.k_scale[r] = v.k_scale; t.v_scale[r] = v.v_scale;
            t.host_k[r] = h.k_q; t.host_v[r] = h.v_q; t.host_k_scale[r] = h.k_scale; t.host_v_scale[r] = h.v_scale;
        } else {
            t.k[r] = v.k_pool; t.v[r] = v.v_pool; t.host_k[r] = h.k_pool; t.host_v[r] = h.v_pool;
        }
    }
    if (fmt == k::kKvHybrid) {
        k::kv_append_q8_slots(t, steps, k::kStepCount, kc, kc, per, R, 1, s, nullptr);
        k::KvAppendSlots vt;
        for (int r = 0; r < R; ++r) { vt.k[r] = t.v[r]; vt.page_table[r] = t.page_table[r]; vt.host_k[r] = t.host_v[r]; }
        k::kv_append_q4_slots(vt, steps, k::kStepCount, R, vc, vc, 1, s, nullptr);
    } else if (fmt == k::kKvQ4) k::kv_append_q4_slots(t, steps, k::kStepCount, R, kc, vc, 2, s, nullptr);
    else if (fmt == k::kKvInt8) k::kv_append_q8_slots(t, steps, k::kStepCount, kc, vc, per, R, 2, s, nullptr);
    else k::kv_append_slots(t, steps, k::kStepCount, kc, vc, per, R, s, nullptr);
}

// A selection like qsa_block_topk's: ascending cells, whole blocks, one partial block, the tail block's cells.
std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t c = 0; c < n_kv; ++c) ids.push_back((int32_t) c);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1), recent(std::max<int64_t>(0, n_bid - 300), n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(rng() % 2 ? recent(rng) : any(rng));
    const int64_t partial = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks)
        for (int64_t i = 0; i < (b == partial ? part : 4); ++i) ids.push_back((int32_t) (b * 4 + i));
    for (int64_t c = n_bid * 4; c < n_kv; ++c) ids.push_back((int32_t) c);
    std::sort(ids.begin(), ids.end());
    return ids;
}

// the residency contract of one row's layer after a resolve of `sel`
bool resident_ok(const Layer& L, const std::vector<int32_t>& sel, int64_t page_size) {
    const auto pt = down(L.m.page_table, L.m.n_blocks), sb = down(L.m.slot_block, L.m.n_slots);
    const auto ctl = down(L.m.ctl, k::kKvCtlInts);
    if (ctl[3] != 0) return false;
    for (const int32_t c : sel)
        if (pt[c / page_size] < 0) return false;
    for (int64_t b = 0; b < L.m.n_blocks; ++b)
        if (pt[b] >= 0 && (pt[b] >= L.m.n_slots || sb[pt[b]] != b)) return false;
    for (int64_t s = 0; s < L.m.n_slots; ++s)
        if (sb[s] >= 0 && pt[sb[s]] != s) return false;
    for (size_t i = 0; i < L.slots.arrays.size(); ++i) {
        const auto v = L.slots.bytes(i);
        const size_t len = L.slots.arrays[i].second / (size_t) L.m.n_slots;
        const auto* h = static_cast<const unsigned char*>(L.host.arrays[i].first);
        for (int64_t s = 0; s < L.m.n_slots; ++s)
            if (sb[s] >= 0 && std::memcmp(v.data() + (size_t) s * len, h + (size_t) sb[s] * len, len) != 0) return false;
    }
    return true;
}

uint64_t lookups(const Layer& L) { return k::kv_stream_counters(L.m).lookups; }

void check(bool ok, const char* what, const char* fmt, int R, int round, int row) {
    ++g_checks;
    if (ok) return;
    ++g_fail;
    if (g_fail <= 20) std::printf("FAIL %s fmt %s rows %d round %d row %d\n", what, fmt, R, round, row);
}

void run(int fmt, int R, int rounds) {
    const char* name = fmt == k::kKvHybrid ? "k8v4" : fmt == k::kKvQ4 ? "q4_0" : fmt == k::kKvInt8 ? "int8" : "fp16";
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    // rows longer than the slot pool holds, so the selections miss and evict
    const int64_t start0 = 4 * 1400, start_max = start0 + 37 * k::kQsaSlotsMax, N = start_max + rounds + 8;
    const int64_t n_blocks = (N + 3) / 4, n_slots = std::max<int64_t>(1024, cap / s.page_size + 2) + 64;   // evicts
    const int per = (int) (s.n_head_kv * s.head_dim);
    std::mt19937 rng(11 + 5 * fmt + R);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<Layer> A(R), B(R);
    for (int r = 0; r < R; ++r) { A[r].alloc(n_blocks, n_slots, s, fmt); B[r].alloc(n_blocks, n_slots, s, fmt); }
    std::vector<int64_t> len(R);
    for (int r = 0; r < R; ++r) len[r] = start0 + 37 * r;
    int32_t* d_steps = dalloc<int32_t>((size_t) R * k::kStepCount);
    int32_t* d_ids = dalloc<int32_t>((size_t) R * cap);
    float* d_k = dalloc<float>((size_t) R * per);
    float* d_v = dalloc<float>((size_t) R * per);
    std::vector<int32_t> steps((size_t) R * k::kStepCount), ids((size_t) R * cap);
    std::vector<float> kv((size_t) R * per);
    const auto put_kv = [&] {
        for (auto& x : kv) x = nd(rng);
        P::copy_bytes_in(d_k, kv.data(), kv.size() * 4);
        for (auto& x : kv) x = nd(rng);
        P::copy_bytes_in(d_v, kv.data(), kv.size() * 4);
    };
    // the rows' prefixes: nothing is resident yet, so an append writes only the host copy; the same random bytes into
    // both copies' host arrays stand in for len[r] appended cells (the comparison is of bytes, not values)
    for (int r = 0; r < R; ++r)
        for (size_t i = 0; i < A[r].host.arrays.size(); ++i) {
            auto* a = static_cast<unsigned char*>(A[r].host.arrays[i].first);
            for (size_t j = 0; j < A[r].host.arrays[i].second; ++j) a[j] = (unsigned char) rng();
            std::memcpy(B[r].host.arrays[i].first, a, A[r].host.arrays[i].second);
        }
    double us_rows = 0, us_slots = 0;
    int resolves = 0;
    for (int round = 0; round < rounds; ++round) {
        put_kv();
        for (int r = 0; r < R; ++r) {
            steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) len[r];
            steps[(size_t) r * k::kStepCount + k::kStepWidth] = 0;
        }
        P::copy_bytes_in(d_steps, steps.data(), steps.size() * 4);
        for (int r = 0; r < R; ++r)
            append_row(A[r], d_steps + (size_t) r * k::kStepCount, d_k + (size_t) r * per, d_v + (size_t) r * per, s, fmt);
        append_rows(B, R, d_steps, d_k, d_v, per, s, fmt);
        for (int r = 0; r < R; ++r) ++len[r];
        if (round % 3 != 2) continue;
        std::vector<std::vector<int32_t>> sels(R);
        for (int r = 0; r < R; ++r) {
            const auto& sel = sels[r] = selection(len[r], std::min<int64_t>(cap, k::qsa_selection_width(len[r], s)), rng);
            std::copy(sel.begin(), sel.end(), ids.begin() + (ptrdiff_t) r * cap);
            steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) sel.size();
        }
        P::copy_bytes_in(d_steps, steps.data(), steps.size() * 4);
        P::copy_bytes_in(d_ids, ids.data(), ids.size() * 4);
        P::sync();
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < R; ++r)
            k::kv_stream_resolve(A[r].m, A[r].slots.attn(A[r].m.page_table), A[r].host.p, fmt, d_ids + (size_t) r * cap,
                                 d_steps + (size_t) r * k::kStepCount, 1, cap, s, P::stream());
        P::sync();
        auto t1 = std::chrono::steady_clock::now();
        k::KvResolveSlots tab;
        for (int r = 0; r < R; ++r) { tab.map[r] = B[r].m; tab.slots[r] = B[r].slots.attn(B[r].m.page_table); tab.host[r] = B[r].host.p; }
        k::kv_stream_resolve_slots(tab, fmt, d_ids, d_steps, R, cap, s, P::stream());
        P::sync();
        auto t2 = std::chrono::steady_clock::now();
        us_rows += std::chrono::duration<double, std::micro>(t1 - t0).count();
        us_slots += std::chrono::duration<double, std::micro>(t2 - t1).count();
        ++resolves;
        for (int r = 0; r < R; ++r) {
            check(resident_ok(A[r], sels[r], s.page_size), "per-row residency", name, R, round, r);
            check(resident_ok(B[r], sels[r], s.page_size), "slots residency", name, R, round, r);
            check(lookups(A[r]) == lookups(B[r]), "lookups", name, R, round, r);
        }
    }
    P::sync();
    int misses = 0;
    for (int r = 0; r < R; ++r) {
        for (size_t i = 0; i < A[r].host.arrays.size(); ++i)
            check(A[r].host.bytes(i) == B[r].host.bytes(i), "host copy", name, R, rounds, r);
        misses += (int) k::kv_stream_counters(A[r].m).misses;
    }
    std::printf("%-5s rows %2d: %d resolves, %d misses; resolve per window %8.1f us per-row calls, %8.1f us slots\n", name,
                R, resolves, misses, us_rows / std::max(resolves, 1), us_slots / std::max(resolves, 1));
}
}  // namespace

int main(int argc, char** argv) try {
    const int rounds = argc > 1 ? std::atoi(argv[1]) : 90;
    for (int fmt : {k::kKvInt8, k::kKvF16, k::kKvQ4, k::kKvHybrid})
        for (int R : {1, 7, k::kQsaSlotsMax}) run(fmt, R, rounds);
    std::printf("kv_stream_slots_parity: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "kv_stream_slots_parity: %s\n", e.what());
    return 1;
}
