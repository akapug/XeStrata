// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/qsa_slots_parity.cpp - the batch window's one-launch QSA rows (STRATA_BATCH_QSA_ROWS) against the
// per-row calls they replace, memcmp on every output byte: the K/V appends (fp16, INT8, Q4_0, K8V4), the indexer tail
// snapshot and appends, the block scores and the decode attention (all four formats).
//
// Six rows over five slots (rows 1 and 2 are one slot's consecutive positions, as --batch-mtp's), each slot with its
// own pools, non-identity page table and indexer state.  Both homogeneous and mixed context-sized allocations;
// the smaller slot's proposal pair ends at its context boundary.  SYCL device, synthetic, no model.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"

#include "parity_device.hpp"
#include "xe/fused_gr_win.hpp"
#include "xe/param_table.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0, g_checks = 0;
std::mt19937 rng(353);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.

template <typename T> T* dalloc(size_t n) {
    T* p = static_cast<T*>(strata::parity::alloc_bytes(n * sizeof(T) + 64));
    const std::vector<unsigned char> z(n * sizeof(T) + 64, 0);
    strata::parity::copy_bytes_in(p, z.data(), z.size());
    return p;
}
template <typename T> T* dput(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    strata::parity::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}
template <typename T> std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n);
    strata::parity::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}
template <typename T> void check(const char* what, const T* a, const T* b, size_t n) {
    ++g_checks;
    if (std::memcmp(fetch(a, n).data(), fetch(b, n).data(), n * sizeof(T)) == 0) return;
    std::printf("    *** %s: the one-launch rows differ from the per-row calls ***\n", what);
    ++g_fail;
}
std::vector<float> gauss(size_t n, float scale) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng) * scale;
    return v;
}
template <typename T> T* twin(T* a, size_t n) {   // a second buffer with a's bytes
    T* b = dalloc<T>(n);
    strata::parity::copy_bytes_in(b, fetch(a, n).data(), n * sizeof(T));
    return b;
}
}  // namespace

void parity(bool mixed) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, NH = (int) s.n_head;
    const int ID = (int) s.idx_dim, IQ = (int) s.idx_n_head, R4 = (int) s.idx_block;
    const int cells = 512, HD = H * D, G8 = D / k::KV_Q8_GROUP;
    constexpr int NS = 5, NR = 6;
    const int slot_of[NR] = {0, 1, 1, 2, 3, 4};
    const int capacity[NS] = {cells, mixed ? cells / 2 : cells, mixed ? P : cells, cells, mixed ? P : cells};
    const int pos_of[NR] = {mixed ? cells - 1 : 7, mixed ? capacity[1] - 2 : 10,
                            mixed ? capacity[1] - 1 : 11, 0, 413, 3};
    std::printf("qsa_slots: %s contexts\n",
                mixed ? "mixed" : "homogeneous");
    void* cs = strata::parity::stream();

    std::vector<int32_t> hstep((size_t) NR * k::kStepCount);
    for (int r = 0; r < NR; ++r) k::qsa_step_fill(hstep.data() + (size_t) r * k::kStepCount, pos_of[r], s);
    const int32_t* d_step = dput(hstep);
    const float* d_k = dput(gauss((size_t) NR * HD, 1.f));
    const float* d_v = dput(gauss((size_t) NR * HD, 3.f));

    struct Slot {
        int32_t* table;
        uint16_t *kf, *vf;
        int8_t *kq, *vq;
        uint16_t *ks, *vs;
        uint8_t *k4, *v4;
    } sl[NS];
    size_t nf[NS], ns8[NS];
    for (int j = 0; j < NS; ++j) { nf[j] = (size_t) capacity[j] * HD; ns8[j] = (size_t) capacity[j] * H * G8; }
    // every slot's pools full of cells (each format through its own append), under a permuted page table
    std::vector<int32_t> all_steps((size_t) cells * k::kStepCount);
    for (int c = 0; c < cells; ++c) k::qsa_step_fill(all_steps.data() + (size_t) c * k::kStepCount, c, s);
    const int32_t* d_all = dput(all_steps);
    for (int j = 0; j < NS; ++j) {
        const int pages = capacity[j] / P;
        std::vector<int32_t> t(pages);
        for (int p = 0; p < pages; ++p) t[p] = (p * 37 + 11 + j) % pages;
        Slot& x = sl[j];
        x.table = dput(t);
        x.kf = dalloc<uint16_t>(nf[j]); x.vf = dalloc<uint16_t>(nf[j]);
        x.kq = dalloc<int8_t>(nf[j]); x.vq = dalloc<int8_t>(nf[j]);
        x.ks = dalloc<uint16_t>(ns8[j]); x.vs = dalloc<uint16_t>(ns8[j]);
        x.k4 = dalloc<uint8_t>(nf[j]); x.v4 = dalloc<uint8_t>(nf[j]);
        const float* kc = dput(gauss(nf[j], 1.f));
        const float* vc = dput(gauss(nf[j], 2.f));
        k::kv_append_steps(x.kf, x.vf, x.table, d_all, k::kStepCount, kc, vc, HD, capacity[j], s, cs);
        k::kv_append_q8_steps(x.kq, x.vq, x.ks, x.vs, x.table, d_all, k::kStepCount, kc, vc, HD, capacity[j], s, cs);
        k::kv_append_q4_steps(x.k4, x.v4, x.table, d_all, k::kStepCount, capacity[j], kc, vc, s, cs);
    }
    strata::parity::sync();

    // ---- the K/V appends: each format into two copies of the rows' slots, per row vs one launch; and a third copy (c)
    // through a captured graph, replayed twice: the tables are new when the capture runs (param_table.hpp fills their
    // device copies outside the recording) and the graph keeps reading them
    {
        Slot a[NS], b[NS], c[NS];
        for (int j = 0; j < NS; ++j) {
            a[j] = b[j] = c[j] = sl[j];
            c[j].kf = twin(sl[j].kf, nf[j]); c[j].vf = twin(sl[j].vf, nf[j]); c[j].kq = twin(sl[j].kq, nf[j]);
            c[j].vq = twin(sl[j].vq, nf[j]); c[j].ks = twin(sl[j].ks, ns8[j]); c[j].vs = twin(sl[j].vs, ns8[j]);
            c[j].k4 = twin(sl[j].k4, nf[j]); c[j].v4 = twin(sl[j].v4, nf[j]);
            a[j].kf = twin(sl[j].kf, nf[j]); a[j].vf = twin(sl[j].vf, nf[j]); b[j].kf = twin(sl[j].kf, nf[j]); b[j].vf = twin(sl[j].vf, nf[j]);
            a[j].kq = twin(sl[j].kq, nf[j]); a[j].vq = twin(sl[j].vq, nf[j]); b[j].kq = twin(sl[j].kq, nf[j]); b[j].vq = twin(sl[j].vq, nf[j]);
            a[j].ks = twin(sl[j].ks, ns8[j]); a[j].vs = twin(sl[j].vs, ns8[j]); b[j].ks = twin(sl[j].ks, ns8[j]); b[j].vs = twin(sl[j].vs, ns8[j]);
            a[j].k4 = twin(sl[j].k4, nf[j]); a[j].v4 = twin(sl[j].v4, nf[j]); b[j].k4 = twin(sl[j].k4, nf[j]); b[j].v4 = twin(sl[j].v4, nf[j]);
        }
        k::KvAppendSlots ctabs[4], cvtabs[4];
        const auto launch = [&](int fmt, const k::KvAppendSlots& tab, const k::KvAppendSlots& vtab) {
            if (fmt == 0) k::kv_append_slots(tab, d_step, k::kStepCount, d_k, d_v, HD, NR, s, cs);
            else if (fmt == 1) k::kv_append_q8_slots(tab, d_step, k::kStepCount, d_k, d_v, HD, NR, 2, s, cs);
            else if (fmt == 2) k::kv_append_q4_slots(tab, d_step, k::kStepCount, NR, d_k, d_v, 2, s, cs);
            else {
                k::kv_append_q8_slots(tab, d_step, k::kStepCount, d_k, d_k, HD, NR, 1, s, cs);
                k::kv_append_q4_slots(vtab, d_step, k::kStepCount, NR, d_v, d_v, 1, s, cs);
            }
        };
        for (int fmt = 0; fmt < 4; ++fmt) {
            k::KvAppendSlots tab, vtab;
            k::KvAppendSlots& ctab = ctabs[fmt];
            k::KvAppendSlots& cvtab = cvtabs[fmt];
            for (int r = 0; r < NR; ++r) {
                const Slot& x = a[slot_of[r]];
                const Slot& y = b[slot_of[r]];
                const Slot& z = c[slot_of[r]];
                ctab.page_table[r] = cvtab.page_table[r] = z.table;
                const int32_t* st = d_step + (size_t) r * k::kStepCount;
                const float* kc = d_k + (size_t) r * HD;
                const float* vc = d_v + (size_t) r * HD;
                tab.page_table[r] = vtab.page_table[r] = y.table;
                if (fmt == 0) {
                    k::kv_append_step(x.kf, x.vf, x.table, st, kc, vc, s, cs);
                    tab.k[r] = y.kf; tab.v[r] = y.vf;
                    ctab.k[r] = z.kf; ctab.v[r] = z.vf;
                } else if (fmt == 1) {
                    k::kv_append_q8_step(x.kq, x.vq, x.ks, x.vs, x.table, st, kc, vc, s, cs);
                    tab.k[r] = y.kq; tab.v[r] = y.vq; tab.k_scale[r] = y.ks; tab.v_scale[r] = y.vs;
                    ctab.k[r] = z.kq; ctab.v[r] = z.vq; ctab.k_scale[r] = z.ks; ctab.v_scale[r] = z.vs;
                } else if (fmt == 2) {
                    k::kv_append_q4_step(x.k4, x.v4, x.table, st, kc, vc, s, cs);
                    tab.k[r] = y.k4; tab.v[r] = y.v4;
                    ctab.k[r] = z.k4; ctab.v[r] = z.v4;
                } else {   // K8V4: the folded int8 K and q4_0 V calls (layer.cpp)
                    k::kv_append_q8_step(x.kq, x.kq, x.ks, x.ks, x.table, st, kc, kc, s, cs);
                    k::kv_append_q4_step(x.v4, x.v4, x.table, st, vc, vc, s, cs);
                    tab.k[r] = y.kq; tab.k_scale[r] = y.ks; vtab.k[r] = y.v4;
                    ctab.k[r] = z.kq; ctab.k_scale[r] = z.ks; cvtab.k[r] = z.v4;
                }
            }
            launch(fmt, tab, vtab);
        }
        strata::parity::sync();
        {   // the captured form (the appends write the same cells again on the second replay: the same bytes)
            namespace exp = sycl::ext::oneapi::experimental;
            sycl::queue& q = *static_cast<sycl::queue*>(cs);
            exp::command_graph graph(q.get_context(), q.get_device());
            graph.begin_recording(q);
            for (int fmt = 0; fmt < 4; ++fmt) launch(fmt, ctabs[fmt], cvtabs[fmt]);
            graph.end_recording(q);
            auto exec = graph.finalize();
            q.ext_oneapi_graph(exec);
            q.ext_oneapi_graph(exec);
            strata::parity::sync();
        }
        for (int j = 0; j < NS; ++j) {
            check("graph fp16 K/V", a[j].kf, c[j].kf, nf[j]); check("graph fp16 K/V", a[j].vf, c[j].vf, nf[j]);
            check("graph int8 K/V codes", a[j].kq, c[j].kq, nf[j]); check("graph int8 K/V codes", a[j].vq, c[j].vq, nf[j]);
            check("graph int8 K/V scales", a[j].ks, c[j].ks, ns8[j]); check("graph int8 K/V scales", a[j].vs, c[j].vs, ns8[j]);
            check("graph q4_0 K/V", a[j].k4, c[j].k4, nf[j]); check("graph q4_0 K/V", a[j].v4, c[j].v4, nf[j]);
        }
        for (int j = 0; j < NS; ++j) {
            check("fp16 K/V", a[j].kf, b[j].kf, nf[j]); check("fp16 K/V", a[j].vf, b[j].vf, nf[j]);
            check("int8 K/V codes", a[j].kq, b[j].kq, nf[j]); check("int8 K/V codes", a[j].vq, b[j].vq, nf[j]);
            check("int8 K/V scales", a[j].ks, b[j].ks, ns8[j]); check("int8 K/V scales", a[j].vs, b[j].vs, ns8[j]);
            check("q4_0 K/V", a[j].k4, b[j].k4, nf[j]); check("q4_0 K/V", a[j].v4, b[j].v4, nf[j]);
        }
    }

    // ---- the indexer: each slot's tail snapshot, then its rows' appends in order
    const size_t TS = (size_t) (R4 - 1) * ID;
    size_t PR[NS];
    for (int j = 0; j < NS; ++j) PR[j] = (size_t) (capacity[j] / R4 + 2) * ID;
    float *tail[NS], *dead[NS], *pooled[NS];
    int32_t* bpos[NS];
    for (int j = 0; j < NS; ++j) {
        tail[j] = dput(gauss(TS, 1.f)); dead[j] = dput(gauss((size_t) ID, 1.f)); pooled[j] = dput(gauss(PR[j], 1.f));
        bpos[j] = dalloc<int32_t>(1);
    }
    {
        const float* raw = dput(gauss((size_t) NR * ID, 2.f));
        const float* gamma = dput(gauss((size_t) ID, 1.f));
        k::RopeScaling sc;
        float *ta[NS], *da[NS], *pa[NS], *tb[NS], *db[NS], *pb[NS], *sa[NS], *sb[NS];
        int32_t *ba[NS], *bb[NS];
        k::QsaIndexerSlots itab;
        for (int j = 0; j < NS; ++j) {
            ta[j] = twin(tail[j], TS); da[j] = twin(dead[j], (size_t) ID); pa[j] = twin(pooled[j], PR[j]); ba[j] = twin(bpos[j], 1);
            tb[j] = twin(tail[j], TS); db[j] = twin(dead[j], (size_t) ID); pb[j] = twin(pooled[j], PR[j]); bb[j] = twin(bpos[j], 1);
            sa[j] = dalloc<float>(TS); sb[j] = dalloc<float>(TS);
        }
        for (int r = 0; r < NR;) {
            const int first = r, j = slot_of[r];
            while (r < NR && slot_of[r] == j) ++r;
            k::copy_from_mapped(sa[j], ta[j], (int64_t) TS, cs);
            const k::QsaIndexerBuffers ib{ta[j], da[j], pa[j], ba[j]};
            for (int t = first; t < r; ++t)
                k::native_qsa_indexer_append(raw + (size_t) t * ID, d_step + (size_t) t * k::kStepCount + k::kStepPos, 0,
                                             gamma, k::qsa_rms_eps(), ib, s, capacity[j], sc, cs);
            const int g = itab.n++;
            itab.snap[g] = sb[j]; itab.tail[g] = tb[j]; itab.dead[g] = db[j]; itab.pooled[g] = pb[j];
            itab.block_pos[g] = bb[j]; itab.first[g] = first; itab.ntok[g] = r - first; itab.max_cells[g] = capacity[j];
        }
        k::native_qsa_indexer_append_slots(raw, d_step + k::kStepPos, k::kStepCount, 0, gamma, k::qsa_rms_eps(), itab, s,
                                           sc, cs);
        strata::parity::sync();
        for (int j = 0; j < NS; ++j) {
            check("indexer snapshot", sa[j], sb[j], TS); check("indexer tail", ta[j], tb[j], TS);
            check("indexer dead key", da[j], db[j], (size_t) ID); check("indexer pooled keys", pa[j], pb[j], PR[j]);
            check("indexer block_pos", ba[j], bb[j], 1);
        }
    }

    // ---- the block scores, over each row's own slot's keys
    {
        const int64_t max_blocks = cells / R4 + 2;
        const float* q = dput(gauss((size_t) NR * IQ * ID, 1.f));
        float* sa = dalloc<float>((size_t) NR * max_blocks);
        float* sb = dalloc<float>((size_t) NR * max_blocks);
        k::QsaScoreSlots tab;
        for (int r = 0; r < NR; ++r) {
            const int j = slot_of[r];
            k::qsa_block_scores(pooled[j], dead[j], q + (size_t) r * IQ * ID, d_step + (size_t) r * k::kStepCount, 1,
                                max_blocks, s, sa + (size_t) r * max_blocks, cs);
            tab.pooled[r] = pooled[j];
            tab.dead[r] = dead[j];
        }
        k::qsa_block_scores_slots(tab, q, d_step, NR, max_blocks, s, sb, cs);
        strata::parity::sync();
        check("block scores", sa, sb, (size_t) NR * max_blocks);
    }

    // ---- the attention, each format, over each row's own slot's K/V and its own selection
    {
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
        std::vector<int32_t> ids((size_t) NR * cap, 0);
        for (int r = 0; r < NR; ++r) {
            const int n_kv = hstep[(size_t) r * k::kStepCount + k::kStepNKv];
            std::uniform_int_distribution<int> cell(0, n_kv - 1);
            for (int c = 0; c < hstep[(size_t) r * k::kStepCount + k::kStepWidth]; ++c) ids[(size_t) r * cap + c] = cell(rng);
        }
        const int32_t* d_ids = dput(ids);
        const float* q = dput(gauss((size_t) NR * NH * D, 0.2f));
        const size_t sf = k::qsa_decode_attn_scratch_floats(cap, s);
        float* scr_a = dalloc<float>(NR * sf);
        float* scr_b = dalloc<float>(NR * sf);
        for (int fmt = 0; fmt < 4; ++fmt) {
            float* aa = dalloc<float>((size_t) NR * NH * D);
            float* ab = dalloc<float>((size_t) NR * NH * D);
            k::QsaAttnSlots tab;
            bool set = true;
            for (int r = 0; r < NR; ++r) {
                const Slot& x = sl[slot_of[r]];
                k::QsaAttnPools p;
                p.page_table = x.table;
                if (fmt == 0) { p.k_pool = x.kf; p.v_pool = x.vf; }
                else if (fmt == 1) { p.k_q = x.kq; p.v_q = x.vq; p.k_scale = x.ks; p.v_scale = x.vs; }
                else if (fmt == 2) { p.k_q4 = x.k4; p.v_q4 = x.v4; }
                else { p.k_q = x.kq; p.k_scale = x.ks; p.v_q4 = x.v4; }
                k::qsa_decode_attn_batch(q + (size_t) r * NH * D, p, d_ids + (size_t) r * cap,
                                         d_step + (size_t) r * k::kStepCount, cap, s, scr_a + (size_t) r * sf,
                                         aa + (size_t) r * NH * D, 1, cs);
                set = set && k::qsa_attn_slots_set(tab, r, p);
            }
            ++g_checks;
            if (!set || tab.mode != fmt) {
                std::printf("    *** attention format %d: the table refused a row ***\n", fmt);
                ++g_fail;
                continue;
            }
            k::qsa_decode_attn_slots(q, tab, d_ids, d_step, cap, s, scr_b, ab, NR, cs);
            strata::parity::sync();
            const char* names[4] = {"attention fp16", "attention int8", "attention q4_0", "attention K8V4"};
            check(names[fmt], aa, ab, (size_t) NR * NH * D);
        }
        // a row of another format is refused (the caller keeps the per-row calls)
        k::QsaAttnSlots mixed;
        k::QsaAttnPools pf, pq;
        pf.page_table = pq.page_table = sl[0].table;
        pf.k_pool = sl[0].kf; pf.v_pool = sl[0].vf;
        pq.k_q = sl[0].kq; pq.v_q = sl[0].vq; pq.k_scale = sl[0].ks; pq.v_scale = sl[0].vs;
        ++g_checks;
        if (!k::qsa_attn_slots_set(mixed, 0, pf) || k::qsa_attn_slots_set(mixed, 1, pq)) {
            std::printf("    *** a mixed-format table was not refused ***\n");
            ++g_fail;
        }
    }

}

// The device-table store (param_table.hpp): one copy per distinct table, the same pointer for the same bytes, and
// the copy holds those bytes.
void device_table_check() {
    auto& q = strata::core::Runtime::get().stream(nullptr);
    k::KvAppendSlots a, b;
    a.k[0] = &g_checks;
    b.k[0] = &g_fail;
    const size_t before = k::xe::device_table_count();
    const k::KvAppendSlots* da = k::xe::device_table(q, a);
    const k::KvAppendSlots* da2 = k::xe::device_table(q, a);
    const k::KvAppendSlots* db = k::xe::device_table(q, b);
    k::KvAppendSlots back;
    q.memcpy(&back, da, sizeof(back)).wait();
    ++g_checks;
    if (da != da2 || da == db || k::xe::device_table_count() != before + 2 || std::memcmp(&back, &a, sizeof(a)) != 0) {
        std::printf("    *** device tables: same bytes must share one copy, other bytes get their own ***\n");
        ++g_fail;
    }
    std::printf("  device tables: %zu held\n", k::xe::device_table_count());
}

// Each `_slots` table, and how its kernel takes it: BY VALUE (its bytes are kernel arguments, so they count against
// the device's max_parameter_size) or BY POINTER (a device copy, param_table.hpp: the kernel argument is one pointer).
// KvResolveSlots' kernels take its derived Maps and RowRuns by pointer.
struct TableEntry { const char* name; size_t bytes; bool by_pointer; };
constexpr TableEntry kTables[] = {
    {"KvAppendSlots", sizeof(k::KvAppendSlots), true}, {"QsaIndexerSlots", sizeof(k::QsaIndexerSlots), true},
    {"QsaAttnSlots", sizeof(k::QsaAttnSlots), false}, {"QsaScoreSlots", sizeof(k::QsaScoreSlots), false},
    {"KvResolveSlots", sizeof(k::KvResolveSlots), true}, {"GdnSlotTable", sizeof(k::GdnSlotTable), false},
    {"GrWin", sizeof(k::grw::GrWin), false}
};
// The argument a table occupies: its bytes by value, one device pointer by pointer.
constexpr size_t passed_bytes(const TableEntry& e) { return e.by_pointer ? sizeof(void*) : e.bytes; }
constexpr bool by_value_tables_fit() {
    for (const TableEntry& e : kTables)
        if (passed_bytes(e) >= k::xe::kParamBytesFloor) return false;
    return true;
}
static_assert(by_value_tables_fit(), "a table passed by value is past the B70's 2048 bytes of kernel arguments");

int main(int argc, char** argv) {
    const auto& device = strata::core::Runtime::get().device();
    const size_t limit = device.get_info<sycl::info::device::max_parameter_size>();
    std::printf("slots parameter probe: device=%s max_parameter_size=%zu bytes capacity=%d (compile-time floor %zu)\n",
                device.get_info<sycl::info::device::name>().c_str(), limit, k::kVerifyMaxT, k::xe::kParamBytesFloor);
    for (const TableEntry& e : kTables) {
        const size_t passed = passed_bytes(e);
        std::printf("  %s sizeof=%zu bytes headroom=%lld bytes (%s: passes %zu bytes; other captures still required)\n",
                    e.name, e.bytes, (long long) limit - (long long) passed, e.by_pointer ? "by pointer" : "by value",
                    passed);
        if (passed >= limit) ++g_fail;
    }
    if (g_fail) return 1;
    if (argc == 2 && std::strcmp(argv[1], "--parameter-probe") == 0) return 0;
    if (argc != 1) { std::fputs("usage: qsa_slots_parity [--parameter-probe]\n", stderr); return 2; }
    parity(false);
    parity(true);
    device_table_check();
    // every row-scaled kernel the runs launched, by its functor's size (its kernel arguments, captures included;
    // xe::fits also checks each against the 2048-byte floor at compile time)
    for (const auto& [name, bytes] : k::xe::kernel_arg_bytes()) {
        std::printf("  kernel %s args=%zu bytes headroom=%lld bytes\n", name.c_str(), bytes, (long long) limit - (long long) bytes);
        ++g_checks;
        if (bytes >= limit) ++g_fail;
    }
    std::printf("qsa_slots: %d checks, %d failures\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("qsa_slots_parity OK\n");
    return 0;
}
