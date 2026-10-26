// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/shared_prefix_parity.cpp - homelab shared-prefix reuse across slots (STRATA_SHARED_PREFIX): a session that
// starts from a prefix another session holds (conversation_prefix_copy) must hold exactly what a session that read the
// prefix itself holds, and must never change the session it copied from.
//
// Three sessions per case, in every KV format (fp16, INT8, Q4_0, K8V4) and residency mode (resident, streamed):
//   A reads the shared prefix S (P tokens), saves its checkpoint R there (as the engine's learned cut or system-prompt
//     root does), then reads its own suffix `a` (another agent's conversation going on);
//   B reads S, then the suffix `c` - the reference, a fresh read of the whole prompt;
//   C starts from junk, takes R with A's K/V below it (conversation_prefix_copy: on the default queue, as the request
//     path's copy_from_slot does, and on a stream of its own, as a window-row prefill slot's setup does), then reads c.
// C must equal B in every K/V cell below its length (codes and scales), every completed pooled indexer row and the
// spare row, the GDN state and conv history, the indexer tail, dead key and block position, the PLE history and its
// token window; A must be byte-for-byte what it was (whole pools, running state, streaming map).  The reads are a
// synthetic model: each token's bytes are a hash of the state before it (read from the session's own GDN state) and
// the token, so a wrong running state or a stale cell changes everything after it.  Controls show the comparison is
// not blind: the running state restored BEFORE the K/V copy (copy_from_slot's order before homelab's fix) leaves A's
// later pooled row in C's spare row, and a checkpoint of the wrong position diverges.  copy_from_slot's read-back
// (conversation_slot_read_back) passes the same comparison at a slot checkpoint, buffer to buffer and through host
// images, and reads a whole conversation back to exactly what the slot holds.  No model, no GPU kernels.
#include "conversation_fixture.hpp"

#include <cstring>
#include <string>

using namespace conversation_fixture;

namespace {
int fails = 0;
void expect(bool ok, const std::string& what) {
    ++checks;
    if (ok) return;
    ++fails;
    std::printf("    *** %s ***\n", what.c_str());
}

uint64_t mix(uint64_t h, int64_t tok, int64_t pos) {
    h ^= (uint64_t) tok * 0x9e3779b97f4a7c15ull + (uint64_t) pos;
    h *= 0xff51afd7ed558ccdull;
    return h ^ (h >> 29);
}
std::vector<uint8_t> pattern(uint64_t h, uint64_t salt, size_t n) {
    std::vector<uint8_t> v(n);
    uint64_t x = h ^ (salt * 0xc4ceb9fe1a85ec53ull);
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v[i] = (uint8_t) x;
    }
    return v;
}
template <class T> std::vector<uint8_t> fetch(const T* p, size_t n) {
    std::vector<uint8_t> v(n);
    if (n) gpu_check(strata::gpu::copy(v.data(), p, n));
    return v;
}
void put(void* p, const std::vector<uint8_t>& v) { if (!v.empty()) gpu_check(strata::gpu::copy(p, v.data(), v.size())); }

ModelGeometry geometry() {
    ModelGeometry g;
    g.n_layers = 4;   // one QSA layer, three GDN layers (as conversation_snapshot_test's full session)
    g.n_expert = 256;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    return g;
}

struct Slot {
    Fixture f;
    SessionState ss;
    ConversationStateSizes z;
    Slot(int fmt, int mode) : f(fmt, mode) {
        f.g = geometry();
        std::string err;
        check(conversation_state_sizes(f.g, z, err), "session sizes");
        ss.max_cells = 96;
        ss.qsa_states = &f.state;
        f.alloc(ss.gdn_state, z.gdn); f.alloc(ss.ple_hist, z.ple);
        f.alloc(f.state.idx_tail, z.tail); f.alloc(f.state.idx_dead, z.dead); f.alloc(f.state.idx_block_pos, z.block_pos);
    }
    std::vector<std::pair<void*, size_t>> running() const {
        return {{ss.gdn_state, z.gdn}, {ss.ple_hist, z.ple}, {f.state.idx_tail, z.tail}, {f.state.idx_dead, z.dead},
                {f.state.idx_block_pos, z.block_pos}};
    }
    void wipe(int v) {   // every buffer to `v` (0: the empty session a read from token 0 starts with)
        for (size_t i = 0; i < f.sources.size(); ++i)
            if (f.sizes[i]) gpu_check(strata::gpu::memset(f.sources[i], v, f.sizes[i]));
        for (const auto& [p, n] : running()) if (n) gpu_check(strata::gpu::memset(p, v, n));
        ss.ple_prev[0] = ss.ple_prev[1] = -1;
        gpu_check(strata::gpu::device_sync());
    }
    // the bytes of cell c, head hh in K/V pool i: [page][kv_head][page_size][unit]
    size_t unit(size_t i) const { return f.sizes[i] / (size_t) (f.state.max_cells * f.g.n_head_kv); }
    size_t at(size_t i, int64_t c, int64_t hh) const {
        const int64_t pg = qsa_real_shapes().page_size;
        return (size_t) (((c / pg) * f.g.n_head_kv + hh) * pg + c % pg) * unit(i);
    }
    size_t row_bytes() const { return (size_t) f.g.idx_key_dim * 4; }
    // reads tokens [from, to): each token's bytes from the state before it (its first 8 GDN bytes) and the token
    void read(const std::vector<int32_t>& toks, int64_t from, int64_t to) {
        uint64_t h = 0x5eed;
        if (from > 0) {
            const std::vector<uint8_t> g8 = fetch(ss.gdn_state, 8);
            std::memcpy(&h, g8.data(), 8);
        }
        const int64_t block = qsa_real_shapes().idx_block;
        for (int64_t p = from; p < to; ++p) {
            h = mix(h, toks[(size_t) p], p);
            for (size_t i = 0; i < 4; ++i)
                for (int64_t hh = 0; hh < f.g.n_head_kv && f.sizes[i]; ++hh)
                    put(static_cast<uint8_t*>(f.sources[i]) + at(i, p, hh), pattern(h, 10 + i * 8 + (uint64_t) hh, unit(i)));
            const int64_t len = p + 1;
            const std::vector<uint8_t> dead = pattern(h, 3, z.dead);
            if (len % block == 0)
                put(f.state.idx_pooled + (size_t) (len / block - 1) * f.g.idx_key_dim, pattern(h, 4, row_bytes()));
            put(f.state.idx_pooled + (size_t) (len / block) * f.g.idx_key_dim, dead);   // the spare row: the dead keys
            std::vector<uint8_t> gdn = pattern(h, 1, z.gdn);
            std::memcpy(gdn.data(), &h, 8);
            put(ss.gdn_state, gdn);
            put(ss.ple_hist, pattern(h, 2, z.ple));
            put(f.state.idx_tail, pattern(h, 5, z.tail));
            put(f.state.idx_dead, dead);
            put(f.state.idx_block_pos, pattern(h, 6, z.block_pos));
        }
        if (to > from) {
            ss.ple_prev[0] = to >= 2 ? toks[(size_t) to - 2] : -1;
            ss.ple_prev[1] = toks[(size_t) to - 1];
        }
        gpu_check(strata::gpu::device_sync());
    }
    // everything a later token reads after `len` tokens: the cells below len, the completed pooled rows and the spare
    // row, the running state and the PLE token window
    std::vector<uint8_t> valid(int64_t len) const {
        std::vector<uint8_t> v;
        auto add = [&](const std::vector<uint8_t>& x) { v.insert(v.end(), x.begin(), x.end()); };
        for (size_t i = 0; i < 4; ++i)
            for (int64_t c = 0; c < len && f.sizes[i]; ++c)
                for (int64_t hh = 0; hh < f.g.n_head_kv; ++hh)
                    add(fetch(static_cast<const uint8_t*>(f.sources[i]) + at(i, c, hh), unit(i)));
        add(fetch(f.state.idx_pooled, (size_t) (len / qsa_real_shapes().idx_block + 1) * row_bytes()));
        for (const auto& [p, n] : running()) add(fetch(static_cast<const uint8_t*>(p), n));
        const int32_t w[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        v.insert(v.end(), (const uint8_t*) w, (const uint8_t*) w + sizeof w);
        return v;
    }
    // every byte the session owns (whole pools, running state, the streaming map's page table)
    std::vector<uint8_t> all() const {
        std::vector<uint8_t> v;
        auto add = [&](const std::vector<uint8_t>& x) { v.insert(v.end(), x.begin(), x.end()); };
        for (size_t i = 0; i < f.sources.size(); ++i) add(fetch(static_cast<const uint8_t*>(f.sources[i]), f.sizes[i]));
        for (const auto& [p, n] : running()) add(fetch(static_cast<const uint8_t*>(p), n));
        if (f.state.kv_mode == 1) add(fetch(f.state.page_table, (size_t) f.state.n_pages * 4));
        return v;
    }
};

std::vector<int32_t> prompt(int64_t shared, int64_t own, int32_t salt) {
    std::vector<int32_t> t;
    for (int64_t i = 0; i < shared; ++i) t.push_back((int32_t) (1000 + i * 7));
    for (int64_t i = 0; i < own; ++i) t.push_back((int32_t) (salt + i * 13));
    return t;
}

// how C takes the prefix: 0 the request path's (default queue), 1 a window slot's setup (its own stream),
// 2 the control with the running state restored before the K/V copy (copy_from_slot's order before the fix),
// 3 the control with the wrong checkpoint, 4 and 5 copy_from_slot's read-back at a slot checkpoint
// (conversation_slot_read_back: buffer to buffer, then through host images)
bool start_from(int how, const ConversationCheckpoint& r, const ConversationCheckpoint& wrong, Slot& a, Slot& c,
                strata::gpu::Stream q) {
    std::string err;
    if (how == 0) return conversation_prefix_copy(r, a.ss, c.ss, a.f.g, nullptr, err);
    if (how == 1) return conversation_prefix_copy(r, a.ss, c.ss, a.f.g, q, err);
    if (how == 2)
        return conversation_checkpoint_restore(r, c.ss, a.f.g, err) &&
               conversation_kv_copy(a.f.state, c.f.state, a.f.g, (int64_t) r.ids.size(), true, err) &&
               strata::gpu::device_sync();
    if (how == 3) return conversation_prefix_copy(wrong, a.ss, c.ss, a.f.g, nullptr, err);
    return conversation_slot_read_back(&r, {}, a.ss, c.ss, a.f.g, how == 4, err);
}
constexpr int kHows = 6;
bool control(int how) { return how == 2 || how == 3; }
const char* const kHow[] = {" (default queue)", " (stream)", "", "", " (slot read-back, direct)",
                            " (slot read-back, images)"};

void one(int fmt, int mode, int64_t P, int64_t la, int64_t lc, strata::gpu::Stream q) {
    static const char* const kFmt[] = {"fp16", "int8", "q4_0", "k8v4"};
    const std::string tag = std::string(kFmt[fmt]) + (mode ? " streamed" : " resident") + " P=" + std::to_string(P) +
                            " a=" + std::to_string(la) + " c=" + std::to_string(lc);
    const std::vector<int32_t> ta = prompt(P, la, 50000), tc = prompt(P, lc, 70000);
    Slot a(fmt, mode), b(fmt, mode);
    std::string err;
    a.wipe(0);
    a.read(ta, 0, P);
    ConversationCheckpoint r;
    r.ids.assign(ta.begin(), ta.begin() + P);
    check(conversation_checkpoint_save(r, a.ss, a.f.g, err), "save the shared prefix's checkpoint");
    a.read(ta, P, P + la);
    ConversationCheckpoint wrong;   // A's state where its own conversation stands, labelled as the prefix
    wrong.ids = r.ids;
    check(conversation_checkpoint_save(wrong, a.ss, a.f.g, err), "save a checkpoint of the wrong position");
    b.wipe(0);
    b.read(tc, 0, P);
    const std::vector<uint8_t> b_at_p = b.valid(P);
    b.read(tc, P, P + lc);
    const std::vector<uint8_t> want = b.valid(P + lc);
    const std::vector<uint8_t> a0 = a.all();
    const int64_t block = qsa_real_shapes().idx_block;
    // A's own suffix filled the prefix's last pooled block, and C's suffix does not reach that block's end
    const bool spare_stale = P % block != 0 && P + la >= (P / block + 1) * block && P + lc < (P / block + 1) * block;
    for (int how = 0; how < kHows; ++how) {
        Slot c(fmt, mode);
        c.wipe(0x5a);
        c.ss.ple_prev[0] = c.ss.ple_prev[1] = -7;
        check(start_from(how, r, wrong, a, c, q), "C takes the prefix");
        if (!control(how))
            expect(c.valid(P) == b_at_p, tag + kHow[how] + ": the copied prefix is the state a read of it leaves");
        c.read(tc, P, P + lc);
        const bool same = c.valid(P + lc) == want;
        if (!control(how)) {
            expect(same, tag + kHow[how] + ": C after its suffix equals the fresh read");
            expect(a.all() == a0, tag + kHow[how] + ": the source session is unchanged");
        } else if (how == 2) {
            if (spare_stale) expect(!same, tag + ": control: the restore before the K/V copy keeps A's later pooled row");
        } else {
            expect(!same, tag + ": control: a checkpoint of the wrong position diverges");
        }
    }
    // copy_from_slot's whole-conversation read-back (no checkpoint): C takes all A holds and equals A
    const std::vector<int32_t> held(ta.begin(), ta.begin() + P + la);
    for (bool direct : {true, false}) {
        Slot c(fmt, mode);
        c.wipe(0x5a);
        check(conversation_slot_read_back(nullptr, held, a.ss, c.ss, a.f.g, direct, err), "C reads A back whole");
        expect(c.valid(P + la) == a.valid(P + la) && a.all() == a0,
               tag + (direct ? " (whole read-back, direct)" : " (whole read-back, images)") +
                   ": C holds what A holds, A unchanged");
    }
    ConversationCheckpoint bad = r;   // a checkpoint of another geometry is refused before any copy
    bad.gdn.pop_back();
    Slot c(fmt, mode);
    c.wipe(0x33);
    const std::vector<uint8_t> c0 = c.all();
    expect(!conversation_prefix_copy(bad, a.ss, c.ss, a.f.g, nullptr, err) && c.all() == c0,
           tag + ": a short checkpoint is refused and nothing is written");
    for (bool direct : {true, false})
        expect(!conversation_slot_read_back(&bad, {}, a.ss, c.ss, a.f.g, direct, err) && c.all() == c0,
               tag + ": the slot read-back refuses a short checkpoint and writes nothing");
}
}  // namespace

int main() {
    void* probe = strata::gpu::alloc_device(1);   // no usable device: skipped
    if (probe == nullptr) return 77;
    strata::gpu::free(probe);
    strata::gpu::Stream q = strata::gpu::stream_create();
    gpu_check(q != nullptr);
    struct Case { int64_t P, la, lc; };
    // mid-block and block-aligned prefixes; A's suffix past the prefix's block (the stale spare row), C's within it,
    // C's across blocks and pages, C with no suffix (the copy alone)
    const Case cases[] = {{11, 9, 0}, {11, 9, 1}, {11, 2, 6}, {12, 5, 2}, {9, 20, 23}, {33, 17, 30}};
    for (int fmt : {(int) kKvF16, (int) kKvInt8, (int) kKvQ4, 3})
        for (int mode : {0, 1}) {
            if (fmt == 3 && mode != 0) continue;   // K8V4 is never streamed (conversation_snapshot refuses it)
            for (const Case& k : cases) one(fmt, mode, k.P, k.la, k.lc, q);
            std::printf("  %s %s: %zu cases\n", fmt == kKvF16 ? "fp16" : fmt == kKvInt8 ? "int8" : fmt == kKvQ4 ? "q4_0" : "k8v4",
                        mode ? "streamed" : "resident", sizeof cases / sizeof cases[0]);
        }
    strata::gpu::stream_destroy(q);
    std::printf("shared_prefix_parity: %d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
