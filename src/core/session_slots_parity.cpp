// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Real session arenas at different contexts, the verifier's admission rules and their malformed-shape refusals.
// No weights or model file.  qsa_slots_parity checks the mixed window's numerical outputs separately.
#include "strata/core/session_slots.hpp"
#include "strata/kernels/elementwise.hpp"

#include "../kernels/parity_device.hpp"

#include <cstdio>
#include <functional>
#include <limits>

namespace c = strata::core;
namespace k = strata::kernels;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what); }
}

struct Session {
    c::SessionState state;
    void* base = nullptr;
    Session(const c::ModelGeometry& g, int64_t cells, int64_t lo, int64_t hi, const c::SessionState* donor = nullptr) {
        const uint64_t bytes = c::session_bytes(g, cells, 10, lo, hi, donor == nullptr);
        base = strata::gpu::alloc_device((size_t) bytes);
        if (base == nullptr) throw c::DeviceError("session_slots: session arena allocation failed");
        const uint64_t used = c::session_init(g, cells, 10, base, state, lo, hi, donor);
        check(used > 0 && used <= bytes, "context-sized session fits its own arena");
        if (used == 0) throw c::DeviceError("session_slots: session initialization failed");
    }
    ~Session() {
        c::session_release(state);
        for (int64_t j = state.qsa_ord0; j < state.qsa_ord0 + state.qsa_alloc; ++j) {
            strata::gpu::free(state.qsa_states[j].host_step);
            strata::gpu::free(state.qsa_states[j].host_pos);
            // Host pool runs are carved from one allocation beginning with the format's first K plane.
            const k::KvHostPools& host = state.qsa_states[j].host;
            strata::gpu::free(host.k_pool != nullptr ? (const void*) host.k_pool :
                              host.k_q != nullptr ? (const void*) host.k_q : (const void*) host.k_q4);
        }
        delete[] state.qsa_states;
        strata::gpu::free(base);
    }
};

void carve(int64_t lo, int64_t hi, int format, bool streamed) {
    c::ModelGeometry g;
    g.n_layers = 8;   // two QSA layers: also exercise a nonzero global ordinal and a stage without QSA
    c::qsa_set_kv_int8(format == k::kKvInt8);
    c::qsa_set_kv_q4(format == k::kKvQ4);
    c::qsa_set_kv_hybrid(format == k::kKvHybrid);
    const int64_t page = k::qsa_real_shapes().page_size;
    const int64_t big = c::qsa_kv_resident_min() + 2 * page, small = page + 1;
    c::qsa_set_kv_resident(streamed ? c::qsa_kv_resident_min() : 0);
    Session main(g, big, lo, hi);
    Session same(g, big, lo, hi, &main.state);
    Session short_slot(g, small, lo, hi, &main.state);
    std::string err;
    const auto compatible = [&] { return c::session_slot_compatible(g, main.state, short_slot.state, err); };
    check(c::session_slot_compatible(g, main.state, same.state, err), "homogeneous slot accepted");
    check(compatible(), "smaller slot accepted beside the big slot");
    const int64_t qi = short_slot.state.qsa_ord0;
    const c::QsaState& qs = short_slot.state.qsa_states[qi];
    check(qs.max_cells == small && qs.n_pages == 2 && qs.cos_tab == main.state.qsa_states[qi].cos_tab,
          "small slot carves its own pages and borrows the big RoPE table");
    check(main.state.qsa_states[qi].kv_mode == (streamed ? 1 : 0) && qs.kv_mode == 0,
          "resident small slot can share with a streamed big slot");

    // Restore every mutation before testing the next one: these are real initialized sessions, not shape doubles.
    const auto refuse = [&](const char* name, const std::function<void()>& mutate) {
        const c::SessionState saved = short_slot.state;
        const c::QsaState saved_qs = qs;
        mutate();
        err.clear();
        check(!compatible() && !err.empty(), name);
        short_slot.state = saved;
        short_slot.state.qsa_states[qi] = saved_qs;
    };
    refuse("zero context refused", [&] { short_slot.state.max_cells = 0; });
    refuse("context above scratch capacity refused", [&] { short_slot.state.max_cells = big + 1; });
    refuse("overflowing context refused", [&] { short_slot.state.max_cells = std::numeric_limits<int64_t>::max(); });
    refuse("wrong expert count refused", [&] { ++short_slot.state.k; });
    refuse("wrong stage refused", [&] { ++short_slot.state.layer_lo; });
    refuse("wrong QSA ordinal refused", [&] { ++short_slot.state.qsa_ord0; });
    refuse("wrong QSA allocation refused", [&] { ++short_slot.state.qsa_alloc; });
    refuse("wrong GDN ordinal refused", [&] { ++short_slot.state.gdn_ord0; });
    refuse("wrong GDN allocation refused", [&] { ++short_slot.state.gdn_alloc; });
    refuse("missing QSA state refused", [&] { short_slot.state.qsa_states = nullptr; });
    refuse("missing recurrence refused", [&] { short_slot.state.gdn_state = nullptr; });
    refuse("QSA carved for another context refused", [&] { ++short_slot.state.qsa_states[qi].max_cells; });
    refuse("short page table refused", [&] { --short_slot.state.qsa_states[qi].n_pages; });
    refuse("short indexer refused", [&] { --short_slot.state.qsa_states[qi].idx_pooled_rows; });
    refuse("short resident pool refused", [&] { --short_slot.state.qsa_states[qi].n_slots; });
    refuse("drafter ring refused", [&] { short_slot.state.qsa_states[qi].kv_mode = 2; });
    refuse("mixed rotation refused", [&] { short_slot.state.qsa_states[qi].kv_rot = !qs.kv_rot; });
    refuse("mixed INT8 format refused", [&] { short_slot.state.qsa_states[qi].kv_int8 = !qs.kv_int8; });
    refuse("mixed Q4 format refused", [&] { short_slot.state.qsa_states[qi].kv_q4 = !qs.kv_q4; });
    refuse("mixed hybrid format refused", [&] { short_slot.state.qsa_states[qi].kv_hybrid = !qs.kv_hybrid; });
    refuse("missing page table refused", [&] { short_slot.state.qsa_states[qi].page_table = nullptr; });
    refuse("missing indexer refused", [&] { short_slot.state.qsa_states[qi].idx_pooled = nullptr; });
    refuse("missing RoPE table refused", [&] { short_slot.state.qsa_states[qi].cos_tab = nullptr; });
    if (streamed) {
        c::QsaState& q = same.state.qsa_states[qi];
        const auto map = q.map;
        q.map.n_slots = 0;
        check(!c::session_slot_compatible(g, main.state, same.state, err), "malformed streamed map refused");
        q.map = map;
    }

    // Invalid borrowing must fail before it allocates staging or changes the destination session.
    c::SessionState uncarved;
    check(c::session_init(g, big, 10, nullptr, uncarved, lo, hi, &short_slot.state) == 0 &&
          uncarved.qsa_states == nullptr, "short RoPE donor refused before carving");
    c::SessionState absent;
    check(c::session_init(g, small, 10, nullptr, uncarved, lo, hi, &absent) == 0 &&
          uncarved.qsa_states == nullptr, "missing RoPE donor refused before carving");
    std::printf("session_slots: stage [%lld,%lld), format %d, streamed %d checked\n",
                (long long) lo, (long long) hi, format, streamed);
}
}  // namespace

int main() {
    for (int format : {k::kKvF16, k::kKvInt8, k::kKvQ4, k::kKvHybrid}) {
        carve(0, 8, format, false);
        carve(4, 8, format, false);
        carve(0, 3, format, false);
        carve(4, 8, format, true);
    }
    std::printf("session_slots_parity: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
