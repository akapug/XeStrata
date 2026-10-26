// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/core/session.hpp"
#include "strata/kernels/qsa.hpp"

#include <climits>

namespace strata::core {

// The verifier's score scratch is sized for `main`, but each slot owns its context-sized state.  In particular,
// a smaller resident slot may share a window with a streamed main session; the KV format, not residency, must match.
inline bool session_slot_compatible(const ModelGeometry& g, const SessionState& main, const SessionState& slot,
                                     std::string& err) {
    if (slot.max_cells <= 0 || slot.max_cells > main.max_cells || slot.max_cells > INT32_MAX) {
        err = "verify: a slot's context is outside the verifier's capacity";
        return false;
    }
    if (slot.k != main.k || slot.layer_lo != main.layer_lo || slot.layer_hi != main.layer_hi ||
        slot.qsa_ord0 != main.qsa_ord0 || slot.qsa_alloc != main.qsa_alloc ||
        slot.gdn_ord0 != main.gdn_ord0 || slot.gdn_alloc != main.gdn_alloc) {
        err = "verify: a slot's session is not carved like the verifier's own (stage or experts)";
        return false;
    }
    if (slot.qsa_states == nullptr || main.qsa_states == nullptr || slot.qsa_ord0 < 0 || slot.qsa_alloc < 1 ||
        slot.qsa_ord0 > g.n_qsa_layers() - slot.qsa_alloc || slot.gdn_state == nullptr ||
        slot.ple_hist == nullptr) {
        err = "verify: a slot's session state is incomplete";
        return false;
    }
    const strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    const int64_t pages = (slot.max_cells + s.page_size - 1) / s.page_size;
    const int64_t pooled = slot.max_cells / s.idx_block + 2;
    for (int64_t j = slot.qsa_ord0; j < slot.qsa_ord0 + slot.qsa_alloc; ++j) {
        const QsaState& st = slot.qsa_states[j];
        const QsaState& ref = main.qsa_states[j];
        if (st.max_cells != slot.max_cells || st.n_pages != pages || st.idx_pooled_rows != pooled ||
            st.n_slots < 1 || st.n_slots > pages || st.kv_mode < 0 || st.kv_mode > 1 ||
            (st.kv_mode == 0 && st.n_slots != pages)) {
            err = "verify: a slot's QSA state is not carved for its own context";
            return false;
        }
        if (st.kv_int8 != ref.kv_int8 || st.kv_q4 != ref.kv_q4 || st.kv_hybrid != ref.kv_hybrid ||
            st.kv_rot != ref.kv_rot) {
            err = "verify: a slot's KV format differs from the verifier's own";
            return false;
        }
        if (st.page_table == nullptr || st.idx_tail == nullptr || st.idx_dead == nullptr ||
            st.idx_pooled == nullptr || st.idx_block_pos == nullptr || st.cos_tab == nullptr || st.sin_tab == nullptr ||
            (st.kv_hybrid ? st.k_q == nullptr || st.k_scale == nullptr || st.v_q4 == nullptr :
             st.kv_q4 ? st.k_q4 == nullptr || st.v_q4 == nullptr :
             st.kv_int8 ? st.k_q == nullptr || st.v_q == nullptr || st.k_scale == nullptr || st.v_scale == nullptr :
             st.k_pool == nullptr || st.v_pool == nullptr) ||
            (st.kv_mode == 1 && (!st.host.present() || st.map.page_table != st.page_table ||
                                st.map.n_blocks != pages || st.map.n_slots != st.n_slots ||
                                st.map.slot_block == nullptr || st.map.slot_stamp == nullptr ||
                                st.map.slot_ref == nullptr || st.map.miss_block == nullptr ||
                                st.map.miss_slot == nullptr || st.map.ctl == nullptr))) {
            err = "verify: a slot's QSA state is incomplete";
            return false;
        }
    }
    return true;
}

}  // namespace strata::core
