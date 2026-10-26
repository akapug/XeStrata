// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/l2_prefetch.hpp - homelab: device memory asked into L2 ahead of the kernel that reads it.
//
// The hyper-connection read takes ~112 us a call at 16 rows with its Q8_0 projections cold in L2 and ~86 hot on the
// B70 (hc_mode_probe), and in serving it always runs cold: each layer's projections were last read a step earlier.
// A prefetch launch has no outputs, so it changes no bit of any result; it costs only the launch and the bandwidth it
// borrows while the kernels between it and the read run.
#pragma once

#include <cstddef>

namespace strata::kernels {

/// STRATA_HC_PREFETCH (default 0, off): 1 = before a layer's attention output projection (gdn ssm_out, qsa o), one
/// launch that asks L2 for the projections of the layer's second hyper-connection read, which follows it.
constexpr int kHcPrefetchDefault = 0;
int hc_prefetch();
void hc_prefetch_set(int form);

/// One launch: every 64-byte line of [a, a + na) and of [b, b + nb) prefetched into L2 (sycl_ext_oneapi_prefetch,
/// prefetch_hint_L2).  Nothing is written; the device may drop any of the hints.  b may be null (nb 0).
void l2_prefetch(const void* a, size_t na, const void* b, size_t nb, void* stream);

}  // namespace strata::kernels
