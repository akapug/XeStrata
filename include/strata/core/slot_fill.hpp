// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/slot_fill.hpp - homelab staged refill: expert blobs into expert-cache slots through a few pinned
// bounce buffers.
//
// After a batched prompt read the slots it borrowed get their experts back (generate.cpp's `refill`).  A blob a
// source assembles into a short-lived buffer (`transient`: the GGUF read in place, which STRATA_STAGE_LEND_DROP=1
// makes of every lent expert) was copied into one pageable buffer and sent with a blocking copy, one slot at a time:
// the host copy, the runtime's staging of the pageable bytes and the wait all in line.  odie 2026-10-24 (STRATA_TRACE,
// 0.1.45): about 0.25 ms a slot, 3.3 s after a 32768-token chunk's read (13280 slots).
//
// Here `threads` host threads copy the blobs into `buffers` pinned buffers in turn, each buffer's DMA goes on a copy
// queue, a buffer is reused once its DMA is done, and the caller waits once at the end.  The bytes in each slot are
// the source's blob, as before (slot_fill_parity checks it against the blocking copy).
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::gpu { struct Event; }

namespace strata::core {

/// STRATA_REFILL_STAGED (off by default), STRATA_REFILL_BUFFERS, STRATA_REFILL_THREADS.
struct SlotFillSettings {
    static constexpr bool kOnDefault = false;
    static constexpr int kBuffersDefault = 8;   ///< pinned buffers per GPU (each the largest blob: ~2.3 MB on IQ3_XXS)
    static constexpr int kThreadsDefault = 4;   ///< host copy threads per GPU while it refills
    static constexpr int kMax = 64;
    bool on = kOnDefault;
    int buffers = kBuffersDefault;
    int threads = kThreadsDefault;
};

/// The settings from the three variables (null: unset).  False with `why` on an invalid value; `s` is then the
/// defaults (off).  buffers 2..kMax, threads 1..kMax.
bool parse_slot_fill_settings(const char* on, const char* buffers, const char* threads, SlotFillSettings& s,
                              std::string& why);

class SlotFiller {
public:
    struct Fill {
        int32_t slot = -1;
        int32_t layer = 0;
        int32_t expert = 0;
    };
    SlotFiller() = default;
    ~SlotFiller();
    SlotFiller(const SlotFiller&) = delete;
    SlotFiller& operator=(const SlotFiller&) = delete;

    /// `src`'s blob of each (layer, expert) into `cache`'s slot, through the bounce buffers (made at the first call, on
    /// the current device; every later call must be on that device).  Returns once every copy has landed.  False with
    /// `err` on a failed copy (the slots it reached hold their blobs; the rest are unchanged or partly written).
    bool fill(ExpertSource& src, ExpertCache& cache, const std::vector<Fill>& fills, int buffers, int threads,
              std::string& err);

    /// Bytes of page-locked host memory the buffers hold (0 before the first call; pageable fallbacks not counted).
    uint64_t pinned_bytes() const { return pinned_bytes_; }

private:
    bool init(int buffers, std::string& err);
    int device_ = -1;
    void* copy_ = nullptr;                          // strata::gpu::Stream: the copy queue
    std::vector<uint8_t*> buf_;
    std::vector<char> pinned_;
    std::vector<std::vector<uint8_t>> pageable_;    // the fallback when no more RAM can be pinned
    std::vector<strata::gpu::Event*> done_;         // buffer i's last DMA
    uint64_t pinned_bytes_ = 0;
};

}  // namespace strata::core
