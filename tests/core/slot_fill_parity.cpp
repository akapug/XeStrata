// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// homelab staged refill (slot_fill.hpp): the slots a SlotFiller fills hold, byte for byte, what the blocking refill
// writes (the source's blob copied into a pageable buffer, then fill_slot_blocking), for every buffer and thread count,
// fewer and more fills than buffers, a native pack's per-layer blob sizes and a failed copy; and the settings parse.
// Needs an engine device (STRATA_TEST_CPU_DEVICE=1: the OpenCL CPU device); exits 77 without one.
#include "strata/core/slot_fill.hpp"

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::SlotFiller;
using strata::core::SlotFillSettings;

namespace {

int g_checks = 0;

void require(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) throw std::runtime_error(what);
}

struct TempDirectory {
    fs::path path;
    TempDirectory() {
        path = fs::temp_directory_path() /
               ("strata-slot-fill-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void test_settings() {
    SlotFillSettings s;
    std::string why;
    require(strata::core::parse_slot_fill_settings(nullptr, nullptr, nullptr, s, why) && !s.on &&
                s.buffers == SlotFillSettings::kBuffersDefault && s.threads == SlotFillSettings::kThreadsDefault,
            "unset: not the defaults");
    require(strata::core::parse_slot_fill_settings("1", "3", "2", s, why) && s.on && s.buffers == 3 && s.threads == 2,
            "1/3/2 not read");
    require(strata::core::parse_slot_fill_settings("0", "", "", s, why) && !s.on, "0 not read as off");
    for (const char* bad_on : {"2", "yes", "1x"}) {
        why.clear();
        require(!strata::core::parse_slot_fill_settings(bad_on, nullptr, nullptr, s, why) && !s.on && !why.empty(),
                std::string("STRATA_REFILL_STAGED=") + bad_on + " accepted");
    }
    for (const char* bad : {"1", "0", "65", "x", "4k"}) {
        why.clear();
        require(!strata::core::parse_slot_fill_settings("1", bad, nullptr, s, why) && !s.on && !why.empty(),
                std::string("STRATA_REFILL_BUFFERS=") + bad + " accepted");
    }
    for (const char* bad : {"0", "65", "-1"}) {
        why.clear();
        require(!strata::core::parse_slot_fill_settings("1", nullptr, bad, s, why) && !s.on && !why.empty(),
                std::string("STRATA_REFILL_THREADS=") + bad + " accepted");
    }
}

// a pack of `layers` x `experts` blobs (layer l's blob `bytes[l]`) of seeded bytes
void write_pack(const fs::path& dir, const std::vector<uint64_t>& bytes, int64_t experts) {
    std::ofstream out(dir / "experts.bin", std::ios::binary | std::ios::trunc);
    std::mt19937 rng(353);
    std::vector<char> b;
    for (uint64_t lb : bytes)
        for (int64_t e = 0; e < experts; ++e) {
            b.resize((size_t) lb);
            for (char& c : b) c = (char) (rng() & 0xff);
            out.write(b.data(), (std::streamsize) b.size());
        }
    require((bool) out, "the fixture pack could not be written");
}

std::vector<uint8_t> read_slot(strata::core::ExpertCache& cache, int32_t slot, size_t bytes) {
    std::vector<uint8_t> v(bytes);
    require(strata::gpu::copy(v.data(), cache.device_slot(slot), bytes), "reading a slot back failed");
    return v;
}

// the blocking refill of generate.cpp (a transient blob: copy_blob into a pageable buffer, fill_slot_blocking)
void fill_blocking(strata::core::ExpertSource& src, strata::core::ExpertCache& cache,
                   const std::vector<SlotFiller::Fill>& fills) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint8_t> tmp((size_t) lay.max_blob);
    std::string err;
    for (const SlotFiller::Fill& f : fills) {
        require(src.copy_blob(f.layer, f.expert, tmp.data()), "copy_blob failed");
        require(cache.fill_slot_blocking(f.slot, tmp.data(), err, (int64_t) lay.blob_bytes(f.layer)), err);
    }
}

void scribble(strata::core::ExpertCache& cache, int32_t slots, size_t bytes) {
    std::vector<uint8_t> junk(bytes, 0xA5);
    std::string err;
    for (int32_t s = 0; s < slots; ++s) require(cache.fill_slot_blocking(s, junk.data(), err, (int64_t) bytes), err);
}

// layers x experts blobs (per-layer sizes `bytes`), `slots` slots; every pattern of fills, buffers and threads
void run_pack(const char* name, const fs::path& dir, const std::vector<uint64_t>& bytes, int64_t experts, bool sized) {
    const auto layers = (int64_t) bytes.size();
    std::string err;
    strata::core::FileExpertSource src;
    require(src.open(dir.string(), layers, experts, err), "open: " + err);
    const auto& lay = strata::kernels::cpu::expert_layout();
    const int32_t slots = 40;
    const size_t max_b = (size_t) lay.max_blob;
    strata::core::ExpertCache cache;
    require(sized ? cache.open_sized(std::vector<int64_t>((size_t) slots, (int64_t) max_b), layers, experts, err)
                  : cache.open(slots, layers, experts, (int64_t) max_b, err),
            "cache open: " + err);
    // fills: distinct slots in a shuffled order, blobs from every layer
    std::mt19937 rng(11573);
    std::vector<SlotFiller::Fill> all;
    for (int32_t s = 0; s < slots; ++s)
        all.push_back({s, (int32_t) (rng() % (uint32_t) layers), (int32_t) (rng() % (uint32_t) experts)});
    std::shuffle(all.begin(), all.end(), rng);
    // the reference: what the blocking refill leaves in each slot
    scribble(cache, slots, max_b);
    fill_blocking(src, cache, all);
    std::vector<std::vector<uint8_t>> want((size_t) slots);
    for (const SlotFiller::Fill& f : all) {
        want[(size_t) f.slot] = read_slot(cache, f.slot, (size_t) lay.blob_bytes(f.layer));
        std::vector<uint8_t> blob((size_t) lay.blob_bytes(f.layer));
        require(src.copy_blob(f.layer, f.expert, blob.data()) && blob == want[(size_t) f.slot],
                "the blocking refill does not hold the source's blob");
    }
    int cases = 0;
    for (int buffers : {2, 3, 8}) {
        SlotFiller filler;   // one per buffer count: the buffers are made at the first fill
        for (int threads : {1, 2, 4, 7})
            for (size_t n : {(size_t) 1, (size_t) 2, (size_t) 5, all.size()}) {
                const std::vector<SlotFiller::Fill> part(all.begin(), all.begin() + (std::ptrdiff_t) n);
                scribble(cache, slots, max_b);
                require(filler.fill(src, cache, part, buffers, threads, err), "fill: " + err);
                for (const SlotFiller::Fill& f : part)
                    require(read_slot(cache, f.slot, (size_t) lay.blob_bytes(f.layer)) == want[(size_t) f.slot],
                            std::string(name) + ": slot " + std::to_string(f.slot) + " differs (buffers " +
                                std::to_string(buffers) + ", threads " + std::to_string(threads) + ", fills " +
                                std::to_string(n) + ")");
                // the slots not filled are untouched
                for (const SlotFiller::Fill& f : std::vector<SlotFiller::Fill>(all.begin() + (std::ptrdiff_t) n, all.end()))
                    require(read_slot(cache, f.slot, 1)[0] == 0xA5, "a slot outside the fills was written");
                ++cases;
            }
        require(filler.fill(src, cache, {}, buffers, 4, err), "an empty fill failed");
    }
    // a copy the source refuses: false with a reason, and the filler works after it
    {
        SlotFiller filler;
        std::vector<SlotFiller::Fill> bad(all.begin(), all.begin() + 12);
        bad[6].expert = (int32_t) experts;   // outside the pack
        err.clear();
        require(!filler.fill(src, cache, bad, 3, 2, err) && !err.empty(), "a refused copy was not reported");
        scribble(cache, slots, max_b);
        require(filler.fill(src, cache, all, 3, 2, err), "fill after a failure: " + err);
        for (const SlotFiller::Fill& f : all)
            require(read_slot(cache, f.slot, (size_t) lay.blob_bytes(f.layer)) == want[(size_t) f.slot],
                    "a fill after a failure differs");
        require(filler.pinned_bytes() == 0 || filler.pinned_bytes() == 3 * max_b, "pinned bytes miscounted");
    }
    // the time of both ways on this device (informational: a CPU device's copies are memcpy)
    {
        SlotFiller filler;
        require(filler.fill(src, cache, all, SlotFillSettings::kBuffersDefault, SlotFillSettings::kThreadsDefault, err), err);
        const auto t0 = std::chrono::steady_clock::now();
        fill_blocking(src, cache, all);
        const auto t1 = std::chrono::steady_clock::now();
        require(filler.fill(src, cache, all, SlotFillSettings::kBuffersDefault, SlotFillSettings::kThreadsDefault, err), err);
        const auto t2 = std::chrono::steady_clock::now();
        std::printf("%s: %zu fills of %.2f MB: blocking %.3f ms a slot, staged (%d buffers, %d threads) %.3f ms a slot; "
                    "%d cases byte-identical\n", name, all.size(), (double) max_b / 1e6,
                    std::chrono::duration<double, std::milli>(t1 - t0).count() / (double) all.size(),
                    SlotFillSettings::kBuffersDefault, SlotFillSettings::kThreadsDefault,
                    std::chrono::duration<double, std::milli>(t2 - t1).count() / (double) all.size(), cases);
    }
}

}  // namespace

int main() {
    try {
        test_settings();
        try { strata::core::Runtime::get(); } catch (const std::exception& e) {
            std::printf("slot_fill_parity: no engine device (%s): settings only, %d checks\n", e.what(), g_checks);
            return 77;
        }
        {
            TempDirectory dir;
            std::string err;
            const int64_t layers = 3, experts = 16;
            require(strata::kernels::cpu::expert_layout_load(dir.path.string(), layers, experts, err), err);
            write_pack(dir.path, std::vector<uint64_t>((size_t) layers, (uint64_t) strata::kernels::cpu::BLOB), experts);
            run_pack("canonical", dir.path, std::vector<uint64_t>((size_t) layers, (uint64_t) strata::kernels::cpu::BLOB),
                     experts, false);
        }
#if defined(STRATA_NATIVE_EXPERTS)
        {   // a native pack: per-layer blob sizes (IQ3_XXS and IQ2_XS gate/up), a sized cache
            TempDirectory dir;
            std::string err;
            const int64_t experts = 16;
            strata::kernels::cpu::NativeFmt a, b;
            require(strata::kernels::cpu::native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, 2560, 640, a, err), err);
            require(strata::kernels::cpu::native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, 2560, 640, b, err), err);
            const std::vector<uint64_t> bytes{(uint64_t) a.bytes, (uint64_t) b.bytes, (uint64_t) a.bytes};
            {
                std::ofstream m(dir.path / "native_experts.txt");
                uint64_t off = 0;
                const int types[3] = {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS};
                for (size_t l = 0; l < bytes.size(); ++l) {
                    m << l << ' ' << types[l] << ' ' << GGML_TYPE_IQ4_NL << ' ' << off << ' ' << bytes[l] << '\n';
                    off += bytes[l] * (uint64_t) experts;
                }
                require((bool) m, "native_experts.txt could not be written");
            }
            write_pack(dir.path, bytes, experts);
            require(strata::kernels::cpu::expert_layout_load(dir.path.string(), (int64_t) bytes.size(), experts, err), err);
            require(strata::kernels::cpu::expert_layout().blob_bytes(0) != strata::kernels::cpu::expert_layout().blob_bytes(1),
                    "the native fixture does not vary the blob size");
            run_pack("native", dir.path, bytes, experts, true);
        }
#endif
        std::printf("slot_fill_parity: PASS (%d checks)\n", g_checks);
        return 0;
    } catch (const std::exception& e) {
        std::printf("slot_fill_parity: FAIL: %s (after %d checks)\n", e.what(), g_checks);
        return 1;
    }
}
