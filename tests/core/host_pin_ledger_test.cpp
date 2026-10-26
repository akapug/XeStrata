// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The host-pin ledger (gpu.hpp): STRATA_HOST_PIN_MAX_GIB is read as the cap, every pool is counted and itemized,
// an allocation past the cap is refused before anything is pinned and says why, and freeing gives the bytes back.
// The ledger part needs no GPU; the alloc_host part runs when a device answers.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

namespace g = strata::gpu;
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;

static bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

int main() {
    setenv("STRATA_HOST_PIN_MAX_GIB", "0.5", 1);   // read once, at the first use
    CHECK(g::host_pin_max_bytes() == GiB / 2);
    CHECK(g::kHostPinMaxGibDefault == 0);

    // no cap: everything is admitted and counted under its pool
    g::host_pin_set_max_bytes(0);
    const int a = 0, b = 0, c = 0, d = 0;
    CHECK(g::host_pin_reserve(GiB, "kv-stream"));
    g::host_pin_commit(&a, GiB, "kv-stream");
    CHECK(g::host_pin_reserve(GiB, "kv-stream"));
    g::host_pin_commit(&b, GiB, "kv-stream");
    CHECK(g::host_pin_reserve(3 * GiB, "expert-complement"));
    g::host_pin_commit(&c, 3 * GiB, "expert-complement");
    CHECK(g::host_pinned_bytes() == 5 * GiB);
    g::host_pin_detail("expert-complement", "test note");
    std::string r = g::host_pin_report();
    std::printf("%s\n", r.c_str());
    CHECK(has(r, "host pins 5.00 GiB (STRATA_HOST_PIN_MAX_GIB off)"));
    CHECK(has(r, "expert-complement 3.00 GiB = 1 x 3.00 GiB [test note]"));
    CHECK(has(r, "kv-stream 2.00 GiB = 2 x 1.00 GiB"));
    CHECK(r.find("expert-complement") < r.find("kv-stream"));   // largest first

    // a cap: refused before pinning, the message names the setting, nothing is counted
    g::host_pin_set_max_bytes(5 * GiB + 256 * MiB);
    CHECK(!g::host_pin_reserve(512 * MiB, "prefill"));
    CHECK(has(g::last_error(), "STRATA_HOST_PIN_MAX_GIB") && has(g::last_error(), "prefill"));
    CHECK(g::host_pinned_bytes() == 5 * GiB);
    CHECK(g::host_pin_reserve(256 * MiB, "prefill"));   // exactly to the cap
    CHECK(!g::host_pin_reserve(1, "prefill"));
    g::host_pin_cancel(256 * MiB);                      // an allocation the driver refused gives its reservation back
    CHECK(g::host_pinned_bytes() == 5 * GiB);

    // freeing: a committed block's bytes come back, an unknown pointer is a no-op
    CHECK(g::host_pin_release(&a) == GiB);
    CHECK(g::host_pin_release(&a) == 0);
    CHECK(g::host_pin_release(&d) == 0);
    CHECK(g::host_pinned_bytes() == 4 * GiB);
    CHECK(has(g::host_pin_report(), "kv-stream 1.00 GiB = 1 x 1.00 GiB"));
    CHECK(g::host_pin_release(&b) == GiB && g::host_pin_release(&c) == 3 * GiB);
    CHECK(g::host_pinned_bytes() == 0);

    // alloc_host and free through the ledger, on a device
    bool device = false;
    try {
        strata::core::Runtime::get();
        device = true;
    } catch (const std::exception& e) {
        std::printf("host_pin_ledger_test: no device (%s): the alloc_host part is skipped\n", e.what());
    }
    if (device) {
        g::host_pin_set_max_bytes(96 * MiB);
        void* p = nullptr;
        CHECK(g::alloc_host(&p, 64 * MiB, "test-pool") && p != nullptr);
        CHECK(g::host_pinned_bytes() == 64 * MiB);
        CHECK(has(g::host_pin_report(), "test-pool 64 MiB = 1 x 64 MiB"));
        void* q = nullptr;
        CHECK(!g::alloc_host(&q, 64 * MiB, "test-pool") && q == nullptr);
        CHECK(has(g::last_error(), "STRATA_HOST_PIN_MAX_GIB"));
        CHECK(g::host_pinned_bytes() == 64 * MiB);
        void* small = g::alloc_host(16 * MiB);   // the default pool
        CHECK(small != nullptr && has(g::host_pin_report(), "other 16 MiB"));
        g::free(p);
        g::free(small);
        CHECK(g::host_pinned_bytes() == 0);
        g::host_pin_set_max_bytes(0);
    }
    std::printf("host_pin_ledger_test: %s (%d failures)\n", fails == 0 ? "ok" : "FAILED", fails);
    return fails == 0 ? 0 : 1;
}
