// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// homelab slot park (include/strata/program/slot_park.hpp): the settings, which admissions overwrite a slot's
// conversation, and which slots are worth writing out.
#include "strata/program/slot_park.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program::slot_park;

namespace {
int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}
} // namespace

int main() {
    Settings s;
    std::string why;
    check(parse_settings(nullptr, nullptr, s, why) && !s.on && s.min_tokens == kMinTokensDefault, "unset: off, 2048");
    check(parse_settings("1", "", s, why) && s.on && s.min_tokens == 2048, "on with the default minimum");
    check(parse_settings("1", "512", s, why) && s.min_tokens == 512, "a minimum");
    check(!parse_settings("2", nullptr, s, why) && !s.on && why.find("STRATA_SLOT_PARK ") == 0, "2 refused, off");
    check(!parse_settings("1", "0", s, why) && !s.on && why.find("MIN_TOKENS") != std::string::npos, "0 tokens refused");
    check(!parse_settings("1", "12k", s, why) && !s.on, "junk refused");

    const std::vector<int32_t> held = {1, 2, 3, 4, 5, 6};
    const std::vector<int32_t> turn = {1, 2, 3};
    check(continued(held, &turn, std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7}), "the next turn goes on with all of it");
    check(continued(held, &turn, std::vector<int64_t>{1, 2, 3, 9, 9}), "a turn re-rendered from the turn checkpoint");
    check(!continued(held, &turn, std::vector<int64_t>{1, 2, 9}), "another conversation sharing a start overwrites it");
    check(!continued(held, nullptr, std::vector<int64_t>{1, 2, 3, 9}), "without a checkpoint only all of it counts");
    check(!continued(held, &turn, std::vector<int64_t>{7, 8}), "an unrelated conversation overwrites it");
    check(continued(held, &turn, std::vector<int32_t>{1, 2, 3, 4, 5, 6}), "the same tokens (int32 prompt)");
    check(!continued({}, nullptr, std::vector<int64_t>{1}), "an empty slot is never continued");

    Settings on;
    on.on = true;
    Slot x;
    x.cached = true;
    x.tokens = 5000;
    check(parkable(on, true, x), "an idle cached 5000-token conversation is parked");
    check(!parkable(on, false, x), "not without --conversation-save");
    check(!parkable(Settings{}, true, x), "not with the lever off");
    for (int k = 0; k < 6; ++k) {
        Slot y = x;
        if (k == 0) y.active = true;
        if (k == 1) y.cached = false;
        if (k == 2) y.img = true;
        if (k == 3) y.tainted = true;
        if (k == 4) y.partial = true;
        if (k == 5) y.lent = 1;
        check(!parkable(on, true, y), "not running, uncached, with pictures, tainted, partial or lending");
    }
    x.tokens = 2047;
    check(!parkable(on, true, x), "not below the minimum");

    AsyncSettings a;
    check(parse_async_settings(nullptr, nullptr, nullptr, a, why) && !a.on && a.jobs == kQueueJobsDefault &&
              a.mib == kQueueMibDefault, "background park unset: off, 2 jobs, 16384 MiB");
    check(parse_async_settings("1", "4", "8192", a, why) && a.on && a.jobs == 4 && a.mib == 8192, "on, 4 jobs, 8 GiB");
    check(parse_async_settings("1", "", "", a, why) && a.on && a.jobs == 2, "empty keeps the defaults");
    check(!parse_async_settings("yes", nullptr, nullptr, a, why) && !a.on && why.find("STRATA_SLOT_PARK_ASYNC") == 0,
          "a junk switch is refused by name, off");
    check(!parse_async_settings("1", "0", nullptr, a, why) && !a.on && why.find("STRATA_SLOT_PARK_QUEUE ") == 0,
          "an empty queue is refused by name");
    check(!parse_async_settings("1", "65", nullptr, a, why) && !a.on, "a queue over 64 is refused");
    check(!parse_async_settings("1", nullptr, "0", a, why) && !a.on && why.find("STRATA_SLOT_PARK_QUEUE_MIB") == 0,
          "a zero byte bound is refused by name");
    PrefetchSettings p;
    check(parse_prefetch_settings(nullptr, nullptr, nullptr, nullptr, p, why) && !p.on && p.jobs == 2 &&
              p.mib == 16384 && p.timeout_ms == 30000, "prefetch unset: off, 2 reads, 16384 MiB, 30 s");
    check(parse_prefetch_settings("1", "8", "4096", "500", p, why) && p.on && p.jobs == 8 && p.mib == 4096 &&
              p.timeout_ms == 500, "prefetch on, 8 reads, 4 GiB, 500 ms");
    check(!parse_prefetch_settings("on", nullptr, nullptr, nullptr, p, why) && !p.on &&
              why.find("STRATA_SLOT_PREFETCH ") == 0, "a junk prefetch switch is refused by name, off");
    check(!parse_prefetch_settings("1", "0", nullptr, nullptr, p, why) && !p.on &&
              why.find("STRATA_SLOT_PREFETCH_MAX") == 0, "no reads held is refused by name");
    check(!parse_prefetch_settings("1", "65", nullptr, nullptr, p, why) && !p.on, "over 64 reads is refused");
    check(!parse_prefetch_settings("1", nullptr, "0", nullptr, p, why) && why.find("STRATA_SLOT_PREFETCH_MIB") == 0,
          "a zero byte bound is refused by name");
    check(!parse_prefetch_settings("1", nullptr, nullptr, "99", p, why) && why.find("STRATA_SLOT_PREFETCH_TIMEOUT_MS") == 0,
          "a timeout under 100 ms is refused by name");
    check(!parse_prefetch_settings("1", nullptr, nullptr, "3600001", p, why) && !p.on, "a timeout over an hour is refused");
    check(restore_reads_file(5000, 2500), "a file deeper than memory holds: read it (hold the admission back)");
    check(!restore_reads_file(5000, 5000) && !restore_reads_file(5000, 6000), "memory holds as much or more: admit now");
    check(!restore_reads_file(0, 0), "no file: admit now");
    std::printf("slot_park_test: %d checks passed\n", checks);
    return 0;
}
