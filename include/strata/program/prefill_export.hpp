// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <atomic>
#include <charconv>
#include <cstdint>
#include <functional>
#include <utility>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace strata::program {

// Exact coordinator tokens, not text to tokenize again. The final token belongs to the decoder; the draft's
// teacher-forced prefix nevertheless depends on it. Paths run to the end of the line, as in SAVE / RESTORE.
struct PrefillRequest {
    bool restore = false;
    std::vector<int64_t> ids;
    std::string path;
};
inline bool parse_prefill_request(const std::string& line, PrefillRequest& out, std::string& why) {
    const bool restore = line.rfind("RESTORE_PREFILL ", 0) == 0;
    const size_t start = restore ? 16 : 8;
    auto fail = [&](const char* text) { why = text; return false; };
    if (!restore && line.rfind("PREFILL ", 0) != 0) return fail("expected PREFILL or RESTORE_PREFILL");
    const size_t gap = line.find(' ', start);
    if (gap == std::string::npos || gap == start) return fail("expected token list and path");
    PrefillRequest parsed;
    parsed.restore = restore;
    size_t at = start;
    while (at < gap) {
        const size_t comma = line.find(',', at);
        const size_t end = comma == std::string::npos || comma > gap ? gap : comma;
        int32_t token = -1;
        const auto result = std::from_chars(line.data() + at, line.data() + end, token);
        if (result.ec != std::errc() || result.ptr != line.data() + end || token < 0)
            return fail("token IDs must be nonnegative int32 integers");
        parsed.ids.push_back(token);
        if (end == gap) break;
        at = end + 1;
        if (at == gap) return fail("empty token ID");
    }
    if (parsed.ids.size() < 2) return fail("prefill needs a prefix and a retained final token");
    const size_t path = line.find_first_not_of(' ', gap);
    if (path == std::string::npos || line.find_first_of("\r\n\0", 0, 3) != std::string::npos)
        return fail("missing path or a control character in the request");
    parsed.path = line.substr(path);
    out = std::move(parsed);
    why.clear();
    return true;
}

// The caller resolves the same context reserve as GEN before handing over any session/device state.
struct PrefillCapabilities {
    bool mtp = false, split = false, batch = false, prompt_cache = false;
    int64_t prompt_limit = 0, vocabulary = 0;
};
inline bool admit_prefill_request(const PrefillRequest& request, const PrefillCapabilities& c, std::string& why) {
    if (!c.mtp || c.split || c.batch || !c.prompt_cache) {
        why = "prefill sessions need --mtp --prompt-cache > 0, no --batch or --layer-split";
        return false;
    }
    if (c.prompt_limit < 2 || request.ids.size() < 2 || request.ids.size() > (uint64_t) c.prompt_limit) {
        why = "prefill prompt exceeds the context or has no retained final token";
        return false;
    }
    for (int64_t token : request.ids) if (token < 0 || token >= c.vocabulary) {
        why = "prefill token ID is outside the vocabulary";
        return false;
    }
    why.clear();
    return true;
}

// A mounted draft is only reusable by the exact full prompt it was teacher-forced with. While bound it must not
// be parked or downgraded into an ordinary session. Interrupted consumption invalidates all live/checkpoint reuse.
class PrefillBinding {
public:
    void bind(const std::vector<int64_t>& prompt) { prompt_ = prompt; }
    void clear() { prompt_.reset(); }
    bool active() const { return prompt_.has_value(); }
    bool discard_for(const std::vector<int64_t>& prompt, bool exporting) {
        if (!prompt_ || (!exporting && *prompt_ == prompt)) return false;
        clear();
        return true;
    }
    bool finish(bool interrupted) {
        const bool discard = active() && interrupted;
        clear();
        return discard;
    }
private:
    std::optional<std::vector<int64_t>> prompt_;
};

// The request owner is serial, but STOP arrives on stdin's thread. Cancellation and the actual persistence rename
// share this lock. A STOP before publication keeps the old artifact; one after publication cannot retract it.
// Holding an earlier generation never authorizes publication for a later request, even if its STOP flag was reset.
class RequestPublication {
public:
    std::atomic<bool> stopped{false};
    uint64_t begin() {
        const std::lock_guard<std::mutex> lock(mu_);
        stopped.store(false);
        return ++generation_;
    }
    void cancel() {
        const std::lock_guard<std::mutex> lock(mu_);
        stopped.store(true);
    }
    bool publish(uint64_t generation, const std::function<bool()>& action, std::string& why) {
        const std::lock_guard<std::mutex> lock(mu_);
        if (generation != generation_ || stopped.load()) {
            why = "prefill export cancelled or superseded before publication";
            return false;
        }
        return action();
    }
private:
    std::mutex mu_;
    uint64_t generation_ = 0;
};

} // namespace strata::program
