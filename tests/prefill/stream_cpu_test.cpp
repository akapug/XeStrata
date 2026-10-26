// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/prefill/stream_cpu.hpp"

#include <cstdio>
#include <limits>
#include <vector>

using namespace strata::prefill;

static int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); } } while (0)

int main() {
    const char* invalid[] = {nullptr, "", "auto", "nan", "inf", "-0.1", "1.01", "0.2junk", "  "};
    for (const char* v : invalid)
        CHECK(stream_cpu_share_request(v) == kStreamCpuShareDefault);
    CHECK(stream_cpu_share_request("0") == 0.0);
    CHECK(stream_cpu_share_request("1") == 1.0);
    CHECK(stream_cpu_share_request("0.25") == 0.25);
    for (int bits = 0; bits < 16; ++bits)
        CHECK(stream_cpu_owner(bits & 1, bits & 2, bits & 4, bits & 8) == (bits == 7));
    for (size_t n : {0u, 1u, 7u, 512u, 4096u}) {
        CHECK(stream_cpu_take(n, 0) == 0);
        CHECK(stream_cpu_take(n, 1) == n);
        CHECK(stream_cpu_take(n, 0.5) == n / 2);
        CHECK(stream_cpu_take(n, -1) == 0);
        CHECK(stream_cpu_take(n, std::numeric_limits<double>::quiet_NaN()) == 0);
    }
    const uint8_t blob = 1;
    for (unsigned mask = 0; mask < 256; ++mask)
        for (double share : {0.0, 0.25, 0.5, 1.0}) {
            std::vector<const uint8_t*> eligible(8, nullptr);
            size_t count = 0;
            for (size_t e = 0; e < eligible.size(); ++e)
                if (mask & (1u << e)) { eligible[e] = &blob; ++count; }
            const auto original = eligible;
            const size_t take = stream_cpu_partition(eligible, share);
            CHECK(take == (size_t) std::floor((double) count * share));
            size_t selected = 0;
            for (size_t e = 0; e < eligible.size(); ++e) {
                CHECK(eligible[e] == nullptr || eligible[e] == original[e]);
                const bool cpu = eligible[e] != nullptr, gpu = original[e] != nullptr && !cpu;
                CHECK((cpu ? 1 : 0) + (gpu ? 1 : 0) == (original[e] != nullptr ? 1 : 0));
                selected += cpu;
            }
            CHECK(selected == take);
        }
    // Count every row exactly once, including a real-read-size hot expert far above MAXT.
    // Expert output offsets are contiguous; each job's pointers map back to the same token row.
    for (size_t limit : {1u, 8u, 16u, 32u})
        for (size_t rows : {0u, 1u, 15u, 16u, 17u, 96u, 4096u, 34000u}) {
            std::vector<int> seen(rows, 0);
            size_t jobs = 0;
            for (size_t r0 = 0; r0 < rows;) {
                const size_t n = stream_cpu_rows(rows - r0, limit);
                CHECK(n > 0 && n <= limit);
                for (size_t r = 0; r < n; ++r) ++seen[r0 + r];
                r0 += n;
                ++jobs;
            }
            CHECK(jobs == (rows + limit - 1) / limit);
            for (int count : seen) CHECK(count == 1);
        }
    // Exercise the actual native-job builder with routed rows, GPU/CPU complements and weighted combine.
    // Synthetic arithmetic isolates row ownership; this is not native expert numerical or device parity.
    for (size_t tokens : {1u, 17u, 4096u})
        for (double share : {0.0, 0.25, 1.0}) {
            constexpr size_t experts = 8, topk = 4, width = 3;
            uint8_t blobs[experts];
            std::vector<const uint8_t*> chosen(experts);
            std::vector<size_t> ids(tokens * topk), count(experts, 0), off(experts), fill;
            for (size_t e = 0; e < experts; ++e) { blobs[e] = (uint8_t) (e + 1); chosen[e] = &blobs[e]; }
            stream_cpu_partition(chosen, share);
            for (size_t i = 0; i < ids.size(); ++i) { ids[i] = (i / topk + i % topk) % experts; ++count[ids[i]]; }
            size_t rows_gpu = 0, at = 0;
            for (size_t e = 0; e < experts; ++e)
                if (!chosen[e]) { off[e] = at; at += count[e]; }
            rows_gpu = at;
            for (size_t e = 0; e < experts; ++e)
                if (chosen[e]) { off[e] = at; at += count[e]; }
            CHECK(at == ids.size());
            fill = off;
            std::vector<size_t> slot(ids.size()), source(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) { slot[i] = fill[ids[i]]++; source[slot[i]] = i / topk; }
            std::vector<float> device(ids.size() * width, -1), cpu((ids.size() - rows_gpu) * width, -1);
            std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
            for (size_t e = 0; e < experts; ++e) {
                if (chosen[e] && count[e] > 0) {
                    stream_cpu_jobs(jobs, chosen[e], count[e], cpu.data() + (off[e] - rows_gpu) * width, width);
                    continue;
                }
                for (size_t r = 0; r < count[e]; ++r)
                    for (size_t c = 0; c < width; ++c)
                        device[(off[e] + r) * width + c] = (float) (blobs[e] * (source[off[e] + r] + 1) + c);
            }
            std::vector<int> written(ids.size() - rows_gpu, 0);
            for (const auto& j : jobs) {
                CHECK(j.nt > 0 && j.nt <= strata::kernels::cpu::MAXT);
                for (int r = 0; r < j.nt; ++r) {
                    const size_t row = (size_t) (j.out[r] - cpu.data()) / width;
                    CHECK(row < written.size());
                    ++written[row];
                    for (size_t c = 0; c < width; ++c)
                        j.out[r][c] = (float) (*j.blob * (source[rows_gpu + row] + 1) + c);
                }
            }
            for (int n : written) CHECK(n == 1);
            std::copy(cpu.begin(), cpu.end(), device.begin() + rows_gpu * width);
            for (size_t t = 0; t < tokens; ++t)
                for (size_t c = 0; c < width; ++c) {
                    float combined = 0, expected = 0;
                    for (size_t k = 0; k < topk; ++k) {
                        const size_t i = t * topk + k;
                        const float weight = (float) (k + 1) / 16;
                        combined += weight * device[slot[i] * width + c];
                        expected += weight * (float) (blobs[ids[i]] * (t + 1) + c);
                    }
                    CHECK(combined == expected);
                }
        }
    std::printf("stream_cpu_test: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
