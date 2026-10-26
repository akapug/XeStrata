// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// homelab: a verify window of up to MAXT (32) tokens through the CPU expert pool, as expert_pool_dispatch_multi groups
// it (one job an expert, its tokens in routing order), bit for bit against the same tokens as two 16-token windows
// (the pool's capacity before homelab) and against each token alone.  Synthetic ggml-quantized experts, every native
// format ggml quantizes without an importance matrix, and the legacy S2 pack where the CPU runs its AVX-512 kernel.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
constexpr int kWindow = cpu::MAXT, kHalf = 16, kK = 10, kExperts = 24;
static_assert(kWindow == 32, "the 32-row window of homelab");

// ids[t * kK + j]: kK distinct experts a token; expert 0 by every token and expert 1 by the first 21, so one job holds
// the whole window and one crosses the two halves' seam
std::vector<int> routing() {
    std::mt19937 rng(353);
    std::vector<int> ids((size_t) kWindow * kK);
    for (int t = 0; t < kWindow; ++t) {
        std::vector<int> pick{0};
        if (t < 21) pick.push_back(1);
        while ((int) pick.size() < kK) {
            const int e = 2 + (int) (rng() % (kExperts - 2));
            bool dup = false;
            for (int p : pick) dup = dup || p == e;
            if (!dup) pick.push_back(e);
        }
        for (int j = 0; j < kK; ++j) ids[(size_t) (t * kK + j)] = pick[(size_t) ((j + t) % kK)];
    }
    return ids;
}

// tokens [t0, t0 + n) as one window: a job per distinct expert in routing order, as the dispatch builds them; `act`
// sets a job's token activation, `run` the pool phase
template <class Act, class Run>
void window(const std::vector<int>& ids, int t0, int n, const std::vector<const uint8_t*>& blobs, std::vector<float>& out,
            int width, Act act, Run run) {
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<int> job_of(kExperts, -1);
    for (int t = t0; t < t0 + n; ++t)
        for (int j = 0; j < kK; ++j) {
            const int e = ids[(size_t) (t * kK + j)];
            if (job_of[(size_t) e] < 0) {
                job_of[(size_t) e] = (int) jobs.size();
                jobs.emplace_back();
                jobs.back().blob = blobs[(size_t) e];
            }
            cpu::ExpertJobMulti& jb = jobs[(size_t) job_of[(size_t) e]];
            act(jb, t);
            jb.out[jb.nt] = out.data() + (size_t) (t * kK + j) * (size_t) width;
            ++jb.nt;
        }
    run(jobs.data(), (int) jobs.size());
}

// the window whole, as two halves and token by token, through pools of 1 and 3 workers with and without the host:
// every row the same bits.  Returns the rows compared, or -1 on a mismatch.
template <class Act, class Run>
long long check(const char* name, const std::vector<const uint8_t*>& blobs, int width, Act act, Run run) {
    const std::vector<int> ids = routing();
    const size_t rows = (size_t) kWindow * kK;
    long long checked = 0;
    for (bool host : {false, true})
        for (int workers : {1, 3}) {
            cpu::ExpertPool pool(workers, false, host);
            auto go = [&](cpu::ExpertJobMulti* j, int n) { run(pool, j, n); };
            std::vector<float> whole(rows * (size_t) width, NAN), halves = whole, single = whole;
            window(ids, 0, kWindow, blobs, whole, width, act, go);
            window(ids, 0, kHalf, blobs, halves, width, act, go);
            window(ids, kHalf, kWindow - kHalf, blobs, halves, width, act, go);
            for (int t = 0; t < kWindow; ++t) window(ids, t, 1, blobs, single, width, act, go);
            for (size_t r = 0; r < rows; ++r) {
                const float* w = whole.data() + r * (size_t) width;
                for (int i = 0; i < width; ++i)
                    if (!std::isfinite(w[i])) {
                        std::fprintf(stderr, "FAIL %s workers=%d host=%d row %zu: not finite\n", name, workers, host, r);
                        return -1;
                    }
                for (const auto* ref : {&halves, &single})
                    if (std::memcmp(w, ref->data() + r * (size_t) width, (size_t) width * sizeof(float)) != 0) {
                        std::fprintf(stderr, "MISMATCH %s workers=%d host=%d token %zu slot %zu: the 32-token window "
                                             "differs from %s\n", name, workers, host, r / kK, r % kK,
                                     ref == &halves ? "two 16-token windows" : "the token alone");
                        return -1;
                    }
                checked += 2;
            }
        }
    std::printf("%s: %lld rows bitwise (32-token window vs two halves and vs one token)\n", name, checked);
    return checked;
}

long long native(int type) {
    const char* tn = ggml_type_name((ggml_type) type);
    if (ggml_quantize_requires_imatrix((ggml_type) type)) {
        std::printf("%s: skipped (quantizing needs an importance matrix)\n", tn);
        return 0;
    }
    cpu::NativeFmt f;
    std::string err;
    int64_t ff = 0;
    for (int64_t want : {(int64_t) cpu::FF, (int64_t) 512})
        if (ff == 0 && cpu::native_fmt(type, type, cpu::H, want, f, err)) ff = want;
    if (ff == 0) {
        std::printf("%s: skipped (%s)\n", tn, err.c_str());
        return 0;
    }
    std::vector<std::vector<uint8_t>> store(kExperts, std::vector<uint8_t>(f.bytes));
    std::vector<const uint8_t*> blobs;
    std::vector<float> w((size_t) cpu::H * (size_t) ff);
    for (int e = 0; e < kExperts; ++e) {
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.02f * std::sin((float) i * 0.17f + 0.9f * (float) e);
        uint8_t* b = store[(size_t) e].data();
        ggml_quantize_chunk((ggml_type) type, w.data(), b, 0, ff, cpu::H, nullptr);
        ggml_quantize_chunk((ggml_type) type, w.data(), b + f.up_off, 0, ff, cpu::H, nullptr);
        ggml_quantize_chunk((ggml_type) type, w.data(), b + f.down_off, 0, cpu::H, ff, nullptr);
        blobs.push_back(b);
    }
    std::vector<cpu::NativeAct> act(kWindow);
    std::vector<float> x(cpu::H);
    for (int t = 0; t < kWindow; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[(size_t) i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[(size_t) t].bytes);
    }
    return check(
        tn, blobs, (int) f.n_embd, [&](cpu::ExpertJobMulti& jb, int t) { jb.nact[jb.nt] = act[(size_t) t].bytes; },
        [&](cpu::ExpertPool& p, cpu::ExpertJobMulti* j, int n) { p.run_split_multi_native(f, j, n); });
}

long long legacy() {
    if (const cpu::CpuFeatures f = cpu::cpu_features(); !f.usable()) {
        std::printf("legacy S2: skipped (%s)\n", f.reason());
        return 0;
    }
    std::mt19937 rng(9);
    std::vector<std::vector<uint8_t>> store(kExperts, std::vector<uint8_t>(cpu::BLOB));
    std::vector<const uint8_t*> blobs;
    for (auto& b : store) {
        for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();   // codes
        for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {                // fp16 scales ~0.004-0.016
            const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
            std::memcpy(&b[i], &h, 2);
        }
        blobs.push_back(b.data());
    }
    std::vector<cpu::ActQ> act(kWindow);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(cpu::H);
    for (auto& a : act) {
        for (float& v : x) v = nd(rng);
        cpu::act_quant_q8_1(x.data(), cpu::H, a);
    }
    return check(
        "legacy S2", blobs, cpu::H, [&](cpu::ExpertJobMulti& jb, int t) { jb.act[jb.nt] = &act[(size_t) t]; },
        [](cpu::ExpertPool& p, cpu::ExpertJobMulti* j, int n) { p.run_split_multi(j, n); });
}
}  // namespace

int main() {
    long long total = 0;
    for (long long n : {native(GGML_TYPE_Q4_0), native(GGML_TYPE_Q8_0), native(GGML_TYPE_Q2_K), native(GGML_TYPE_Q4_K),
                        native(GGML_TYPE_Q6_K), native(GGML_TYPE_IQ3_XXS), native(GGML_TYPE_IQ3_S),
                        native(GGML_TYPE_IQ4_XS), native(GGML_TYPE_IQ4_NL), legacy()}) {
        if (n < 0) return 1;
        total += n;
    }
    if (total == 0) {
        std::printf("pool_window_test: nothing ran\n");
        return 1;
    }
    std::printf("pool_window_test: OK, %lld rows bitwise\n", total);
    return 0;
}
