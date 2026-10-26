// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/fuse_small_parity.cpp - homelab's small-kernel lane: STRATA_GDN_AB_FORM, STRATA_MMVF_WIDE and
// STRATA_ROUTER_FUSE against the established calls, bitwise (memcmp), each also against FP64, and the time of each.
//
//   * gdn alpha/beta (n_embd 2560, 48 value heads): gdn_ab_multi in every form, and gdn_conv_l2_slots with the units in
//     its launch (forms 4..6; its conv outputs and committed histories too), 1..16 rows, against form 0;
//   * the BF16 GEMVs of a window (2560 -> 512 the router, 2560 -> 128 and 2560 -> 512 the QSA indexer's): the wide form
//     (1, 2, 4 rows a work-group), plain and with the aux row, 2..16 rows, against the established calls;
//   * the router in one launch (logits, the shared expert's gate, the top-10 ids and weights) against the GEMV + aux
//     + native_router_top10_multi, 2..16 rows, called directly and replayed from a captured graph (twice: the arrival
//     counter must be back at 0), with a second queue between (its own counters);
//   * STRATA_HC_PREFETCH's l2_prefetch launch ahead of a kernel reading the prefetched weights: that kernel's bits
//     unchanged (direct and from a graph replayed twice), the weights untouched;
//   * the time a call (a graph of `reps` calls a round, the forms' rounds in turns; fastest / median) at 1, 7, 16 rows.
//
//     build/fuse_small_parity [reps] [rounds]      reps 0: parity only
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/l2_prefetch.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"
#include "strata/core/gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {
namespace P = strata::parity;
namespace k = strata::kernels;
namespace G = strata::gpu;

template <typename T> T* up(const std::vector<T>& h) {
    T* p = static_cast<T*>(P::alloc_bytes(h.size() * sizeof(T) + 64));
    P::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}
template <typename T> T* buf(size_t n) { return static_cast<T*>(P::alloc_bytes(n * sizeof(T) + 64)); }
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    P::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}
template <typename T> void poison(T* d, size_t n) {
    std::vector<uint8_t> h(n * sizeof(T), 0xff);
    P::copy_bytes_in(d, h.data(), h.size());
}
uint16_t bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (uint16_t) (u >> 16);
}
float f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

int g_checks = 0, g_fail = 0;
void same(const char* what, const void* a, const void* b, size_t bytes) {
    ++g_checks;
    if (std::memcmp(a, b, bytes) == 0) return;
    ++g_fail;
    const auto* x = static_cast<const uint8_t*>(a);
    const auto* y = static_cast<const uint8_t*>(b);
    size_t i = 0;
    while (x[i] == y[i]) ++i;
    std::printf("FAIL %s: differs at byte %zu of %zu\n", what, i, bytes);
}
// |got - ref| <= atol + rtol |ref| everywhere
void near64(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double atol, double rtol) {
    ++g_checks;
    double worst = 0;
    size_t wi = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = std::fabs(got[i] - ref[i]) / (atol + rtol * std::fabs(ref[i]));
        if (!(e <= worst)) { worst = e; wi = i; }
    }
    if (worst <= 1.0) return;
    ++g_fail;
    std::printf("FAIL %s vs FP64: element %zu got %.9g ref %.9g (%.2fx the tolerance)\n", what, wi, got[wi], ref[wi], worst);
}

// a graph of `reps` calls of f a round, every arm's rounds in turns: the fastest and median us a call
struct Timed {
    std::string name;
    std::function<void(void*)> f;
    std::vector<double> us = {};
    G::Graph* g = nullptr;
};
void time_arms(std::vector<Timed>& arms, int reps, int rounds) {
    G::Stream st = G::stream_create();
    for (auto& a : arms) {
        a.f(st);   // warm (and any one-time allocation) outside the capture
        G::stream_sync(st);
        if (!G::begin_capture(st)) throw std::runtime_error("begin_capture");
        for (int i = 0; i < reps; ++i) a.f(st);
        if (!G::end_capture(st, &a.g)) throw std::runtime_error("end_capture");
    }
    for (int r = 0; r <= rounds; ++r)
        for (auto& a : arms) {
            G::graph_launch(a.g, st);
            G::stream_sync(st);
            const auto t0 = std::chrono::steady_clock::now();
            G::graph_launch(a.g, st);
            G::stream_sync(st);
            const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            if (r > 0) a.us.push_back(t);
        }
    for (auto& a : arms) {
        std::sort(a.us.begin(), a.us.end());
        std::printf("  %-48s %8.1f / %8.1f\n", a.name.c_str(), a.us[0], a.us[a.us.size() / 2]);
        G::graph_destroy(a.g);
    }
    G::stream_destroy(st);
}

}  // namespace

int main(int argc, char** argv) try {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20, rounds = argc > 2 ? std::atoi(argv[2]) : 7;
    constexpr int N = 2560, HV = 48, HK = 16, S = 128, C = 2 * S * HK + S * HV, NE = 512, MT = k::kVerifyMaxT;
    void* s = P::stream();
    std::mt19937 rng(353);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);

    // ============ the window's activations (the hyper-connection read's mixed: about unit scale)
    std::vector<float> x((size_t) MT * N);
    for (auto& v : x) v = nd(rng);
    const float* dx = up(x);

    // ============ gdn alpha/beta
    {
        std::vector<uint16_t> wa((size_t) HV * N), wb((size_t) HV * N);
        for (auto& v : wa) v = bf16(0.02f * nd(rng));
        for (auto& v : wb) v = bf16(0.02f * nd(rng));
        std::vector<float> dt(HV), sa(HV);
        for (auto& v : dt) v = nd(rng);
        for (auto& v : sa) v = -0.1f - ud(rng);
        const uint16_t *dwa = up(wa), *dwb = up(wb);
        const float *ddt = up(dt), *dsa = up(sa);
        float *g0 = buf<float>((size_t) MT * HV), *b0 = buf<float>((size_t) MT * HV);
        float *g1 = buf<float>((size_t) MT * HV), *b1 = buf<float>((size_t) MT * HV);
        // the conv's inputs: one row a slot, 16 slots, committed (the batch window's self-commit)
        std::vector<float> qkv((size_t) MT * C), cw((size_t) C * 4), hist0((size_t) MT * C * 3);
        for (auto& v : qkv) v = nd(rng);
        for (auto& v : cw) v = 0.3f * nd(rng);
        for (auto& v : hist0) v = nd(rng);
        const float *dq = up(qkv), *dcw = up(cw);
        float* dh0 = buf<float>((size_t) MT * C);
        float* dh1 = buf<float>((size_t) MT * C);
        float* hist_a = buf<float>(hist0.size());
        float* hist_b = buf<float>(hist0.size());
        const auto table = [&](float* hist, int T) {
            k::GdnSlotTable t;
            for (int j = 0; j < T; ++j) { t.hist[j] = hist + (size_t) j * C * 3; t.state[j] = nullptr; t.first[j] = j; t.ntok[j] = 1; }
            t.n = T;
            return t;
        };
        for (int T = 1; T <= MT; ++T) {
            k::gdn_ab_set_form(0);
            poison(g0, (size_t) MT * HV); poison(b0, (size_t) MT * HV);
            k::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, g0, b0, N, HV, T, s);
            P::copy_bytes_in(hist_a, hist0.data(), hist0.size() * 4);
            k::gdn_conv_l2_slots(table(hist_a, T), dq, dcw, dh0, C, 2 * HK, 1e-6f, s, true);
            P::sync();
            const auto rg = down(g0, (size_t) T * HV), rb = down(b0, (size_t) T * HV);
            const auto rh = down(dh0, (size_t) T * C), rhist = down(hist_a, hist0.size());
            // FP64: gate = softplus(a + dt) * ssm_a (x the bf16 weights), beta = sigmoid(b)
            std::vector<double> eg((size_t) T * HV), eb((size_t) T * HV);
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < HV; ++r) {
                    double a = 0, b = 0;
                    for (int i = 0; i < N; ++i) {
                        a += (double) f32(wa[(size_t) r * N + i]) * x[(size_t) t * N + i];
                        b += (double) f32(wb[(size_t) r * N + i]) * x[(size_t) t * N + i];
                    }
                    const double v = a + dt[r];
                    eg[(size_t) t * HV + r] = (v > 20 ? v : std::log1p(std::exp(v))) * sa[r];
                    eb[(size_t) t * HV + r] = 1 / (1 + std::exp(-b));
                }
            if (T == 1 || T == 7 || T == MT) {
                near64(("gdn gate T" + std::to_string(T)).c_str(), rg, eg, 1e-5, 1e-4);
                near64(("gdn beta T" + std::to_string(T)).c_str(), rb, eb, 1e-6, 1e-4);
            }
            for (int form = 1; form < 7; ++form) {
                k::gdn_ab_set_form(form);
                const std::string tag = "gdn ab form " + std::to_string(form) + " T" + std::to_string(T);
                poison(g1, (size_t) MT * HV); poison(b1, (size_t) MT * HV);
                k::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, g1, b1, N, HV, T, s);
                P::sync();
                same((tag + " gate").c_str(), rg.data(), down(g1, (size_t) T * HV).data(), rg.size() * 4);
                same((tag + " beta").c_str(), rb.data(), down(b1, (size_t) T * HV).data(), rb.size() * 4);
                if (form < 4) continue;
                // in the conv's launch
                poison(g1, (size_t) MT * HV); poison(b1, (size_t) MT * HV); poison(dh1, (size_t) MT * C);
                P::copy_bytes_in(hist_b, hist0.data(), hist0.size() * 4);
                k::GdnAbArgs ab{dx, dwa, dwb, ddt, dsa, g1, b1, N, HV, T};
                k::gdn_conv_l2_slots(table(hist_b, T), dq, dcw, dh1, C, 2 * HK, 1e-6f, s, true, &ab);
                P::sync();
                same((tag + " in conv: gate").c_str(), rg.data(), down(g1, (size_t) T * HV).data(), rg.size() * 4);
                same((tag + " in conv: beta").c_str(), rb.data(), down(b1, (size_t) T * HV).data(), rb.size() * 4);
                same((tag + " in conv: h").c_str(), rh.data(), down(dh1, (size_t) T * C).data(), rh.size() * 4);
                same((tag + " in conv: history").c_str(), rhist.data(), down(hist_b, hist0.size()).data(), rhist.size() * 4);
            }
        }
        k::gdn_ab_set_form(0);
        if (reps > 0) {
            std::printf("\ngdn alpha/beta, us a call (fastest / median of %d rounds of %d calls in a graph):\n", rounds, reps);
            for (const int T : {1, 7, 16}) {
                std::vector<Timed> arms;
                for (int form = 0; form < 4; ++form)
                    arms.push_back({"T " + std::to_string(T) + " ab form " + std::to_string(form), [=](void* st) {
                                        k::gdn_ab_set_form(form);
                                        k::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, g1, b1, N, HV, T, st);
                                    }});
                arms.push_back({"T " + std::to_string(T) + " conv alone", [=](void* st) {
                                    k::gdn_conv_l2_slots(table(hist_b, T), dq, dcw, dh1, C, 2 * HK, 1e-6f, st, true);
                                }});
                arms.push_back({"T " + std::to_string(T) + " conv + ab form 0 (two launches)", [=](void* st) {
                                    k::gdn_ab_set_form(0);
                                    k::gdn_conv_l2_slots(table(hist_b, T), dq, dcw, dh1, C, 2 * HK, 1e-6f, st, true);
                                    k::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, g1, b1, N, HV, T, st);
                                }});
                for (int form = 4; form < 7; ++form)
                    arms.push_back({"T " + std::to_string(T) + " conv with ab form " + std::to_string(form), [=](void* st) {
                                        k::gdn_ab_set_form(form);
                                        k::GdnAbArgs ab{dx, dwa, dwb, ddt, dsa, g1, b1, N, HV, T};
                                        k::gdn_conv_l2_slots(table(hist_b, T), dq, dcw, dh1, C, 2 * HK, 1e-6f, st, true, &ab);
                                    }});
                time_arms(arms, reps, rounds);
            }
            k::gdn_ab_set_form(0);
        }
    }

    // ============ the wide BF16 GEMV and the fused router
    {
        const int64_t outs[] = {NE, 128, 512 + 3};   // the router; the indexer's k_proj; a ragged count (RPB tails)
        std::vector<uint16_t> w((size_t) 515 * N), wsg(N);
        for (auto& v : w) v = bf16(0.02f * nd(rng));
        for (auto& v : wsg) v = bf16(0.02f * nd(rng));
        const uint16_t *dw = up(w), *dwsg = up(wsg);
        float *y0 = buf<float>((size_t) MT * 515), *y1 = buf<float>((size_t) MT * 515);
        float *a0 = buf<float>(MT), *a1 = buf<float>(MT);
        int32_t *i0 = buf<int32_t>((size_t) MT * 10), *i1 = buf<int32_t>((size_t) MT * 10);
        float *w0 = buf<float>((size_t) MT * 10), *w1 = buf<float>((size_t) MT * 10);
        for (const int64_t no : outs)
            for (int T = 2; T <= MT; ++T) {
                k::bf16_gemv_set_wide(0);
                poison(y0, (size_t) MT * 515); poison(a0, MT);
                k::bf16_gemv_fp32_mmvf_multi(dx, N, dw, y0, no, N, no, T, s);
                const bool aux0 = k::bf16_gemv_fp32_mmvf_multi_aux(dx, N, dw, y1, no, N, no, T, dwsg, a0, 1, s);
                P::sync();
                const auto ry = down(y0, (size_t) T * no);
                const auto ra = down(a0, T);
                if (no == NE && (T == 2 || T == 7 || T == MT)) {
                    std::vector<double> e((size_t) T * no);
                    for (int t = 0; t < T; ++t)
                        for (int o = 0; o < no; ++o) {
                            double acc = 0;
                            for (int i = 0; i < N; ++i) acc += (double) f32(w[(size_t) o * N + i]) * x[(size_t) t * N + i];
                            e[(size_t) t * no + o] = acc;
                        }
                    near64(("gemv 2560x512 T" + std::to_string(T)).c_str(), ry, e, 1e-5, 1e-4);
                }
                for (const int rpb : {1, 2, 4}) {
                    k::bf16_gemv_set_wide(rpb);
                    const std::string tag = "wide rpb " + std::to_string(rpb) + " n_out " + std::to_string(no) + " T" + std::to_string(T);
                    poison(y1, (size_t) MT * 515); poison(a1, MT);
                    k::bf16_gemv_fp32_mmvf_multi(dx, N, dw, y1, no, N, no, T, s);
                    P::sync();
                    same(tag.c_str(), ry.data(), down(y1, (size_t) T * no).data(), ry.size() * 4);
                    if (!aux0) continue;
                    poison(y1, (size_t) MT * 515);
                    if (!k::bf16_gemv_fp32_mmvf_multi_aux(dx, N, dw, y1, no, N, no, T, dwsg, a1, 1, s))
                        throw std::runtime_error("the wide aux call refused");
                    P::sync();
                    same((tag + " aux: y").c_str(), ry.data(), down(y1, (size_t) T * no).data(), ry.size() * 4);
                    same((tag + " aux: gate").c_str(), ra.data(), down(a1, T).data(), ra.size() * 4);
                }
                k::bf16_gemv_set_wide(0);
                if (no != NE) continue;
                // the fused router against the GEMV (+ aux) + native top-10
                poison(i0, (size_t) MT * 10); poison(w0, (size_t) MT * 10);
                k::native_router_top10_multi(y0, i0, w0, T, s);
                P::sync();
                const auto ri = down(i0, (size_t) T * 10);
                const auto rw = down(w0, (size_t) T * 10);
                G::Stream other = G::stream_create();
                for (const int rpb : {0, 1, 2, 4})
                    for (int mode = 0; mode < 3; ++mode) {   // direct; graph replayed twice; a second queue then the first
                        k::bf16_gemv_set_wide(rpb);
                        const std::string tag = "router fused rpb " + std::to_string(rpb) + " T" + std::to_string(T) +
                                                (mode == 0 ? "" : mode == 1 ? " (graph x2)" : " (two queues)");
                        poison(y1, (size_t) MT * 515); poison(a1, MT); poison(i1, (size_t) MT * 10); poison(w1, (size_t) MT * 10);
                        if (T > k::kWideMaxRows) {
                            // past the wide launch's rows: it must refuse and write nothing (the engine then runs the
                            // separate calls, which the unfused checks above cover)
                            ++g_checks;
                            if (k::bf16_router_fused(dx, N, dw, y1, N, T, dwsg, a1, 1, i1, w1, s)) {
                                ++g_fail;
                                std::printf("FAIL %s: took %d rows, past kWideMaxRows %d\n", tag.c_str(), T, k::kWideMaxRows);
                            }
                            P::sync();
                            const std::vector<uint8_t> ff((size_t) MT * 515 * 4, 0xff);
                            same((tag + " refused: logits untouched").c_str(), ff.data(), down(y1, (size_t) T * NE).data(), (size_t) T * NE * 4);
                            same((tag + " refused: gate untouched").c_str(), ff.data(), down(a1, T).data(), (size_t) T * 4);
                            same((tag + " refused: ids untouched").c_str(), ff.data(), down(i1, (size_t) T * 10).data(), (size_t) T * 40);
                            same((tag + " refused: weights untouched").c_str(), ff.data(), down(w1, (size_t) T * 10).data(), (size_t) T * 40);
                            break;
                        }
                        const auto call = [&](void* st) {
                            if (!k::bf16_router_fused(dx, N, dw, y1, N, T, dwsg, a1, 1, i1, w1, st))
                                throw std::runtime_error("bf16_router_fused refused");
                        };
                        if (mode == 0) {
                            call(s);
                        } else if (mode == 1) {
                            G::Graph* g = nullptr;
                            if (!G::begin_capture(other)) throw std::runtime_error("begin_capture");
                            call(other);
                            if (!G::end_capture(other, &g)) throw std::runtime_error("end_capture");
                            G::graph_launch(g, other);
                            G::stream_sync(other);
                            poison(i1, (size_t) MT * 10); poison(w1, (size_t) MT * 10);
                            G::graph_launch(g, other);
                            G::stream_sync(other);
                            G::graph_destroy(g);
                        } else {
                            call(other);
                            G::stream_sync(other);
                            poison(i1, (size_t) MT * 10); poison(w1, (size_t) MT * 10);
                            call(s);
                        }
                        P::sync();
                        same((tag + ": logits").c_str(), ry.data(), down(y1, (size_t) T * NE).data(), ry.size() * 4);
                        if (aux0) same((tag + ": gate").c_str(), ra.data(), down(a1, T).data(), ra.size() * 4);
                        same((tag + ": ids").c_str(), ri.data(), down(i1, (size_t) T * 10).data(), ri.size() * 4);
                        same((tag + ": weights").c_str(), rw.data(), down(w1, (size_t) T * 10).data(), rw.size() * 4);
                    }
                k::bf16_gemv_set_wide(0);
                G::stream_destroy(other);
            }
        if (reps > 0) {
            std::printf("\nrouter (2560 -> 512, the shared expert's gate, top-10), us a call (fastest / median):\n");
            for (const int T : {7, 16}) {
                std::vector<Timed> arms;
                arms.push_back({"T " + std::to_string(T) + " established (gemv+aux, top-10)", [=](void* st) {
                                    k::bf16_gemv_set_wide(0);
                                    if (!k::bf16_gemv_fp32_mmvf_multi_aux(dx, N, dw, y1, NE, N, NE, T, dwsg, a1, 1, st))
                                        k::bf16_gemv_fp32_mmvf_multi(dx, N, dw, y1, NE, N, NE, T, st);
                                    k::native_router_top10_multi(y1, i1, w1, T, st);
                                }});
                for (const int rpb : {1, 2, 4}) {
                    arms.push_back({"T " + std::to_string(T) + " wide rpb " + std::to_string(rpb) + " (gemv+aux, top-10)",
                                    [=](void* st) {
                                        k::bf16_gemv_set_wide(rpb);
                                        k::bf16_gemv_fp32_mmvf_multi_aux(dx, N, dw, y1, NE, N, NE, T, dwsg, a1, 1, st);
                                        k::native_router_top10_multi(y1, i1, w1, T, st);
                                    }});
                    arms.push_back({"T " + std::to_string(T) + " fused rpb " + std::to_string(rpb) + " (one launch)",
                                    [=](void* st) {
                                        k::bf16_gemv_set_wide(rpb);
                                        k::bf16_router_fused(dx, N, dw, y1, N, T, dwsg, a1, 1, i1, w1, st);
                                    }});
                }
                time_arms(arms, reps, rounds);
                std::vector<Timed> idx;
                for (const int64_t no : {(int64_t) 128, (int64_t) 512})
                    for (const int rpb : {0, 1, 2, 4})
                        idx.push_back({"T " + std::to_string(T) + " 2560 -> " + std::to_string(no) + " wide " + std::to_string(rpb),
                                       [=](void* st) {
                                           k::bf16_gemv_set_wide(rpb);
                                           k::bf16_gemv_fp32_mmvf_multi(dx, N, dw, y1, no, N, no, T, st);
                                       }});
                time_arms(idx, reps, rounds);
            }
            k::bf16_gemv_set_wide(0);
        }
    }

    // ============ STRATA_HC_PREFETCH's launch (l2_prefetch): a hint with no outputs.  The kernel that reads the
    // prefetched weights after it keeps its bits, the weights are untouched; direct, from a graph replayed twice, and
    // on an unaligned range of odd length and a null second range
    {
        std::vector<uint16_t> wa((size_t) HV * N), wb((size_t) HV * N);
        for (auto& v : wa) v = bf16(0.02f * nd(rng));
        for (auto& v : wb) v = bf16(0.02f * nd(rng));
        std::vector<float> dt(HV), sa(HV);
        for (auto& v : dt) v = nd(rng);
        for (auto& v : sa) v = -0.1f - ud(rng);
        const uint16_t *dwa = up(wa), *dwb = up(wb);
        const float *ddt = up(dt), *dsa = up(sa);
        float *g0 = buf<float>((size_t) MT * HV), *b0 = buf<float>((size_t) MT * HV);
        float *g1 = buf<float>((size_t) MT * HV), *b1 = buf<float>((size_t) MT * HV);
        const size_t wbytes = wa.size() * 2;
        k::gdn_ab_set_form(0);
        for (const int T : {1, 7, MT}) {
            const std::string tag = "l2_prefetch T" + std::to_string(T);
            poison(g0, (size_t) MT * HV); poison(b0, (size_t) MT * HV);
            k::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, g0, b0, N, HV, T, s);
            P::sync();
            const auto rg = down(g0, (size_t) T * HV), rb = down(b0, (size_t) T * HV);
            poison(g1, (size_t) MT * HV); poison(b1, (size_t) MT * HV);
            k::l2_prefetch(dwa, wbytes, dwb, wbytes, s);
            k::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, g1, b1, N, HV, T, s);
            P::sync();
            same((tag + " gate").c_str(), rg.data(), down(g1, (size_t) T * HV).data(), rg.size() * 4);
            same((tag + " beta").c_str(), rb.data(), down(b1, (size_t) T * HV).data(), rb.size() * 4);
            G::Stream st = G::stream_create();
            G::Graph* g = nullptr;
            if (!G::begin_capture(st)) throw std::runtime_error("begin_capture");
            k::l2_prefetch(dwa, wbytes, dwb, wbytes, st);
            k::gdn_ab_multi(dx, dwa, dwb, ddt, dsa, g1, b1, N, HV, T, st);
            if (!G::end_capture(st, &g)) throw std::runtime_error("end_capture");
            for (int rep = 0; rep < 2; ++rep) {
                poison(g1, (size_t) MT * HV); poison(b1, (size_t) MT * HV);
                G::graph_launch(g, st);
                G::stream_sync(st);
                same((tag + " graph gate").c_str(), rg.data(), down(g1, (size_t) T * HV).data(), rg.size() * 4);
                same((tag + " graph beta").c_str(), rb.data(), down(b1, (size_t) T * HV).data(), rb.size() * 4);
            }
            G::graph_destroy(g);
            G::stream_destroy(st);
        }
        k::l2_prefetch(reinterpret_cast<const char*>(dwa) + 3, 4097, nullptr, 0, s);   // unaligned, odd, one range
        k::l2_prefetch(dwa, 0, nullptr, 0, s);                                       // nothing: no launch
        P::sync();
        same("l2_prefetch weights untouched (alpha)", wa.data(), down(dwa, wa.size()).data(), wbytes);
        same("l2_prefetch weights untouched (beta)", wb.data(), down(dwb, wb.size()).data(), wbytes);
    }

    std::printf("fuse_small_parity: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "fuse_small_parity: %s\n", e.what());
    return 1;
}
