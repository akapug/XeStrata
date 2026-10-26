// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/hcx_parity.cpp - homelab STRATA_BATCH_HCX: hcx_read (hc_xmx.hpp) checked three ways.
//
//   exact  The int8 products bit for bit: the down partial sums and the up product against the host's emulation of
//          the same integer sums and the same float steps (from the device's R' and lo); lo's int8 slices and scales
//          against the host's quantizer; mixed's q8_1 against native_quantize_q8_1 of the device's mixed; R' against
//          the established read's (fused_gr_read_multi) bit for bit; a second launch the same bits; on a card with
//          XMX, the XMX body against the portable one bit for bit (every output and the scratch).
//   bound  Every output (R', rs, lo, inject, u, mixed) against the exact read in FP64 within the error bound derived
//          from the activations' rounding (amax / 254 an element with one slice, amax / 32512 with two), the FP32
//          steps and the transcendental functions (see bounds() below); the largest error / bound is printed.
//   path   R' bit for bit, and rs, inject, mixed (and lo, where it is an output) against the established FP32 read
//          (fused_gr_read_multi), within the same bound plus that read's own FP32 error: the size of the change hcx
//          makes, printed per row count.  The established Q8_0 read keeps lo on chip (gr_up_q8's local memory, the
//          window forms' scratch: fused_gr_win.hpp "no lo") and never writes FusedGrArgs::lo, so its lo is compared
//          only where it was written; every value the checks read from it must have been written (a value left at
//          reset()'s fill is a failure, never a skipped comparison: a NaN drops out of std::max and passes `!(x > b)`).
//
// Rows 1..32 (odd counts and the 32-row window: kHcxMaxRows), one and two slices, the previous write folded or not,
// with and without the inject rows and mixed's q8_1.  Then (--time, default on a GPU) each body and the established
// read timed per row count, eager launches, the median of --reps.
//
//     build/hcx_parity [--rows 1,2,3,7,8,9,15,16,17,24,31,32] [--slices 1,2] [--reps 20] [--ref-form 2|-1]
//                      [--time|--no-time]
//
// On the OpenCL CPU device (STRATA_TEST_CPU_DEVICE=1) only the portable body runs; every other check runs there too,
// the path checks against the served form included (--ref-form, below).
#include "strata/kernels/hc_xmx.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace P = strata::parity;
namespace k = strata::kernels;

constexpr int N = k::kHcxN, HC = k::kHcxHC, D = k::kHcxD, LR = k::kHcxLR, QB = k::kHcxBlock, NCH = k::kHcxChunks;
constexpr int PR = k::kHcxPR, NBL = LR / QB, MT = k::kHcxMaxRows, KCH = k::kHcxKch, NBC = KCH / QB;
constexpr size_t ROW_D = (size_t) D + D / QB * 2, ROW_U = (size_t) LR + LR / QB * 2, Q81 = (size_t) N / 32 * 36;
constexpr double U24 = 1.0 / 16777216.0;   // 2^-24, FP32's unit roundoff
constexpr double kSlack = 1.01;            // every bound: 1 % for the second-order terms the derivation drops
constexpr double kFusedGam = 256 * U24;    // the established read's own FP32 error, relative to its sums' magnitudes
// the established read's form (STRATA_HC_Q8_WIN; every form gives the same bits, hc_read16_parity): 2, the served one,
// on a card and on the OpenCL CPU device alike; -1 none, the checks against it skipped.  (Until the down launches took
// (1, THREADS) work-groups, fused_gr.cpp's kQ8DownGroup, the CPU device ran them as sub-groups of one work-item and
// read past q8_down: the CPU default was -1, and the CPU proof never ran the path checks.)
constexpr int kRefFormDefault = 2;

std::mt19937 rng(353);  // NOLINT(bugprone-random-generator-seed): reproducible fixtures.
int g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::printf("  FAIL %s\n", what.c_str());
    }
}

template <typename T> T* dalloc(size_t n) { return static_cast<T*>(P::alloc_bytes(n * sizeof(T) + 64)); }
template <typename T> T* dput(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    P::copy_bytes_in(p, h.data(), h.size() * sizeof(T));
    return p;
}
template <typename T> std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n);
    P::copy_bytes_out(h.data(), d, n * sizeof(T));
    return h;
}
template <typename T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T));
}
float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return (float) sycl::bit_cast<sycl::half>(h);
}
float bf2f(uint16_t v) { return sycl::bit_cast<float>((uint32_t) v << 16); }
double silu(double x) { return x / (1.0 + std::exp(-x)); }
double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// rows of `ne0` Q8_0 values: the int8 values, then the fp16 scales (kNativeQ8Rows)
std::vector<uint8_t> q8_rows(int rows, int ne0) {
    const size_t row = (size_t) ne0 + (size_t) ne0 / 32 * 2;
    std::vector<uint8_t> w((size_t) rows * row);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> sc(0.002f, 0.02f);
    for (int r = 0; r < rows; ++r) {
        uint8_t* p = w.data() + (size_t) r * row;
        for (int i = 0; i < ne0; ++i) p[i] = (uint8_t) (int8_t) q(rng);
        for (int b = 0; b < ne0 / 32; ++b) {
            const uint16_t h = sycl::bit_cast<uint16_t>(sycl::half(sc(rng)));
            std::memcpy(p + ne0 + 2 * b, &h, 2);
        }
    }
    return w;
}

// hc_xmx.cpp's quant_pair on a block of 32: slice one round(v / d), d = amax / 127; slice two the residual at d / 128
float quant_block(const float* v, int S, int8_t* q1, int8_t* q2) {
    float amax = 0.0f;
    for (int i = 0; i < QB; ++i) amax = std::fmax(amax, std::fabs(v[i]));
    const float d = amax / 127.0f, id = amax > 0.0f ? 1.0f / d : 0.0f, id2 = id * 128.0f;
    for (int i = 0; i < QB; ++i) {
        const float a = std::round(v[i] * id);
        q1[i] = (int8_t) (int) a;
        if (S == 2) q2[i] = (int8_t) (int) std::min(127.0f, std::max(-127.0f, std::round(std::fma(-a, d, v[i]) * id2)));
    }
    return S == 2 ? d * (1.0f / 128.0f) : d;
}
// the bound on |v - its slices' value| an element of a block of largest magnitude amax (see bounds())
double quant_err(double amax, int S) {
    return S == 2 ? amax / 127.0 / 256.0 * (1.0 + 1.0 / 16384) : amax / 127.0 * (0.5 + 1.0 / 16384);
}

struct Weights {
    std::vector<uint8_t> qd, qu;
    std::vector<float> wn;
    std::vector<uint16_t> wi;
    std::vector<double> wd, wu;   // dequantized, [out][in]
    uint8_t *dqd, *dqu;
    float* dwn;
    uint16_t* dwi;
};
Weights make_weights() {
    Weights w;
    w.qd = q8_rows(LR, D);
    w.qu = q8_rows(D, LR);
    std::normal_distribution<float> nd(0.f, 1.f);
    w.wn.resize(D);
    for (auto& x : w.wn) x = 1.0f + 0.1f * nd(rng);
    w.wi.resize((size_t) HC * D);
    for (auto& x : w.wi) x = (uint16_t) (sycl::bit_cast<uint32_t>(0.02f * nd(rng)) >> 16);
    w.wd.resize((size_t) LR * D);
    for (int r = 0; r < LR; ++r)
        for (int kk = 0; kk < D; ++kk)
            w.wd[(size_t) r * D + kk] = (double) h2f(&w.qd[r * ROW_D + D + 2 * (kk / 32)]) * (int8_t) w.qd[r * ROW_D + kk];
    w.wu.resize((size_t) D * LR);
    for (int i = 0; i < D; ++i)
        for (int kk = 0; kk < LR; ++kk)
            w.wu[(size_t) i * LR + kk] = (double) h2f(&w.qu[i * ROW_U + LR + 2 * (kk / 32)]) * (int8_t) w.qu[i * ROW_U + kk];
    w.dqd = dput(w.qd);
    w.dqu = dput(w.qu);
    w.dwn = dput(w.wn);
    w.dwi = dput(w.wi);
    return w;
}

struct Cfg {
    bool apply, inject, q8m;
    std::string name() const {
        return std::string(apply ? "apply" : "no-apply") + (inject ? "+inject" : "") + (q8m ? "+q8_1" : "");
    }
};

// the device buffers of one read (each row contiguous, as Verifier's window)
struct Bufs {
    float *R0, *R, *bo, *ip, *lo, *rs, *io, *mixed, *xn, *u;
    uint8_t *q8m, *scratch;
};
Bufs make_bufs(const std::vector<float>& R0, const std::vector<float>& bo, const std::vector<float>& ip) {
    Bufs b{};
    b.R0 = dput(R0);
    b.R = dalloc<float>(R0.size());
    b.bo = dput(bo);
    b.ip = dput(ip);
    b.lo = dalloc<float>((size_t) MT * LR);
    b.rs = dalloc<float>((size_t) MT * HC);
    b.io = dalloc<float>((size_t) MT * HC);
    b.mixed = dalloc<float>((size_t) MT * N);
    b.xn = dalloc<float>((size_t) MT * D);
    b.u = dalloc<float>((size_t) MT * D);
    b.q8m = dalloc<uint8_t>((size_t) MT * Q81);
    b.scratch = dalloc<uint8_t>(k::hcx_scratch_bytes(MT));
    return b;
}
std::vector<k::FusedGrArgs> args(const Weights& w, const Bufs& b, int T, const Cfg& c) {
    std::vector<k::FusedGrArgs> a((size_t) T);
    for (int t = 0; t < T; ++t) {
        k::FusedGrArgs& x = a[(size_t) t];
        x.R = b.R + (size_t) t * D;
        x.R_out = b.R + (size_t) t * D;
        x.apply = c.apply;
        x.bo_prev = b.bo + (size_t) t * N;
        x.inj_prev = b.ip + (size_t) t * HC;
        x.w_norm = w.dwn;
        x.w_inject = c.inject ? w.dwi : nullptr;
        x.q8_down = w.dqd;
        x.q8_up = w.dqu;
        x.eps = 1e-6f;
        x.lo = b.lo + (size_t) t * LR;
        x.rs = b.rs + (size_t) t * HC;
        x.inject_out = b.io + (size_t) t * HC;
        x.mixed = b.mixed + (size_t) t * N;
        x.q8_mixed = c.q8m ? b.q8m + (size_t) t * Q81 : nullptr;
    }
    return a;
}
void reset(const Bufs& b) {
    auto& q = strata::core::Runtime::get().compute();
    q.memcpy(b.R, b.R0, (size_t) MT * D * 4);
    q.memset(b.lo, 0xff, (size_t) MT * LR * 4);
    q.memset(b.rs, 0xff, (size_t) MT * HC * 4);
    q.memset(b.io, 0xff, (size_t) MT * HC * 4);
    q.memset(b.mixed, 0xff, (size_t) MT * N * 4);
    q.memset(b.q8m, 0xff, (size_t) MT * Q81);
    q.memset(b.u, 0xff, (size_t) MT * D * 4);
    q.memset(b.scratch, 0xff, k::hcx_scratch_bytes(MT));
    P::sync();
}

struct Out {
    std::vector<float> R, rs, lo, io, mixed, part, ssg, lsc, u;
    std::vector<int8_t> lq;
    std::vector<uint8_t> q8m;
};
Out outputs(const Bufs& b, int T, bool hcx) {
    Out o;
    o.R = fetch(b.R, (size_t) T * D);
    o.rs = fetch(b.rs, (size_t) T * HC);
    o.lo = fetch(b.lo, (size_t) T * LR);
    o.io = fetch(b.io, (size_t) T * HC);
    o.mixed = fetch(b.mixed, (size_t) T * N);
    o.q8m = fetch(b.q8m, (size_t) T * Q81);
    if (!hcx) return o;
    const k::HcxScratch s = k::hcx_carve(b.scratch, MT);
    o.part = fetch(s.part, (size_t) T * NCH * PR);
    o.ssg = fetch(s.ssg, (size_t) T * NCH);
    o.lq = fetch(s.lq, (size_t) T * k::kHcxMaxSlices * LR);
    o.lsc = fetch(s.lsc, (size_t) T * NBL);
    o.u = fetch(b.u, (size_t) T * D);
    return o;
}
Out run_hcx(const Weights& w, const Bufs& b, int T, int S, const Cfg& c, k::HcxBody body) {
    reset(b);
    const auto a = args(w, b, T, c);
    k::HcxScratch s = k::hcx_carve(b.scratch, MT);
    s.u = b.u;
    k::hcx_read(a.data(), T, s, P::stream(), S, body);
    P::sync();
    return outputs(b, T, true);
}
Out run_fused(const Weights& w, const Bufs& b, int T, const Cfg& c) {
    reset(b);
    const auto a = args(w, b, T, c);
    k::fused_gr_read_multi(a.data(), T, b.xn, P::stream());
    P::sync();
    return outputs(b, T, false);
}

// The checks of one hcx read `o` (rows T, slices S): exact, bound, and against the established read `f`.
struct Stats {
    double lo_rel = 0, mixed_rel = 0, bound_ratio = 0, path_lo = 0, path_mixed = 0;
    bool ref_lo = false;   // the established read wrote lo (path_lo is meaningful)
};
// the largest error so far; a NaN (an unwritten or broken value) is the largest of all, never dropped as std::max drops it
double worse(double a, double b) { return std::isnan(b) ? INFINITY : std::max(a, b); }
// reset()'s fill (0xff bytes) left in a range: the value was never written
long unwritten(const float* p, size_t n) {
    long c = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, p + i, 4);
        c += u == 0xffffffffu;
    }
    return c;
}
Stats verify(const Weights& w, const std::vector<float>& R0, const std::vector<float>& bo, const std::vector<float>& ip,
             int T, int S, const Cfg& c, const Out& o, const Out& f, const std::string& tag) {
    Stats st;
    const bool ref = !f.R.empty();   // the established read ran (--ref-form)
    // lo is compared where the established read writes it (not the Q8_0 read's, see the header): all of it or none
    const size_t nlo = (size_t) T * LR;
    const long lo_unwritten = ref ? unwritten(f.lo.data(), nlo) : 0;
    st.ref_lo = ref && (size_t) lo_unwritten < nlo;
    long bad_lo = 0;
    if (ref) {
        const long miss = unwritten(f.R.data(), f.R.size()) + unwritten(f.rs.data(), (size_t) T * HC) +
                          unwritten(f.mixed.data(), (size_t) T * N) + (c.inject ? unwritten(f.io.data(), (size_t) T * HC) : 0) +
                          (st.ref_lo ? lo_unwritten : 0);
        check(miss == 0, tag + ": the established read wrote every value compared (R', rs, mixed" +
                             (c.inject ? ", inject" : "") + (st.ref_lo ? ", lo" : "") + "; " + std::to_string(miss) +
                             " left at the fill)");
    }
    const float eps = 1e-6f;
    long bad_r = 0, bad_part = 0, bad_q = 0, bad_u = 0, bad_b[6] = {0, 0, 0, 0, 0, 0}, bad_path = 0;
    double worst[6] = {0, 0, 0, 0, 0, 0};   // error / bound: R', rs, lo, inject, u, mixed
    const auto bound_check = [&](int which, double err, double bound) {
        const double r = err / std::max(bound * kSlack, 1e-300);
        worst[which] = std::max(worst[which], r);
        if (!(err <= bound * kSlack)) ++bad_b[which];
    };
    std::vector<int8_t> q1(D), q2(D);
    std::vector<float> ysc(D / QB), y(D);
    double lo_max = 0, dlo_max = 0, mx_max = 0, dmx_max = 0;
    for (int t = 0; t < T; ++t) {
        const float* R = &o.R[(size_t) t * D];
        // R': the established read's bits (the same expression) and the FP64 bound (sigmoid's exp ~ a few ulps)
        for (int kk = 0; kk < D; ++kk) {
            const int cc = kk / N;
            const double r0 = R0[(size_t) t * D + kk];
            const double g = c.apply ? 2.0 * sigm((double) ip[t * HC + cc] / HC) : 0.0;
            const double b0 = bo[(size_t) t * N + kk % N];
            const double rx = r0 + b0 * g;
            bound_check(0, std::fabs(R[kk] - rx), std::fabs(b0 * g) * 16 * U24 + std::fabs(rx) * U24 + 1e-300);
            if (ref && std::memcmp(&R[kk], &f.R[(size_t) t * D + kk], 4) != 0) ++bad_r;
        }
        // exact: y = R' w_norm quantized as the device does, the down partial sums emulated
        for (int kk = 0; kk < D; ++kk) y[kk] = R[kk] * w.wn[kk];
        for (int b = 0; b < D / QB; ++b) ysc[b] = quant_block(&y[(size_t) b * QB], S, &q1[(size_t) b * QB], &q2[(size_t) b * QB]);
        for (int ch = 0; ch < NCH; ++ch)
            for (int r = 0; r < LR; ++r) {
                const uint8_t* wr = &w.qd[r * ROW_D];
                float acc = 0.0f;
                for (int b = 0; b < NBC; ++b) {
                    const int kb = ch * NBC + b;
                    int c1 = 0, c2 = 0;
                    for (int i = 0; i < QB; ++i) {
                        const int wv = (int8_t) wr[kb * QB + i];
                        c1 += q1[(size_t) kb * QB + i] * wv;
                        c2 += q2[(size_t) kb * QB + i] * wv;
                    }
                    const int cs = S == 2 ? c1 * 128 + c2 : c1;
                    acc = std::fma(h2f(wr + D + 2 * kb) * ysc[kb], (float) cs, acc);
                }
                if (std::memcmp(&acc, &o.part[((size_t) t * NCH + ch) * PR + r], 4) != 0) ++bad_part;
            }
        // the exact read in FP64 from R'; the bounds
        double ss[HC] = {0, 0, 0, 0}, rs[HC];
        for (int kk = 0; kk < D; ++kk) ss[kk / N] += (double) R[kk] * R[kk];
        for (int cc = 0; cc < HC; ++cc) rs[cc] = 1.0 / std::sqrt(ss[cc] / N + eps);
        const double drs = 64 * U24;   // rs's relative bound: 24 sums of squares deep, /N, +eps, rsqrt's few ulps
        for (int cc = 0; cc < HC; ++cc) {
            bound_check(1, std::fabs(o.rs[t * HC + cc] - rs[cc]), drs * rs[cc]);
            if (ref && !(std::fabs(o.rs[t * HC + cc] - f.rs[t * HC + cc]) <= drs * rs[cc] * kSlack + kFusedGam * rs[cc]))
                ++bad_path;
        }
        std::vector<double> e(D);   // an element's rounding bound: its block's slices, y's own rounding
        for (int b = 0; b < D / QB; ++b) {
            double amax = 0;
            for (int i = 0; i < QB; ++i) amax = std::max(amax, (double) std::fabs(y[(size_t) b * QB + i]));
            for (int i = 0; i < QB; ++i) e[(size_t) b * QB + i] = quant_err(amax, S);
        }
        std::vector<double> lo_x(LR), dlo(LR), flb(LR);
        const double gam = 32 * U24;   // the float steps of a partial sum, the 8 chunks and the 4 streams
        for (int r = 0; r < LR; ++r) {
            const double* wr = &w.wd[(size_t) r * D];
            double dot = 0, bq = 0, babs = 0, brs = 0;
            for (int cc = 0; cc < HC; ++cc) {
                double sd = 0, sq = 0, sa = 0;
                for (int kk = cc * N; kk < (cc + 1) * N; ++kk) {
                    const double a = std::fabs(wr[kk]), yy = std::fabs((double) y[kk]);
                    sd += wr[kk] * (double) R[kk] * w.wn[kk];
                    sq += a * (e[kk] + U24 * yy);
                    sa += a * (yy + e[kk]);
                }
                dot += rs[cc] * sd;
                bq += rs[cc] * sq;
                babs += rs[cc] * sa;
                brs += rs[cc] * std::fabs(sd);
            }
            const double pre_bound = bq + gam * babs + drs * brs;
            lo_x[r] = silu(dot / HC);
            const double lb = 1.1 * pre_bound / HC + 16 * U24 * std::fabs(lo_x[r]) + 1e-300;
            dlo[r] = std::fabs(o.lo[(size_t) t * LR + r] - lo_x[r]);
            bound_check(2, dlo[r], lb);
            lo_max = std::max(lo_max, std::fabs(lo_x[r]));
            dlo_max = std::max(dlo_max, dlo[r]);
            flb[r] = kFusedGam * (babs / HC + std::fabs(lo_x[r]));   // the established read's own FP32 error
            if (st.ref_lo) {
                const double dpath = std::fabs(o.lo[(size_t) t * LR + r] - f.lo[(size_t) t * LR + r]);
                if (!(dpath <= lb * kSlack + flb[r])) ++bad_lo;
                st.path_lo = worse(st.path_lo, dpath);
            }
        }
        // inject (FP32, as the established read)
        if (c.inject)
            for (int j = 0; j < HC; ++j) {
                double dot = 0, babs = 0, brs = 0;
                for (int cc = 0; cc < HC; ++cc) {
                    double sd = 0, sa = 0;
                    for (int kk = cc * N; kk < (cc + 1) * N; ++kk) {
                        const double wij = bf2f(w.wi[(size_t) j * D + kk]);
                        sd += wij * (double) R[kk] * w.wn[kk];
                        sa += std::fabs(wij) * std::fabs((double) y[kk]);
                    }
                    dot += rs[cc] * sd;
                    babs += rs[cc] * sa;
                    brs += rs[cc] * std::fabs(sd);
                }
                const double ib = 48 * U24 * babs + drs * brs + 1e-300;
                bound_check(3, std::fabs(o.io[t * HC + j] - dot), ib);
                if (ref && !(std::fabs(o.io[t * HC + j] - f.io[t * HC + j]) <= ib * kSlack + kFusedGam * babs)) ++bad_path;
            }
        // exact: lo's slices from the device's lo, the up product emulated
        std::vector<int8_t> l1(LR), l2(LR);
        std::vector<float> lsc(NBL);
        std::vector<double> le(LR);
        for (int b = 0; b < NBL; ++b) {
            lsc[b] = quant_block(&o.lo[(size_t) t * LR + b * QB], S, &l1[(size_t) b * QB], &l2[(size_t) b * QB]);
            double amax = 0;
            for (int i = 0; i < QB; ++i) amax = std::max(amax, (double) std::fabs(o.lo[(size_t) t * LR + b * QB + i]));
            for (int i = 0; i < QB; ++i) le[(size_t) b * QB + i] = quant_err(amax, S);
        }
        if (std::memcmp(lsc.data(), &o.lsc[(size_t) t * NBL], NBL * 4) != 0 ||
            std::memcmp(l1.data(), &o.lq[(size_t) t * k::kHcxMaxSlices * LR], LR) != 0 ||
            (S == 2 && std::memcmp(l2.data(), &o.lq[((size_t) t * k::kHcxMaxSlices + 1) * LR], LR) != 0))
            ++bad_q;
        for (int i = 0; i < D; ++i) {
            const uint8_t* wr = &w.qu[i * ROW_U];
            float acc = 0.0f;
            for (int b = 0; b < NBL; ++b) {
                int c1 = 0, c2 = 0;
                for (int j = 0; j < QB; ++j) {
                    const int wv = (int8_t) wr[b * QB + j];
                    c1 += l1[(size_t) b * QB + j] * wv;
                    c2 += l2[(size_t) b * QB + j] * wv;
                }
                const int cs = S == 2 ? c1 * 128 + c2 : c1;
                acc = std::fma(h2f(wr + LR + 2 * b) * lsc[b], (float) cs, acc);
            }
            if (std::memcmp(&acc, &o.u[(size_t) t * D + i], 4) != 0) ++bad_u;
        }
        // the up product and mixed against the exact read; u's bound from the device's own lo error (a fact here)
        std::vector<double> ux(D), du(D), fu(D);
        for (int i = 0; i < D; ++i) {
            const double* wr = &w.wu[(size_t) i * LR];
            double dot = 0, bq = 0, babs = 0, fua = 0;
            for (int kk = 0; kk < LR; ++kk) {
                const double a = std::fabs(wr[kk]);
                dot += wr[kk] * lo_x[kk];
                bq += a * (dlo[kk] + le[kk]);
                babs += a * (std::fabs((double) o.lo[(size_t) t * LR + kk]) + le[kk]);
                fua += a * (flb[kk] + kFusedGam * std::fabs(lo_x[kk]));
            }
            ux[i] = dot;
            fu[i] = fua;   // the established read's u error
            du[i] = std::fabs(o.u[(size_t) t * D + i] - dot);
            bound_check(4, du[i], bq + gam * babs + 1e-300);
        }
        for (int d = 0; d < N; ++d) {
            double m = 0, mb = 0, fm = 0;
            for (int cc = 0; cc < HC; ++cc) {
                const int i = cc * N + d;
                const double x = (double) R[i] * w.wn[i] * rs[cc];
                m += x * sigm(ux[i]);
                mb += std::fabs(x) * (du[i] / 4 + drs + 16 * U24);
                fm += std::fabs(x) * (fu[i] / 4 + kFusedGam);
            }
            m /= HC;
            mb = mb / HC + 1e-300;
            const double dev = o.mixed[(size_t) t * N + d];
            bound_check(5, std::fabs(dev - m), mb);
            mx_max = std::max(mx_max, std::fabs(m));
            dmx_max = std::max(dmx_max, std::fabs(dev - m));
            const double dpath = ref ? std::fabs(dev - f.mixed[(size_t) t * N + d]) : 0.0;
            if (!(dpath <= mb * kSlack + fm / HC)) ++bad_path;
            st.path_mixed = worse(st.path_mixed, dpath);
        }
    }
    if (ref) check(bad_r == 0, tag + ": R' is the established read's, bit for bit (" + std::to_string(bad_r) + " differ)");
    check(bad_part == 0, tag + ": the down partial sums are the host's integer emulation, bit for bit (" +
                             std::to_string(bad_part) + " differ)");
    check(bad_q == 0, tag + ": lo's int8 slices and scales are the host quantizer's");
    check(bad_u == 0, tag + ": the up product is the host's integer emulation, bit for bit (" + std::to_string(bad_u) +
                          " differ)");
    const char* names[6] = {"R'", "rs", "lo", "inject", "u", "mixed"};
    for (int i = 0; i < 6; ++i) {
        if (i == 3 && !c.inject) continue;
        check(bad_b[i] == 0, tag + ": " + names[i] + " within its bound (" + std::to_string(bad_b[i]) +
                                 " past it, worst error/bound " + std::to_string(worst[i]) + ")");
        st.bound_ratio = std::max(st.bound_ratio, worst[i]);
    }
    if (ref)
        check(bad_path == 0, tag + ": rs, inject and mixed within the bound of the established read's (" +
                                 std::to_string(bad_path) + " past it)");
    if (st.ref_lo)
        check(bad_lo == 0, tag + ": lo within the bound of the established read's (" + std::to_string(bad_lo) + " past it)");
    st.lo_rel = dlo_max / std::max(lo_max, 1e-300);
    st.mixed_rel = dmx_max / std::max(mx_max, 1e-300);
    return st;
}

std::vector<int> int_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) v.push_back(std::stoi(x));
    return v;
}

double time_us(const std::function<void()>& f, int reps) {
    std::vector<double> v;
    for (int i = 0; i < reps + 3; ++i) {
        P::sync();
        const auto t0 = std::chrono::steady_clock::now();
        f();
        P::sync();
        if (i >= 3) v.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) try {
    std::vector<int> rows{1, 2, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32}, slices{1, 2};
    int reps = 20, timing = -1, ref_form = kRefFormDefault;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) rows = int_list(argv[++i]);
        else if (a == "--slices" && i + 1 < argc) slices = int_list(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--ref-form" && i + 1 < argc) ref_form = std::atoi(argv[++i]);
        else if (a == "--time") timing = 1;
        else if (a == "--no-time") timing = 0;
        else {
            std::fprintf(stderr, "usage: hcx_parity [--rows L] [--slices L] [--reps N] [--ref-form F] [--time|--no-time]\n");
            return 2;
        }
    }
    for (int T : rows)
        if (T < 1 || T > MT) { std::fprintf(stderr, "rows 1..%d\n", MT); return 2; }
    if (ref_form < -1 || ref_form >= k::kFusedGrQ8WinForms) {
        std::fprintf(stderr, "ref-form -1..%d\n", k::kFusedGrQ8WinForms - 1);
        return 2;
    }
    if (ref_form >= 0) k::fused_gr_set_q8_win(ref_form);
    sycl::queue& q = strata::core::Runtime::get().compute();
    const bool gpu = q.get_device().is_gpu(), xmx = k::hcx_engine();
    if (timing < 0) timing = gpu ? 1 : 0;
    std::printf("hcx_parity: device %s (local memory %zu bytes), XMX body %s, rows up to %d, chunks %d of %d inputs, "
                "reference STRATA_HC_Q8_WIN=%d (-1: none)\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                (size_t) q.get_device().get_info<sycl::info::device::local_mem_size>(), xmx ? "yes" : "no (portable only)",
                MT, NCH, KCH, ref_form);
    std::fflush(stdout);

    const Weights w = make_weights();
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R0((size_t) MT * D), bo((size_t) MT * N), ip((size_t) MT * HC);
    for (int t = 0; t < MT; ++t)
        for (int cc = 0; cc < HC; ++cc)
            for (int d = 0; d < N; ++d) R0[(size_t) t * D + cc * N + d] = nd(rng) * (float) (1 << (2 * cc));
    for (auto& x : bo) x = nd(rng);
    for (auto& x : ip) x = 3.0f * nd(rng);
    const Bufs b = make_bufs(R0, bo, ip);
    uint8_t* q8ref = dalloc<uint8_t>((size_t) MT * Q81);
    float* mref = dalloc<float>((size_t) MT * N);

    const Cfg cfgs[2] = {{true, true, true}, {false, false, false}};
    std::printf("%5s %6s %-24s %12s %12s %12s %14s %14s\n", "rows", "slices", "case", "lo err/max", "mixed err/max",
                "err/bound", "|lo-fused|", "|mixed-fused|");
    for (int T : rows)
        for (const Cfg& c : cfgs) {
            const Out f = ref_form >= 0 ? run_fused(w, b, T, c) : Out{};
            for (int S : slices) {
                const std::string tag = "rows " + std::to_string(T) + " slices " + std::to_string(S) + " " + c.name();
                const Out o = run_hcx(w, b, T, S, c, k::HcxBody::portable);
                const Out o2 = run_hcx(w, b, T, S, c, k::HcxBody::portable);
                check(same(o.R, o2.R) && same(o.lo, o2.lo) && same(o.rs, o2.rs) && same(o.io, o2.io) &&
                          same(o.mixed, o2.mixed) && same(o.q8m, o2.q8m) && same(o.part, o2.part) && same(o.u, o2.u),
                      tag + ": a second launch gives the same bits");
                if (xmx) {
                    const Out x = run_hcx(w, b, T, S, c, k::HcxBody::xmx);
                    check(same(o.R, x.R) && same(o.lo, x.lo) && same(o.rs, x.rs) && same(o.mixed, x.mixed) &&
                              same(o.q8m, x.q8m) && same(o.lsc, x.lsc) && same(o.u, x.u) && same(o.ssg, x.ssg),
                          tag + ": the XMX body is the portable one, bit for bit");
                    std::vector<float> pa(o.part), px(x.part);   // the down sums (and inject partials when on)
                    check(same(pa, px), tag + ": the XMX down partial sums are the portable ones, bit for bit");
                    check(!c.inject || same(o.io, x.io), tag + ": the XMX inject is the portable one");
                }
                if (c.q8m) {   // mixed's q8_1 as native_quantize_q8_1 writes it for the same mixed
                    P::copy_bytes_in(mref, o.mixed.data(), o.mixed.size() * 4);
                    k::native_quantize_q8_1(mref, q8ref, N, T, P::stream());
                    P::sync();
                    check(same(fetch(q8ref, (size_t) T * Q81), o.q8m), tag + ": mixed's q8_1 is native_quantize_q8_1's");
                }
                const Stats st = verify(w, R0, bo, ip, T, S, c, o, f, tag);
                std::printf("%5d %6d %-24s %12.3e %12.3e %12.3f ", T, S, c.name().c_str(), st.lo_rel, st.mixed_rel,
                            st.bound_ratio);
                if (st.ref_lo) std::printf("%14.3e %14.3e\n", st.path_lo, st.path_mixed);
                else if (ref_form >= 0) std::printf("%14s %14.3e\n", "(on chip)", st.path_mixed);
                else std::printf("%14s %14s\n", "-", "-");
                std::fflush(stdout);
            }
        }

    if (timing) {
        std::printf("\ntime per read (us, eager launches, median of %d; the engine replays them in a graph)\n", reps);
        std::printf("%5s %12s %12s %12s %12s\n", "rows", "fused", "hcx S1", "hcx S2", "portable S2");
        const Cfg c = cfgs[0];
        for (int T : rows) {
            reset(b);
            const auto a = args(w, b, T, c);
            const k::HcxScratch s = k::hcx_carve(b.scratch, MT);
            const auto body = xmx ? k::HcxBody::xmx : k::HcxBody::portable;
            const double tf =
                ref_form >= 0 ? time_us([&] { k::fused_gr_read_multi(a.data(), T, b.xn, P::stream()); }, reps) : 0.0;
            const double t1 = time_us([&] { k::hcx_read(a.data(), T, s, P::stream(), 1, body); }, reps);
            const double t2 = time_us([&] { k::hcx_read(a.data(), T, s, P::stream(), 2, body); }, reps);
            const double tp = time_us([&] { k::hcx_read(a.data(), T, s, P::stream(), 2, k::HcxBody::portable); }, reps);
            std::printf("%5d %12.1f %12.1f %12.1f %12.1f\n", T, tf, t1, t2, tp);
        }
    }
    std::printf("hcx_parity: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "hcx_parity: %s\n", e.what());
    return 1;
}
