// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/verify_kernels.cpp - the Xe port of Strata's src/kernels/cuda/verify_kernels.cu (see verify_kernels.hpp):
// the verify window's multi-token GDN kernels, the resident-expert plan, the flag waits on host-mapped memory, and
// the small gathers, copies and selections around them.
//
// The per-token arithmetic of each kernel is its single-token original's (fused_gdn, elementwise) with the same
// operation order, so a verify window reproduces plain decode.  The flag waits use the reading measured in
// the 2026-09-30-xe-doorbell record: a volatile read followed by a system-scope acquire fence.  gpu_stamp writes
// device-clock ticks (the scope STRATA_VERIFY_PROFILE_CLOCK picks), not the nanoseconds of CUDA's %globaltimer.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/core/runtime.hpp"
#include "device_caps.hpp"
#include "q8_1_finite.hpp"
#include "strata/core/per_device.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_verify_kernels
#include "strata/kernels/device_arch_tag.hpp"
#include "param_table.hpp"

// sycl_ext_oneapi_clock is newer than some DPC++ releases (intel/llvm 6.2 lacks it): without it there is no stamp
#if __has_include(<sycl/ext/oneapi/experimental/clock.hpp>)
#include <sycl/ext/oneapi/experimental/clock.hpp>
#endif
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;
constexpr int WARP = 32;

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }
[[noreturn]] void fail(const std::string& what) { throw core::DeviceError(what); }
void done(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
// The q8_1 block of a sub-group's 32 values, lane l's value xi (native_quantize_q8_1's quantizer, the same bytes):
// contraction off, so the product that made xi cannot fuse into the sum's first add.
inline void q8_1_store(const sycl::sub_group& sg, const float xi, uint8_t* block, const int lane) {
#pragma clang fp contract(off)
    float amax = sycl::fabs(xi), sum = xi;
    for (int o = 16; o > 0; o >>= 1) {
        amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
        sum += sycl::permute_group_by_xor(sg, sum, o);
    }
    const float d = xe::q8_1_finite(amax / 127.0f);
    block[4 + lane] = (uint8_t) xe::q8_1_quant(xi, d, amax);
    if (lane == 0) *reinterpret_cast<sycl::half2*>(block) = sycl::half2(sycl::half(d), sycl::half(xe::q8_1_finite(sum)));
}
// lane l of a sub-group reads or writes p[l], one block message for the sub-group (Block) or one gather/scatter
template <bool Block> inline float state_load(const sycl::sub_group& sg, const float* p, int lane) {
    if constexpr (Block) {
        namespace sx = sycl::ext::oneapi::experimental;
        float v;
        sx::group_load(sg, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::yes>(p).get_decorated(),
                       v, sx::properties{sx::full_group, sx::alignment<16>});
        return v;
    } else {
        return p[lane];
    }
}
template <bool Block> inline void state_store(const sycl::sub_group& sg, float v, float* p, int lane) {
    if constexpr (Block) {
        namespace sx = sycl::ext::oneapi::experimental;
        sx::group_store(sg, v, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                        sycl::access::decorated::yes>(p).get_decorated(),
                        sx::properties{sx::full_group, sx::alignment<16>});
    } else {
        p[lane] = v;
    }
}

// A kernel lambda launched with kernel properties, which a lambda cannot carry itself: its sub-group size, its
// work-group size and the large register file (grf_size<256>).
template <int WG, typename F> struct LargeGrf {
    F f;
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        namespace sx = sycl::ext::oneapi::experimental;
        return sx::properties{sycl::ext::intel::experimental::grf_size<256>, sx::sub_group_size<WARP>,
                              sx::work_group_size<WG>};
    }
    void operator()(sycl::nd_item<1> it) const { f(it); }
};
// A block store needs 16-byte alignment (a sub-group's run of the state is a multiple of 128 bytes from the state's
// start): a state that is not 16-byte aligned keeps the per-lane stores.
inline bool aligned16(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

// STRATA_GDN_BLOCK_GRF=1: the per-slot recurrence kernels (gdn_step_plain, gdn_step_split's recurrence,
// gdn_step_prefetch, gdn_state_commit) read and write the state by sub-group block loads and stores and are built with
// the large register file, the cure of STRATA_BATCH_GDN_FORM=4: IGC builds them with 128 registers and spills 7-9 KiB
// a work-item on the B70 (the 32 rows' per-lane 64-bit addresses, kept from the loads to the stores).  The same bits.
constexpr bool kGdnBlockGrfDefault = false;
int g_gdn_block_grf = -1;   // -1: not yet read
bool gdn_block_grf() {
    if (g_gdn_block_grf < 0) {
        const char* v = std::getenv("STRATA_GDN_BLOCK_GRF");
        g_gdn_block_grf = v == nullptr ? kGdnBlockGrfDefault : std::strtol(v, nullptr, 10) != 0;
    }
    return g_gdn_block_grf != 0;
}
inline uint32_t read_host(const uint32_t* p) {
    const uint32_t v = *reinterpret_cast<const volatile uint32_t*>(p);
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    return v;
}

}  // namespace

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin, bool commit) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT ||
        (commit && (n_tok != 1 || t_begin != 0)))
        fail("gdn_conv_l2_multi: invalid arguments");
    float* hist_out = commit ? const_cast<float*>(history) : nullptr;   // a work-item reads and writes its channel only
    const int C = channels;
    const auto e = Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(S / WARP), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) n_tok, (size_t) C}, {1, S}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = t_begin + (int) it.get_group(0);
            const int tid = (int) it.get_local_id(1);
            const int c = (int) it.get_group(1) * S + tid;
            // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
            float win[3];
            for (int j = 0; j < 3; ++j) {
                const int src = t + j;   // index into [hist(3) | x...]
                win[j] = src < 3 ? history[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
            }
            const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
            const float sum = v0 * conv_w[c * 4] + v1 * conv_w[c * 4 + 1] + v2 * conv_w[c * 4 + 2] + x * conv_w[c * 4 + 3];
            float y = sum / (1.0f + sycl::exp(-sum));
            if ((int) it.get_group(1) < qk_heads) {   // uniform over the work-group
                const float sq = warp_sum(sg, y * y);
                if ((tid & 31) == 0) part[tid >> 5] = sq;
                sycl::group_barrier(it.get_group());
                const float ss = part[0] + part[1] + part[2] + part[3];
                y *= sycl::rsqrt(ss + eps);
            }
            h[(size_t) t * C + c] = y;
            if (hist_out != nullptr) {   // the window keeps its one token: [h1, h2, x], gdn_conv_commit's n_keep 1
                float* ho = hist_out + (size_t) c * 3;
                ho[0] = v1;
                ho[1] = v2;
                ho[2] = x;
            }
        });
    });
    done(stream, e, "gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    const int C = channels;
    const auto e = Q(stream).parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        const int n = *n_keep;
        if (n <= 0) return;
        float seq[3];
        for (int j = 0; j < 3; ++j) {
            const int src = n + j;   // the last three of [hist(3) | x_0..x_{n-1}]
            seq[j] = src < 3 ? history[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
        }
        history[c * 3] = seq[0];
        history[c * 3 + 1] = seq[1];
        history[c * 3 + 2] = seq[2];
    });
    done(stream, e, "gdn_conv_commit");
}

namespace {
// gdn_conv_commit for `n_layers` layers in one launch (gdn_commit_layers)
void gdn_conv_commit_layers(float* hist0, int64_t hist_stride, const float* qkv0, int64_t qkv_stride, int channels,
                            int n_layers, const int32_t* n_keep, void* stream) {
    const int C = channels;
    const auto e = Q(stream).parallel_for(sycl::range<2>((size_t) n_layers, (size_t) C), [=](sycl::id<2> id) {
        const int c = (int) id[1];
        const int n = *n_keep;
        if (n <= 0) return;
        float* history = hist0 + (size_t) id[0] * (size_t) hist_stride;
        const float* qkv = qkv0 + (size_t) id[0] * (size_t) qkv_stride;
        float seq[3];
        for (int j = 0; j < 3; ++j) {
            const int src = n + j;   // the last three of [hist(3) | x_0..x_{n-1}], as gdn_conv_commit
            seq[j] = src < 3 ? history[(size_t) c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
        }
        history[(size_t) c * 3] = seq[0];
        history[(size_t) c * 3 + 1] = seq[1];
        history[(size_t) c * 3 + 2] = seq[2];
    });
    done(stream, e, "gdn_conv_commit_layers");
}
}  // namespace

namespace {
// gdn_ab_multi for at most MAX_T tokens; EXACT: exactly MAX_T (the token loop fixed at compile time, upstream ae3b249f).
// A lane unpacks its eight BF16 weights once for all tokens; the FMA order is the same as before.
template <int MAX_T, bool EXACT>
sycl::event gdn_ab_launch(sycl::queue& q, const float* x, const uint16_t* w_alpha, const uint16_t* w_beta,
                          const float* dt, const float* ssm_a, float* gate, float* beta, int n, int h_v, int T_arg) {
    const size_t groups = (size_t) ((2 * h_v + 7) / 8);
    return q.parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const int T = EXACT ? MAX_T : T_arg;
        const sycl::sub_group sg = it.get_sub_group();
        const int row = (int) it.get_group(0) * 8 + (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (row >= 2 * h_v) return;   // uniform over the sub-group
        const bool is_beta = row >= h_v;
        const int r = is_beta ? row - h_v : row;
        const uint32_t* w = reinterpret_cast<const uint32_t*>((is_beta ? w_beta : w_alpha) + (size_t) r * n);
        auto lo = [](uint32_t u) { return sycl::bit_cast<float>(u << 16); };
        auto hi = [](uint32_t u) { return sycl::bit_cast<float>(u & 0xffff0000u); };
        float acc[MAX_T];
        #pragma unroll
        for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
        for (int j = lane; j < n / 8; j += 32) {
            const uint32_t* wj = w + (size_t) j * 4;
            const uint32_t u0 = wj[0], u1 = wj[1], u2 = wj[2], u3 = wj[3];
            const float w0 = lo(u0), w1 = hi(u0), w2 = lo(u1), w3 = hi(u1);
            const float w4 = lo(u2), w5 = hi(u2), w6 = lo(u3), w7 = hi(u3);
            #pragma unroll
            for (int t = 0; t < MAX_T; ++t) {
                if (!EXACT && t >= T) break;
                const float* xa = x + (size_t) t * n + j * 8;
                float a = acc[t];
                a = sycl::fma(w0, xa[0], a); a = sycl::fma(w1, xa[1], a);
                a = sycl::fma(w2, xa[2], a); a = sycl::fma(w3, xa[3], a);
                a = sycl::fma(w4, xa[4], a); a = sycl::fma(w5, xa[5], a);
                a = sycl::fma(w6, xa[6], a); a = sycl::fma(w7, xa[7], a);
                acc[t] = a;
            }
        }
        #pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (!EXACT && t >= T) break;
            const float a = warp_sum(sg, acc[t]);
            if (lane != 0) continue;
            if (is_beta) {
                beta[(size_t) t * h_v + r] = 1.0f / (1.0f + sycl::exp(-a));
            } else {
                const float v = a + dt[r];
                const float sp = v > 20.0f ? v : sycl::log1p(sycl::exp(v));
                gate[(size_t) t * h_v + r] = sp * ssm_a[r];
            }
        }
    });
}
}  // namespace

namespace {
// STRATA_GDN_AB_FORM (homelab; default 0, the launch above): the same arithmetic over more sub-groups.  The launch above
// gives one sub-group each of the 2 h_v output rows and walks every token in it: 96 sub-groups on a 256-EU card, and
// past 6 tokens its token loop is not fixed at compile time (8 rows 36 us, 16 rows 77 us against a 1.2 us byte floor).
// Here a sub-group takes one row for TC tokens (a "unit"): each token's sum is the same lane's FMAs in the same order
// and the same butterfly, so gate and beta are the launch above's bits (gdn_ab_parity).
//   1, 2, 3: a launch of its own, TC 1, 2, 4
//   4, 5, 6: the same units as extra work-groups of the batch window's conv launch (gdn_conv_l2_slots with GdnAbArgs:
//            one launch fewer a GDN layer); gdn_ab_multi itself takes TC 1, 2, 4 then
constexpr int kAbForms = 7;
int g_ab_form = -1;
int ab_tc(int form) { return form == 1 || form == 4 ? 1 : form == 2 || form == 5 ? 2 : 4; }

template <int TC>
inline void gdn_ab_unit(const sycl::sub_group& sg, int lane, int row, int t0, int T, const float* x,
                        const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                        float* gate, float* beta, int n, int h_v) {
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint32_t* w = reinterpret_cast<const uint32_t*>((is_beta ? w_beta : w_alpha) + (size_t) r * n);
    auto lo = [](uint32_t u) { return sycl::bit_cast<float>(u << 16); };
    auto hi = [](uint32_t u) { return sycl::bit_cast<float>(u & 0xffff0000u); };
    float acc[TC];
#pragma unroll
    for (int t = 0; t < TC; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint32_t* wj = w + (size_t) j * 4;
        const uint32_t u0 = wj[0], u1 = wj[1], u2 = wj[2], u3 = wj[3];
        const float w0 = lo(u0), w1 = hi(u0), w2 = lo(u1), w3 = hi(u1);
        const float w4 = lo(u2), w5 = hi(u2), w6 = lo(u3), w7 = hi(u3);
#pragma unroll
        for (int t = 0; t < TC; ++t) {
            if (TC > 1 && t0 + t >= T) break;
            const float* xa = x + (size_t) (t0 + t) * n + j * 8;
            float a = acc[t];
            a = sycl::fma(w0, xa[0], a); a = sycl::fma(w1, xa[1], a);
            a = sycl::fma(w2, xa[2], a); a = sycl::fma(w3, xa[3], a);
            a = sycl::fma(w4, xa[4], a); a = sycl::fma(w5, xa[5], a);
            a = sycl::fma(w6, xa[6], a); a = sycl::fma(w7, xa[7], a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < TC; ++t) {
        if (TC > 1 && t0 + t >= T) break;
        const float a = warp_sum(sg, acc[t]);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) (t0 + t) * h_v + r] = 1.0f / (1.0f + sycl::exp(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : sycl::log1p(sycl::exp(v));
            gate[(size_t) (t0 + t) * h_v + r] = sp * ssm_a[r];
        }
    }
}

// unit u of a window: row u % (2 h_v), tokens TC (u / (2 h_v)) ..
template <int TC>
inline void gdn_ab_unit_at(const sycl::sub_group& sg, int lane, int u, const GdnAbArgs& a) {
    const int rows = 2 * a.h_v;
    if (u >= rows * ((a.n_tok + TC - 1) / TC)) return;   // uniform over the sub-group
    gdn_ab_unit<TC>(sg, lane, u % rows, TC * (u / rows), a.n_tok, a.x, a.w_alpha, a.w_beta, a.dt, a.ssm_a, a.gate,
                    a.beta, a.n_embd, a.h_v);
}

template <int TC>
sycl::event gdn_ab_split(sycl::queue& q, const GdnAbArgs& a) {
    const int units = 2 * a.h_v * ((a.n_tok + TC - 1) / TC);
    const size_t groups = (size_t) ((units + 7) / 8);
    return q.parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        gdn_ab_unit_at<TC>(sg, (int) sg.get_local_linear_id(), (int) it.get_group(0) * 8 + (int) sg.get_group_linear_id(), a);
    });
}
}  // namespace

void gdn_ab_set_form(int form) { g_ab_form = form >= 0 && form < kAbForms ? form : 0; }
int gdn_ab_form() {
    if (g_ab_form < 0) {
        const char* v = std::getenv("STRATA_GDN_AB_FORM");
        gdn_ab_set_form(v != nullptr ? (int) std::strtol(v, nullptr, 10) : 0);
    }
    return g_ab_form;
}
bool gdn_ab_fused_in_conv() { return gdn_ab_form() >= 4; }

void gdn_ab_activate(float* gate, float* beta, const float* dt, const float* ssm_a,
                     int h_v, int n_tok, void* stream) {
    if (!gate || !beta || !dt || !ssm_a || h_v < 1 || n_tok < 1 || n_tok > kVerifyMaxT)
        fail("gdn_ab_activate: invalid arguments");
    auto e = Q(stream).parallel_for(sycl::range<1>((size_t) h_v * n_tok), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int r = (int) (i % h_v);
        const float v = gate[i] + dt[r];
        gate[i] = (v > 20.f ? v : sycl::log1p(sycl::exp(v))) * ssm_a[r];
        beta[i] = 1.f / (1.f + sycl::exp(-beta[i]));
    });
    done(stream, e, "gdn_ab_activate");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) fail("gdn_ab_multi: invalid arguments");
    auto& q = Q(stream);
    sycl::event e;
    if (const int form = gdn_ab_form(); form > 0) {
        const GdnAbArgs a{x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok};
        const int tc = ab_tc(form);
        e = tc == 1 ? gdn_ab_split<1>(q, a) : tc == 2 ? gdn_ab_split<2>(q, a) : gdn_ab_split<4>(q, a);
        done(stream, e, "gdn_ab_multi");
        return;
    }
    switch (n_tok) {
        case 1: e = gdn_ab_launch<1, true>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 2: e = gdn_ab_launch<2, true>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 3: e = gdn_ab_launch<3, true>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 4: e = gdn_ab_launch<4, true>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 5: e = gdn_ab_launch<5, true>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        case 6: e = gdn_ab_launch<6, true>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok); break;
        default: e = n_tok <= kVerifyDefaultT ? gdn_ab_launch<kVerifyDefaultT, false>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok)
                    : n_tok <= 16 ? gdn_ab_launch<16, false>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok)
                    : gdn_ab_launch<kVerifyMaxT, false>(q, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    }
    done(stream, e, "gdn_ab_multi");
}

namespace {
// upstream e3c3d6ba: the recurrence with each head's 128 state columns over 4 work-groups of 32
// columns (128 work-items: the same (column, row group) work-items, each with the same 32 state rows), 4x the
// work-groups of the one-group-a-head kernel.  Every column's arithmetic is that kernel's; the output norm, which needs
// a head's 128 columns, follows in a second kernel: this one leaves the unnormalized output in y, and the norm sums
// the squares by the same sub-groups (columns 32w .. 32w+31, the same butterfly) in the same order.
constexpr int GS_COLS = 32;
template <bool Q8, bool Fast = false>
sycl::event gdn_step_split(sycl::queue& q, float* state, const float* hbuf, int C, const float* gate, const float* beta,
                           const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int T,
                           const int32_t* n_keep, int t_out_begin, uint8_t* y_q8_1) {
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd), sq(sycl::range<1>(S), hd),
            red(sycl::range<1>((size_t) RG * GS_COLS), hd);
        const auto body = [=](const sycl::nd_item<1>& it) {
            const int grp = (int) it.get_group(0), head = grp / (S / GS_COLS), c0 = (grp % (S / GS_COLS)) * GS_COLS;
            const int tid = (int) it.get_local_id(0), lc = tid % GS_COLS, rg = tid / GS_COLS, col = c0 + lc;
            const int qh = head % h_k;
            const int qk = S * h_k;
            const int value_dim = S * h_v;
            const int n = n_keep ? *n_keep : T;
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
            if constexpr (Fast) {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = state_load<true>(it.get_sub_group(), base - lc + r * row_stride, lc);
            } else {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            }
            for (int t = 0; t < n; ++t) {
                const float* ht = hbuf + (size_t) t * C;
                sycl::group_barrier(it.get_group());
                if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
                sycl::group_barrier(it.get_group());
                const float g = sycl::exp(gate[(size_t) t * h_v + head]);
                float kv = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                red[rg * GS_COLS + lc] = kv;
                sycl::group_barrier(it.get_group());
                const float kv_col = red[lc] + red[GS_COLS + lc] + red[2 * GS_COLS + lc] + red[3 * GS_COLS + lc];
                const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
                float o = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) {
                    s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                    o = sycl::fma(s[r], sq[rg * RPG + r], o);
                }
                sycl::group_barrier(it.get_group());
                red[rg * GS_COLS + lc] = o;
                sycl::group_barrier(it.get_group());
                if (rg == 0 && t >= t_out_begin)
                    y[(size_t) t * value_dim + (size_t) (head * S + col)] =
                        (red[lc] + red[GS_COLS + lc] + red[2 * GS_COLS + lc] + red[3 * GS_COLS + lc]) * sycl::rsqrt((float) S);
            }
            if (n_keep != nullptr && n > 0) {
                if constexpr (Fast) {
                    #pragma unroll
                    for (int r = 0; r < RPG; ++r) state_store<true>(it.get_sub_group(), s[r], base - lc + r * row_stride, lc);
                } else {
                    #pragma unroll
                    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
                }
            }
        };
        const sycl::nd_range<1> nd((size_t) h_v * (S / GS_COLS) * GS_COLS * RG, (size_t) GS_COLS * RG);
        if constexpr (Fast)
            hd.parallel_for(nd, LargeGrf<GS_COLS * RG, decltype(body)>{body});
        else
            hd.parallel_for(nd, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] { body(it); });
    });
    const int rows = T - t_out_begin;
    if (rows <= 0) return q.ext_oneapi_submit_barrier();
    // y = oc * rsqrt(mean(oc^2) + eps) * gamma * sigmoid(z) a head and token, with the one-group kernel's sums
    return q.submit([&](sycl::handler& hd) {
        const bool silu = gdn_silu_gate();   // the closing norm's gate (gdn.hpp)
        sycl::local_accessor<float, 1> wsum(sycl::range<1>(S / WARP), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) h_v * rows * S, S), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int grp = (int) it.get_group(0), head = grp % h_v, t = grp / h_v + t_out_begin;
            const int col = (int) it.get_local_id(0);
            const int n = n_keep ? *n_keep : T;
            if (t >= n) return;   // the whole group
            const int value_dim = S * h_v;
            const size_t at = (size_t) t * value_dim + (size_t) (head * S + col);
            const float oc = y[at];
            const float sq_part = warp_sum(sg, oc * oc);
            if ((col & 31) == 0) wsum[col >> 5] = sq_part;
            sycl::group_barrier(it.get_group());
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float) S + eps);
            const float zz = z[at];
            const float gz = 1.0f / (1.0f + sycl::exp(-zz));
            const float out = oc * scale * gamma[col] * (silu ? zz * gz : gz);
            y[at] = out;
            if constexpr (Q8)
                q8_1_store(sg, out, y_q8_1 + ((size_t) (t - t_out_begin) * value_dim + (size_t) (head * S + col)) / 32 * 36,
                           col & 31);
        });
    });
}
}  // namespace

namespace {
// upstream's gdn_step_commit_kernel: the commit's state only, without the outputs the window's forms also compute (the
// queries, the second reduction and its two barriers a token).  The split's work-items and the same state arithmetic in
// the same order, so the state is every form's bit for bit.
// `n_layers` layers in one launch (the commit's layers, upstream 6f3e3810): layer z's state at state + z * state_stride,
// its rows at hbuf + z * h_stride and gate|beta + z * gb_stride; each layer's work-groups as its own launch's.
template <bool Fast>
sycl::event gdn_state_commit_launch(sycl::queue& q, float* state0, const float* hbuf0, int C, const float* gate0,
                                    const float* beta0, int h_k, int h_v, const int32_t* n_keep, int n_layers,
                                    int64_t state_stride, int64_t h_stride, int64_t gb_stride) {
    return q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd), red(sycl::range<1>((size_t) RG * GS_COLS), hd);
        const int per_layer = h_v * (S / GS_COLS);
        const auto body = [=](const sycl::nd_item<1>& it) {
            const int n = *n_keep;
            if (n <= 0) return;   // the whole launch
            const int z = (int) it.get_group(0) / per_layer;
            float* state = state0 + (size_t) z * (size_t) state_stride;
            const float* hbuf = hbuf0 + (size_t) z * (size_t) h_stride;
            const float* gate = gate0 + (size_t) z * (size_t) gb_stride;
            const float* beta = beta0 + (size_t) z * (size_t) gb_stride;
            const int grp = (int) it.get_group(0) % per_layer, head = grp / (S / GS_COLS), c0 = (grp % (S / GS_COLS)) * GS_COLS;
            const int tid = (int) it.get_local_id(0), lc = tid % GS_COLS, rg = tid / GS_COLS, col = c0 + lc;
            const int qh = head % h_k;
            const int qk = S * h_k;
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
            if constexpr (Fast) {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = state_load<true>(it.get_sub_group(), base - lc + r * row_stride, lc);
            } else {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            }
            for (int t = 0; t < n; ++t) {
                const float* ht = hbuf + (size_t) t * C;
                sycl::group_barrier(it.get_group());
                if (tid < S) sk[tid] = ht[qk + qh * S + tid];
                sycl::group_barrier(it.get_group());
                const float g = sycl::exp(gate[(size_t) t * h_v + head]);
                float kv = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                red[rg * GS_COLS + lc] = kv;
                sycl::group_barrier(it.get_group());
                const float kv_col = red[lc] + red[GS_COLS + lc] + red[2 * GS_COLS + lc] + red[3 * GS_COLS + lc];
                const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
            }
            if constexpr (Fast) {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) state_store<true>(it.get_sub_group(), s[r], base - lc + r * row_stride, lc);
            } else {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
            }
        };
        const sycl::nd_range<1> nd((size_t) n_layers * per_layer * GS_COLS * RG, (size_t) GS_COLS * RG);
        if constexpr (Fast)
            hd.parallel_for(nd, LargeGrf<GS_COLS * RG, decltype(body)>{body});
        else
            hd.parallel_for(nd, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] { body(it); });
    });
}
sycl::event gdn_state_commit(sycl::queue& q, float* state0, const float* hbuf0, int C, const float* gate0,
                             const float* beta0, int h_k, int h_v, const int32_t* n_keep, int n_layers = 1, int64_t state_stride = 0,
                             int64_t h_stride = 0, int64_t gb_stride = 0) {
    // STRATA_GDN_BLOCK_GRF: every layer's state 16-byte aligned (its start and the stride between layers)
    if (gdn_block_grf() && aligned16(state0) && (n_layers == 1 || state_stride % 4 == 0))
        return gdn_state_commit_launch<true>(q, state0, hbuf0, C, gate0, beta0, h_k, h_v, n_keep, n_layers, state_stride,
                                             h_stride, gb_stride);
    return gdn_state_commit_launch<false>(q, state0, hbuf0, C, gate0, beta0, h_k, h_v, n_keep, n_layers, state_stride,
                                          h_stride, gb_stride);
}
}  // namespace

namespace {
// One work-group a head (512 work-items: (column, row group)).
template <bool Q8, bool Fast = false>
sycl::event gdn_step_plain(sycl::queue& q, float* state, const float* hbuf, int C, const float* gate, const float* beta,
                           const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int T,
                           const int32_t* n_keep, int t_out_begin, uint8_t* y_q8_1) {
    return q.submit([&](sycl::handler& hd) {
        const bool silu = gdn_silu_gate();   // the closing norm's gate (gdn.hpp)
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd), sq(sycl::range<1>(S), hd),
            red(sycl::range<1>(RG * S), hd), wsum(sycl::range<1>(S * RG / WARP), hd);
        // one work-group per head; work-item tid = rg * 128 + col, as CUDA's (col, rg) block linearises
        // the q8_1 image takes registers: CUDA needs the work-group size bound for that kernel only
        const auto body = [=](const sycl::nd_item<1>& it) {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0), tid = (int) it.get_local_id(0), col = tid % S, rg = tid / S;
            const int qh = head % h_k;
            const int qk = S * h_k;   // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
            const int value_dim = S * h_v;
            const int n = n_keep ? *n_keep : T;
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
            if constexpr (Fast) {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = state_load<true>(sg, base - (tid & 31) + r * row_stride, tid & 31);
            } else {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            }
            for (int t = 0; t < n; ++t) {
                const float* ht = hbuf + (size_t) t * C;
                sycl::group_barrier(it.get_group());   // the previous token is done with sk/sq/red/wsum
                if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
                sycl::group_barrier(it.get_group());
                const float g = sycl::exp(gate[(size_t) t * h_v + head]);
                float kv = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                red[rg * S + col] = kv;
                sycl::group_barrier(it.get_group());
                const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
                const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
                float o = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) {
                    s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                    o = sycl::fma(s[r], sq[rg * RPG + r], o);
                }
                sycl::group_barrier(it.get_group());
                red[rg * S + col] = o;
                sycl::group_barrier(it.get_group());
                float oc = 0.0f, sq_part = 0.0f;
                if (rg == 0) {
                    oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * sycl::rsqrt((float) S);
                    sq_part = oc * oc;
                }
                if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
                sq_part = warp_sum(sg, sq_part);
                if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
                sycl::group_barrier(it.get_group());
                if (rg == 0) {
                    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                    const float scale = sycl::rsqrt(ss / (float) S + eps);
                    const float zz = z[(size_t) t * value_dim + head * S + col];
                    const float gz = 1.0f / (1.0f + sycl::exp(-zz));
                    const float out = oc * scale * gamma[col] * (silu ? zz * gz : gz);
                    y[(size_t) t * value_dim + (size_t) (head * S + col)] = out;
                    if constexpr (Q8)
                        q8_1_store(sg, out, y_q8_1 + ((size_t) (t - t_out_begin) * value_dim + (size_t) (head * S + col)) / 32 * 36,
                                   tid & 31);
                }
            }
            if (n_keep != nullptr && n > 0) {
                if constexpr (Fast) {
                    #pragma unroll
                    for (int r = 0; r < RPG; ++r) state_store<true>(sg, s[r], base - (tid & 31) + r * row_stride, tid & 31);
                } else {
                    #pragma unroll
                    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
                }
            }
        };
        if constexpr (Fast)
            hd.parallel_for(sycl::nd_range<1>((size_t) h_v * S * RG, (size_t) S * RG), LargeGrf<S * RG, decltype(body)>{body});
        else if constexpr (Q8)
            hd.parallel_for(sycl::nd_range<1>((size_t) h_v * S * RG, (size_t) S * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP), sycl::reqd_work_group_size(S * RG)]] { body(it); });
        else
            hd.parallel_for(sycl::nd_range<1>((size_t) h_v * S * RG, (size_t) S * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] { body(it); });
    });
}

// upstream ae3b249f: the same arithmetic with the next token's q/k staged while the current one runs.
template <bool Q8, bool Fast = false>
sycl::event gdn_step_prefetch(sycl::queue& q, float* state, const float* hbuf, int C, const float* gate, const float* beta,
                           const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int T,
                           const int32_t* n_keep, int t_out_begin, uint8_t* y_q8_1) {
    return q.submit([&](sycl::handler& hd) {
        const bool silu = gdn_silu_gate();   // the closing norm's gate (gdn.hpp)
        // sk/sq: two buffers, the next token's staged while the current one runs; red_kv and red_o apart, so a token
        // needs four barriers instead of six
        constexpr size_t two_s = 2 * (size_t) S, rg_s = (size_t) RG * S;
        sycl::local_accessor<float, 1> sk(sycl::range<1>(two_s), hd), sq(sycl::range<1>(two_s), hd),
            red_kv(sycl::range<1>(rg_s), hd), red_o(sycl::range<1>(rg_s), hd), wsum(sycl::range<1>(S / WARP), hd);
        // the q8_1 image takes registers: CUDA needs the work-group size bound for that kernel only
        const auto body = [=](const sycl::nd_item<1>& it) {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = (int) it.get_group(0), tid = (int) it.get_local_id(0), col = tid % S, rg = tid / S;
            const int qh = head % h_k;
            const int qk = S * h_k;   // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
            const int value_dim = S * h_v;
            const int n = n_keep ? *n_keep : T;
            if (n > 0) {
                if (rg == 0) sk[col] = hbuf[qk + qh * S + col];
                else if (rg == 1) sq[col] = hbuf[qh * S + col];
            }
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
            if constexpr (Fast) {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = state_load<true>(sg, base - (tid & 31) + r * row_stride, tid & 31);
            } else {
                #pragma unroll
                for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            }
            for (int t = 0; t < n; ++t) {
                const int cur = (t & 1) * S, nxt = ((t + 1) & 1) * S;
                const float* ht = hbuf + (size_t) t * C;
                // every work-item is done with the previous token (and its buffer, now nxt); the current one is staged
                sycl::group_barrier(it.get_group());
                if (t + 1 < n) {
                    const float* hn = ht + C;
                    if (rg == 0) sk[nxt + col] = hn[qk + qh * S + col];
                    else if (rg == 1) sq[nxt + col] = hn[qh * S + col];
                }
                const float g = sycl::exp(gate[(size_t) t * h_v + head]);
                float kv = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[cur + rg * RPG + r], kv);
                red_kv[rg * S + col] = kv;
                sycl::group_barrier(it.get_group());
                const float kv_col = red_kv[col] + red_kv[S + col] + red_kv[2 * S + col] + red_kv[3 * S + col];
                const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
                float o = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) {
                    s[r] = sycl::fma(g, s[r], sk[cur + rg * RPG + r] * delta);
                    o = sycl::fma(s[r], sq[cur + rg * RPG + r], o);
                }
                if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
                red_o[rg * S + col] = o;
                sycl::group_barrier(it.get_group());
                float oc = 0.0f;
                if (rg == 0) {
                    oc = (red_o[col] + red_o[S + col] + red_o[2 * S + col] + red_o[3 * S + col]) * sycl::rsqrt((float) S);
                    const float sq_part = warp_sum(sg, oc * oc);
                    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
                }
                sycl::group_barrier(it.get_group());
                if (rg == 0) {
                    const size_t at = (size_t) t * value_dim + (size_t) (head * S + col);
                    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                    const float scale = sycl::rsqrt(ss / (float) S + eps);
                    const float gz = 1.0f / (1.0f + sycl::exp(-z[at]));
                    const float out = oc * scale * gamma[col] * (silu ? z[at] * gz : gz);
                    y[at] = out;
                    if constexpr (Q8)
                        q8_1_store(sg, out, y_q8_1 + ((size_t) (t - t_out_begin) * value_dim + (size_t) (head * S + col)) / 32 * 36,
                                   tid & 31);
                }
            }
            if (n_keep != nullptr && n > 0) {
                if constexpr (Fast) {
                    #pragma unroll
                    for (int r = 0; r < RPG; ++r) state_store<true>(sg, s[r], base - (tid & 31) + r * row_stride, tid & 31);
                } else {
                    #pragma unroll
                    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
                }
            }
        };
        if constexpr (Fast)
            hd.parallel_for(sycl::nd_range<1>((size_t) h_v * S * RG, (size_t) S * RG), LargeGrf<S * RG, decltype(body)>{body});
        else if constexpr (Q8)
            hd.parallel_for(sycl::nd_range<1>((size_t) h_v * rg_s, rg_s), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP), sycl::reqd_work_group_size(S * RG)]] { body(it); });
        else
            hd.parallel_for(sycl::nd_range<1>((size_t) h_v * rg_s, rg_s), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] { body(it); });
    });
}

// The three forms give the same bits; which is fastest depends on the device and the window (the split's 4x
// work-groups and the prefetch's fewer barriers paid on the RTX 4070, the prefetch cost the B70 half again).
// gdn_step_tune times them once per window size on the device; until then the one-group form runs.
enum GdnForm : int8_t { kGdnPlain = 0, kGdnSplit = 1, kGdnPrefetch = 2, kGdnForms = 3 };
struct GdnChoice { std::array<int8_t, kVerifyMaxT + 1> form{}; };
core::PerDevice<GdnChoice>& gdn_choices() {
    static core::PerDevice<GdnChoice> choices;
    return choices;
}
int g_gdn_forced = -2;   // -2: not yet read from STRATA_GDN_STEP; -1: the tuned form
int gdn_forced() {   // STRATA_GDN_STEP=0|1|2: one form always (one group a head, split, prefetch)
    if (g_gdn_forced == -2) {
        const char* v = std::getenv("STRATA_GDN_STEP");
        const long f = v == nullptr ? -1 : std::strtol(v, nullptr, 10);
        g_gdn_forced = f >= 0 && f < kGdnForms ? (int) f : -1;
    }
    return g_gdn_forced;
}
template <bool Q8, bool Fast>
sycl::event gdn_step_form_as(int form, sycl::queue& q, float* state, const float* hbuf, int C, const float* gate,
                             const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                             int T, const int32_t* n_keep, int t_out_begin, uint8_t* y_q8_1) {
    if (form == kGdnSplit)
        return gdn_step_split<Q8, Fast>(q, state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T, n_keep, t_out_begin,
                                        y_q8_1);
    if (form == kGdnPrefetch)
        return gdn_step_prefetch<Q8, Fast>(q, state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T, n_keep,
                                           t_out_begin, y_q8_1);
    return gdn_step_plain<Q8, Fast>(q, state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T, n_keep, t_out_begin,
                                    y_q8_1);
}
sycl::event gdn_step_form(int form, sycl::queue& q, float* state, const float* hbuf, int C, const float* gate, const float* beta,
                           const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int T,
                           const int32_t* n_keep, int t_out_begin, uint8_t* y_q8_1) {
    const bool fast = gdn_block_grf() && aligned16(state);   // STRATA_GDN_BLOCK_GRF
    if (y_q8_1 != nullptr)
        return fast ? gdn_step_form_as<true, true>(form, q, state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T,
                                                   n_keep, t_out_begin, y_q8_1)
                    : gdn_step_form_as<true, false>(form, q, state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T,
                                                    n_keep, t_out_begin, y_q8_1);
    return fast ? gdn_step_form_as<false, true>(form, q, state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T, n_keep,
                                                t_out_begin, nullptr)
                : gdn_step_form_as<false, false>(form, q, state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T,
                                                 n_keep, t_out_begin, nullptr);
}
}  // namespace

void gdn_step_set_form(int form) {
    if (form < -1 || form >= kGdnForms) fail("gdn_step_set_form: no such form");
    g_gdn_forced = form;
}
void gdn_step_set_block_grf(bool on) { g_gdn_block_grf = on; }

void gdn_step_tune(int C, int h_k, int h_v, void* stream) {
    if (C <= 0 || h_k <= 0 || h_v <= 0 || h_v % h_k) fail("gdn_step_tune: invalid arguments");
    auto& q = Q(stream);
    gdn_choices().get(q.get_device(), [&] {
        GdnChoice c;
        const int forced = gdn_forced();
        if (forced >= 0) {
            c.form.fill((int8_t) forced);
            return c;
        }
        // scratch inputs of the model's shape; small values keep the recurrence finite
        const size_t n_state = (size_t) S * S * h_v, n_h = (size_t) kVerifyMaxT * C;
        const size_t n_g = (size_t) kVerifyMaxT * h_v, n_y = (size_t) kVerifyMaxT * S * h_v;
        float* buf = sycl::malloc_device<float>(n_state + n_h + 2 * n_g + 2 * n_y + S + n_y / 32 * 9, q);
        if (buf == nullptr) fail("gdn_step_tune: device allocation failed");
        float *state = buf, *h = state + n_state, *gate = h + n_h, *beta = gate + n_g, *z = beta + n_g;
        float *y = z + n_y, *gamma = y + n_y;
        auto* yq = reinterpret_cast<uint8_t*>(gamma + S);   // the window's q8_1 image (36 bytes a 32-value block)
        q.fill(buf, 0.01f, n_state + n_h + 2 * n_g + 2 * n_y + S).wait();
        // timed as the engine runs them: 32 launches of each form in a graph (submitted one by one, the host's launch
        // cost hid the kernels' on CUDA), the forms' graphs replayed in turns (a fresh process's B70 was still raising
        // its clock under the first ones), the first round a warm-up, each form's fastest round kept
        namespace sx = sycl::ext::oneapi::experimental;
        using Exec = sx::command_graph<sx::graph_state::executable>;
        for (int T = 1; T <= verify_max_t(); ++T) {
            std::vector<Exec> exec;
            for (int form = 0; form < kGdnForms; ++form) {
                sx::command_graph<sx::graph_state::modifiable> graph(q.get_context(), q.get_device());
                graph.begin_recording(q);
                for (int i = 0; i < 32; ++i)
                    gdn_step_form(form, q, state, h, C, gate, beta, z, gamma, 1e-6f, y, h_k, h_v, T, nullptr, 0, yq);
                graph.end_recording(q);
                exec.push_back(graph.finalize());
            }
            double t_min[kGdnForms] = {};
            for (int round = 0; round < 6; ++round)
                for (int form = 0; form < kGdnForms; ++form) {
                    const auto t0 = std::chrono::steady_clock::now();
                    q.ext_oneapi_graph(exec[form]);
                    q.wait();
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    if (round > 0 && (t_min[form] == 0.0 || us < t_min[form])) t_min[form] = us;
                }
            c.form[T] = kGdnPlain;
            for (int form = 1; form < kGdnForms; ++form)
                if (t_min[form] < t_min[c.form[T]]) c.form[T] = (int8_t) form;
        }
        sycl::free(buf, q);
        char forms[kVerifyMaxT + 1] = {};
        for (int T = 1; T <= verify_max_t(); ++T) forms[T - 1] = (char) ('0' + c.form[T]);
        std::fprintf(stderr, "strata: GDN step forms for windows of 1..%d tokens: %s (0 one group a head, 1 split, "
                     "2 prefetch)\n", verify_max_t(), forms);
        return c;
    });
}

namespace {
// STRATA_DBG_GDN=1 (upstream dfc8f74d): the window count a commit-only launch reads from device memory is checked
// first by a one-work-item kernel, which hands the form 0 in place of a count past the window's rows (the commit
// skipped, where the fault would be a read past the window) and leaves the count in host memory: the next call
// reports it (no device printf, which CUDA's link warns about).  The default path is unchanged.
struct GdnDbg { int32_t* keep; int32_t* bad; };
const int32_t* gdn_dbg_keep(sycl::queue& q, const int32_t* n_keep, int n_max) {
    static const bool dbg = [] { const char* e = std::getenv("STRATA_DBG_GDN"); return e && e[0] == '1'; }();
    if (!dbg) return n_keep;
    static core::PerDevice<GdnDbg> per_device;
    const GdnDbg d = per_device.get(q.get_device(), [&q] {
        GdnDbg r{sycl::malloc_device<int32_t>(1, q.get_device(), q.get_context()),
                 sycl::malloc_host<int32_t>(2, q.get_context())};
        if (!r.keep || !r.bad) throw core::DeviceError("gdn_step_norm_multi: allocation failed");
        r.bad[0] = r.bad[1] = 0;
        return r;
    });
    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::system> seen(d.bad[0]);
    if (const int32_t n = seen.exchange(0); n != 0)
        std::fprintf(stderr, "strata DBG: gdn_step commit: n_keep %d is past the window's %d rows: commit skipped (#937)\n",
                     n, d.bad[1]);
    int32_t *keep = d.keep, *bad = d.bad;
    q.single_task([=] {
        const int n = *n_keep;
        if (n > n_max) { bad[1] = n_max; bad[0] = n; }
        *keep = n > n_max ? 0 : n;
    });
    return keep;
}
}  // namespace

void gdn_step_norm_multi(float* state, const float* hbuf, int C, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, void* y_q8_1) {
    if (!state || !hbuf || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT)
        fail("gdn_step_norm_multi: invalid arguments");
    auto& q = Q(stream);
    const int forced = gdn_forced();
    const int form = forced >= 0 ? forced : gdn_choices().get(q.get_device(), [] { return GdnChoice{}; }).form[n_tok];
    if (y_q8_1 != nullptr && (S * h_v) % 32 != 0) fail("gdn_step_norm_multi: q8_1 blocks need 32 | value_dim");
    if (n_keep != nullptr && t_out_begin >= n_tok) n_keep = gdn_dbg_keep(q, n_keep, n_tok);
    static const bool commit_state = [] {   // STRATA_GDN_COMMIT_SPLIT=0: the commit runs the window's form
        const char* v = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    if (commit_state && n_keep != nullptr && t_out_begin >= n_tok) {   // the commit: no outputs
        done(stream, gdn_state_commit(q, state, hbuf, C, gate, beta, h_k, h_v, n_keep), "gdn_step_norm_multi");
        return;
    }
    done(stream, gdn_step_form(form, q, state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin,
                               static_cast<uint8_t*>(y_q8_1)),
         "gdn_step_norm_multi");
}

// ---- batch windows: every slot group in one launch (verify_kernels.hpp, STRATA_BATCH_GDN_SLOTS)
void gdn_conv_l2_slots(const GdnSlotTable& tab, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, void* stream, bool commit, const GdnAbArgs* ab) {
    if (!qkv || !conv_w || !h || channels <= 0 || channels % S != 0 || tab.n < 1 || tab.n > kVerifyMaxT) fail("gdn_conv_l2_slots: invalid arguments");
    int rows = 0;
    int row_slot[kVerifyMaxT] = {};
    for (int j = 0; j < tab.n; ++j) {
        if (tab.hist[j] == nullptr || tab.ntok[j] < 1 || tab.ntok[j] > kVerifyMaxT - rows ||
            tab.first[j] != rows || (commit && tab.ntok[j] != 1))
            fail("gdn_conv_l2_slots: the slot groups must tile the rows (one row each with commit)");
        for (int u = 0; u < tab.ntok[j]; ++u) row_slot[rows + u] = j;
        rows += tab.ntok[j];
    }
    const GdnSlotTable t_ = tab;   // by value into the kernel (baked into a captured graph: the slots' addresses are fixed)
    std::array<int, kVerifyMaxT> rs{};
    for (int i = 0; i < kVerifyMaxT; ++i) rs[(size_t) i] = row_slot[i];
    const int C = channels;
    // STRATA_GDN_AB_FORM 4..6: alpha/beta's units (gdn_ab_unit_at) as `extra` more rows of work-groups (C / S of them a
    // row, S / WARP sub-groups each, a unit a sub-group); the conv's work-groups are unchanged
    if (ab != nullptr && (ab->n_tok < 1 || ab->n_tok > kVerifyMaxT || ab->n_embd % 8 != 0))
        fail("gdn_conv_l2_slots: invalid alpha/beta arguments");
    const int tc = ab != nullptr ? ab_tc(gdn_ab_form() >= 4 ? gdn_ab_form() : 5) : 0;
    const int per_row = C / S * (S / WARP);
    const int extra = ab != nullptr ? (2 * ab->h_v * ((ab->n_tok + tc - 1) / tc) + per_row - 1) / per_row : 0;
    const GdnAbArgs ab_ = ab != nullptr ? *ab : GdnAbArgs{};
    const auto submit = [&](auto tcc) {
      constexpr int TC = decltype(tcc)::value;
      return Q(stream).submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(S / WARP), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) (rows + extra), (size_t) C}, {1, S}), xe::fits("gdn_conv_l2_slots", [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int row = (int) it.get_group(0);
            if constexpr (TC > 0) {
                if (row >= rows) {   // uniform over the work-group
                    const int u = ((row - rows) * (C / S) + (int) it.get_group(1)) * (S / WARP) + (int) sg.get_group_linear_id();
                    gdn_ab_unit_at<TC>(sg, (int) sg.get_local_linear_id(), u, ab_);
                    return;
                }
            }
            const int j = rs[(size_t) row];
            const float* history = t_.hist[j];
            const float* x0 = qkv + (size_t) t_.first[j] * C;   // the group's own rows
            const int t = row - t_.first[j];                     // the row within its group: gdn_conv_l2_multi's t
            const int tid = (int) it.get_local_id(1);
            const int c = (int) it.get_group(1) * S + tid;
            float win[3];
            for (int k = 0; k < 3; ++k) {
                const int src = t + k;
                win[k] = src < 3 ? history[c * 3 + src] : x0[(size_t) (src - 3) * C + c];
            }
            const float v0 = win[0], v1 = win[1], v2 = win[2], x = x0[(size_t) t * C + c];
            const float sum = v0 * conv_w[c * 4] + v1 * conv_w[c * 4 + 1] + v2 * conv_w[c * 4 + 2] + x * conv_w[c * 4 + 3];
            float y = sum / (1.0f + sycl::exp(-sum));
            if ((int) it.get_group(1) < qk_heads) {   // uniform over the work-group
                const float sq = warp_sum(sg, y * y);
                if ((tid & 31) == 0) part[tid >> 5] = sq;
                sycl::group_barrier(it.get_group());
                const float ss = part[0] + part[1] + part[2] + part[3];
                y *= sycl::rsqrt(ss + eps);
            }
            h[(size_t) row * C + c] = y;
            if (commit && (t_.keep == nullptr || t_.keep[t_.first[j]] != 0)) {   // one row a group, kept: [h1, h2, x]
                float* ho = t_.hist[j] + (size_t) c * 3;
                ho[0] = v1;
                ho[1] = v2;
                ho[2] = x;
            }
        }));
      });
    };
    const auto e = tc == 0   ? submit(std::integral_constant<int, 0>{})
                   : tc == 1 ? submit(std::integral_constant<int, 1>{})
                   : tc == 2 ? submit(std::integral_constant<int, 2>{})
                             : submit(std::integral_constant<int, 4>{});
    done(stream, e, "gdn_conv_l2_slots");
}

namespace {
// STRATA_BATCH_GDN_FORM picks how gdn_step_norm_slots runs; every form gives the plain one's bits (gdn_slots_parity):
//   0 plain: one work-group a (slot, head), gdn_step_plain's body (the default)
//   1 split: gdn_step_split's layout, each head's 128 state columns over 4 work-groups of 32 (4x the work-groups, a
//     quarter of each one's state), then the output norm over every row in a second launch (STRATA_BATCH_GDN_SPLIT=1
//     is this form when STRATA_BATCH_GDN_FORM is unset)
//   2 large: the plain body built with the large register file (grf_size<256>)
//   3 block: the plain body with the state read and written by sub-group block loads and stores
//   4 block+large: both; the fastest on the B70 (gdn_slots_parity, 16 slots: 210 us against plain's 569, 479 GB/s
//     of state against the card's 548 GB/s copy)
//   5 block16: block in 16-lane sub-groups (the state in 32 registers a work-item, not 64), within 128 registers
// Why 2-5: IGC built the plain body with 128 registers and spilled 10 KiB a work-item on the B70 (its ISA dump:
// "spill size 10368"), most of it the 32 rows' per-lane 64-bit addresses, kept from the loads to the stores; the spill
// traffic, not the state's, bounded the kernel.  A block load or store takes one scalar address a sub-group, and the
// large register file holds what is left: 4 builds without a spill (2 spills 2.4 KiB, 3 5.3 KiB, 5 1 KiB).
// Tried and dropped (bitwise too, all slower than 4 at 16 slots): one work-item a column with all 128 rows (one
// barrier a token, but it spilled 10 KiB even in 256 registers: 377 us), and a persistent grid of resident
// work-groups loading each next item's state while running the current one (the second state copy spilled: 333 us
// in SIMD32, 254 us in SIMD16).
enum GdnSlotsForm : int { kSlotsPlain = 0, kSlotsSplit, kSlotsLarge, kSlotsBlock, kSlotsBlockLarge, kSlotsBlock16, kSlotsForms };
constexpr int kSlotsFormDefault = kSlotsPlain;
int g_gdn_slots_form = -1;   // -1: not yet read from STRATA_BATCH_GDN_FORM / STRATA_BATCH_GDN_SPLIT
int gdn_slots_form() {
    if (g_gdn_slots_form < 0) {
        const char* f = std::getenv("STRATA_BATCH_GDN_FORM");
        const char* v = std::getenv("STRATA_BATCH_GDN_SPLIT");
        const long n = f != nullptr ? std::strtol(f, nullptr, 10)
                       : v != nullptr && std::strtol(v, nullptr, 10) != 0 ? (long) kSlotsSplit : (long) kSlotsFormDefault;
        g_gdn_slots_form = n >= 0 && n < kSlotsForms ? (int) n : kSlotsFormDefault;
    }
    return g_gdn_slots_form;
}

// gdn_step_norm_slots's arguments as the kernels take them
struct GdnSlotsArgs {
    GdnSlotTable t_;
    const float *hbuf, *gate, *beta, *z, *gamma;
    float* y;
    uint8_t* y_q8_1;
    float eps;
    int C, h_k, h_v, row0;
    bool commit, silu;
};

// block16 rebuilds the plain body's 32-lane sums bit for bit from shared memory, in 16-lane sub-groups.
constexpr int LANES_SG = 16;
// q8_1_store for a 32-value block over two 16-lane sub-groups: xi is lane l's value, xo its partner's (column ^ 16);
// the plain one's butterfly, its first step the pair's add.
inline void q8_1_store_pair(const sycl::sub_group& sg, const float xi, const float xo, uint8_t* block, const int lane32) {
#pragma clang fp contract(off)
    float amax = sycl::fmax(sycl::fabs(xi), sycl::fabs(xo)), sum = xi + xo;
    for (int o = LANES_SG / 2; o > 0; o >>= 1) {
        amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
        sum += sycl::permute_group_by_xor(sg, sum, o);
    }
    const float d = xe::q8_1_finite(amax / 127.0f);
    block[4 + lane32] = (uint8_t) xe::q8_1_quant(xi, d, amax);
    if (lane32 == 0) *reinterpret_cast<sycl::half2*>(block) = sycl::half2(sycl::half(d), sycl::half(xe::q8_1_finite(sum)));
}
// The plain body's warp_sum of 32 squares (a 32-column group's), from shared memory: lane l's first step is
// sq[l] + sq[l + 16], the rest the 16-lane butterfly; every lane ends with the plain body's lane-0 value.
inline float sum32_pair(const sycl::sub_group& sg, const float* sq, const int lane) {
#pragma clang fp contract(off)
    float v = sq[lane] + sq[lane + LANES_SG];
    for (int o = LANES_SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float square(const float v) {
#pragma clang fp contract(off)
    return v * v;
}

// gdn_step_plain's body with one more grid dimension: work-group (j, head) is slot group j's head.  Block: the state's
// rows by sub-group block loads and stores (a sub-group's columns are one run of a row); Large: the large register
// file; SGS: the sub-group size.  At 16 (block16) the state takes half the registers (32 rows a work-item are 32
// registers, not 64) and the output norm and q8_1 sums, the plain body's 32-lane butterflies, are rebuilt from shared
// memory (sum32_pair, q8_1_store_pair).  The arithmetic is the same in every instance.
template <bool Block, bool Large, int SGS = WARP> struct GdnSlotsPlain {
    GdnSlotsArgs a;
    sycl::local_accessor<float, 1> sk{}, sq{}, red{}, wsum{};   // wsum: SGS 16 keeps each column's oc and oc^2 there (2 S)

    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        namespace sx = sycl::ext::oneapi::experimental;
        if constexpr (Large)
            return sx::properties{sycl::ext::intel::experimental::grf_size<256>, sx::sub_group_size<SGS>,
                                  sx::work_group_size<S * RG>};
        else
            return sx::properties{sx::sub_group_size<SGS>, sx::work_group_size<S * RG>};
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int h_v = a.h_v, h_k = a.h_k;
        const int gidx = (int) it.get_group(0), j = gidx / h_v, head = gidx % h_v;
        const int tid = (int) it.get_local_id(0), col = tid % S, rg = tid / S, lane = tid % SGS;
        const int qh = head % h_k, qk = S * h_k, value_dim = S * h_v;
        const int f = a.t_.first[j], n = a.t_.ntok[j];
        const float* hb = a.hbuf + (size_t) f * a.C;
        const float* gt = a.gate + (size_t) f * h_v;
        const float* bt = a.beta + (size_t) f * h_v;
        const float* zt = a.z + (size_t) f * value_dim;
        float* yt = a.y + (size_t) f * value_dim;
        const bool q8 = a.y_q8_1 != nullptr;
        uint8_t* yq = q8 ? a.y_q8_1 + (size_t) (f - a.row0) * (size_t) (value_dim / 32) * 36 : nullptr;
        float s[RPG];
        float* base = a.t_.state[j] + ((size_t) (rg * RPG) * h_v + head) * S + (col - lane);   // the sub-group's run
        const size_t row_stride = (size_t) h_v * S;
        #pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = state_load<Block>(sg, base + r * row_stride, lane);
        for (int t = 0; t < n; ++t) {
            const float* ht = hb + (size_t) t * a.C;
            sycl::group_barrier(it.get_group());
            if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
            sycl::group_barrier(it.get_group());
            const float g = sycl::exp(gt[(size_t) t * h_v + head]);
            float kv = 0.0f;
            #pragma unroll
            for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
            red[rg * S + col] = kv;
            sycl::group_barrier(it.get_group());
            const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
            const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * bt[(size_t) t * h_v + head];
            float o = 0.0f;
            #pragma unroll
            for (int r = 0; r < RPG; ++r) {
                s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                o = sycl::fma(s[r], sq[rg * RPG + r], o);
            }
            sycl::group_barrier(it.get_group());
            red[rg * S + col] = o;
            sycl::group_barrier(it.get_group());
            float oc = 0.0f, sq_part = 0.0f;
            if constexpr (SGS == WARP) {
                if (rg == 0) {
                    oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * sycl::rsqrt((float) S);
                    sq_part = oc * oc;
                }
                sq_part = warp_sum(sg, sq_part);
                if (lane == 0) wsum[tid >> 5] = sq_part;
            } else if (rg == 0) {
                oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * sycl::rsqrt((float) S);
                wsum[col] = oc;
                wsum[S + col] = square(oc);
            }
            sycl::group_barrier(it.get_group());
            if (rg == 0) {
                float ss;
                if constexpr (SGS == WARP) {
                    ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                } else {
                    float w[RG];
                    #pragma unroll
                    for (int q = 0; q < RG; ++q) w[q] = sum32_pair(sg, &wsum[S + q * 32], lane);
                    ss = w[0] + w[1] + w[2] + w[3];
                }
                const float scale = sycl::rsqrt(ss / (float) S + a.eps);
                const auto out_of = [&](const int cc, const float occ) {
                    const float zz = zt[(size_t) t * value_dim + head * S + cc];
                    const float gz = 1.0f / (1.0f + sycl::exp(-zz));
                    return occ * scale * a.gamma[cc] * (a.silu ? zz * gz : gz);
                };
                const float out = out_of(col, oc);
                yt[(size_t) t * value_dim + (size_t) (head * S + col)] = out;
                if (q8) {
                    uint8_t* blk = yq + ((size_t) t * value_dim + (size_t) (head * S + col)) / 32 * 36;
                    if constexpr (SGS == WARP) q8_1_store(sg, out, blk, lane);
                    else q8_1_store_pair(sg, out, out_of(col ^ LANES_SG, wsum[col ^ LANES_SG]), blk, col & 31);
                }
            }
        }
        if (a.commit && (a.t_.keep == nullptr || a.t_.keep[f] != 0))   // uniform over the work-group (one slot group)
            #pragma unroll
            for (int r = 0; r < RPG; ++r) state_store<Block>(sg, s[r], base + r * row_stride, lane);
    }
};


sycl::event gdn_slots_split(sycl::queue& q, const GdnSlotTable& t_, const float* hbuf, int C, const float* gate,
                            const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                            bool commit, int row0, uint8_t* y_q8_1, int lo, int rows) {
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), hd), sq(sycl::range<1>(S), hd),
            red(sycl::range<1>((size_t) RG * GS_COLS), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) t_.n * h_v * (S / GS_COLS) * GS_COLS * RG, (size_t) GS_COLS * RG),
                        xe::fits("gdn_slots_split", [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const int per = h_v * (S / GS_COLS), gidx = (int) it.get_group(0), j = gidx / per, grp = gidx % per;
            const int head = grp / (S / GS_COLS), c0 = (grp % (S / GS_COLS)) * GS_COLS;
            const int tid = (int) it.get_local_id(0), lc = tid % GS_COLS, rg = tid / GS_COLS, col = c0 + lc;
            const int qh = head % h_k, qk = S * h_k, value_dim = S * h_v;
            const int f = t_.first[j], n = t_.ntok[j];
            float s[RPG];
            float* base = t_.state[j] + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
            #pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            for (int t = 0; t < n; ++t) {
                const float* ht = hbuf + (size_t) (f + t) * C;
                sycl::group_barrier(it.get_group());
                if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
                sycl::group_barrier(it.get_group());
                const float g = sycl::exp(gate[(size_t) (f + t) * h_v + head]);
                float kv = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                red[rg * GS_COLS + lc] = kv;
                sycl::group_barrier(it.get_group());
                const float kv_col = red[lc] + red[GS_COLS + lc] + red[2 * GS_COLS + lc] + red[3 * GS_COLS + lc];
                const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) (f + t) * h_v + head];
                float o = 0.0f;
                #pragma unroll
                for (int r = 0; r < RPG; ++r) {
                    s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                    o = sycl::fma(s[r], sq[rg * RPG + r], o);
                }
                sycl::group_barrier(it.get_group());
                red[rg * GS_COLS + lc] = o;
                sycl::group_barrier(it.get_group());
                if (rg == 0)
                    y[(size_t) (f + t) * value_dim + (size_t) (head * S + col)] =
                        (red[lc] + red[GS_COLS + lc] + red[2 * GS_COLS + lc] + red[3 * GS_COLS + lc]) * sycl::rsqrt((float) S);
            }
            if (commit && (t_.keep == nullptr || t_.keep[f] != 0))
                #pragma unroll
                for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
        }));
    });
    // the norm over rows [lo, lo + rows): y = oc * rsqrt(mean(oc^2) + eps) * gamma * gate(z), the plain kernel's sums
    return q.submit([&](sycl::handler& hd) {
        const bool silu = gdn_silu_gate();
        sycl::local_accessor<float, 1> wsum(sycl::range<1>(S / WARP), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t) h_v * rows * S, S), xe::fits("gdn_slots_split norm", [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int grp = (int) it.get_group(0), head = grp % h_v, row = lo + grp / h_v;
            const int col = (int) it.get_local_id(0), value_dim = S * h_v;
            const size_t at = (size_t) row * value_dim + (size_t) (head * S + col);
            const float oc = y[at];
            const float sq_part = warp_sum(sg, oc * oc);
            if ((col & 31) == 0) wsum[col >> 5] = sq_part;
            sycl::group_barrier(it.get_group());
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float) S + eps);
            const float zz = z[at];
            const float gz = 1.0f / (1.0f + sycl::exp(-zz));
            const float out = oc * scale * gamma[col] * (silu ? zz * gz : gz);
            y[at] = out;
            if (y_q8_1 != nullptr)
                q8_1_store(sg, out, y_q8_1 + ((size_t) (row - row0) * value_dim + (size_t) (head * S + col)) / 32 * 36,
                           col & 31);
        }));
    });
}
}  // namespace

void gdn_slots_set_form(int form) {
    if (form < 0 || form >= kSlotsForms) fail("gdn_slots_set_form: no such form");
    g_gdn_slots_form = form;
}
int gdn_slots_forms() { return kSlotsForms; }

void gdn_step_norm_slots(const GdnSlotTable& tab, const float* hbuf, int C, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, bool commit,
                         void* stream, int row0, void* y_q8_1_v) {
    if (!hbuf || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k || row0 < 0 || tab.n < 1 || tab.n > kVerifyMaxT)
        fail("gdn_step_norm_slots: invalid arguments");
    for (int j = 0; j < tab.n; ++j)
        if (tab.state[j] == nullptr || tab.ntok[j] < 1 || tab.first[j] < row0 || tab.first[j] > kVerifyMaxT ||
            tab.ntok[j] > kVerifyMaxT - tab.first[j] || (commit && tab.ntok[j] != 1))
            fail("gdn_step_norm_slots: a slot group is out of range");
    if (y_q8_1_v != nullptr && (S * h_v) % 32 != 0) fail("gdn_step_norm_slots: q8_1 blocks need 32 | value_dim");
    const GdnSlotTable t_ = tab;
    auto* y_q8_1 = static_cast<uint8_t*>(y_q8_1_v);
    auto& q = Q(stream);
    bool states_aligned = true;
    for (int j = 0; j < t_.n; ++j) states_aligned = states_aligned && aligned16(t_.state[j]);
    // the block forms store the state by 16-byte-aligned blocks: a state off that keeps the large register file alone
    const int chosen = gdn_slots_form();
    const int form = states_aligned || (chosen != kSlotsBlock && chosen != kSlotsBlockLarge && chosen != kSlotsBlock16)
                         ? chosen : kSlotsLarge;
    if (form == kSlotsSplit) {   // when the slot groups tile one run of rows (the batch window's)
        int lo = kVerifyMaxT, hi = 0, rows = 0;
        for (int j = 0; j < t_.n; ++j) {
            lo = std::min(lo, t_.first[j]);
            hi = std::max(hi, t_.first[j] + t_.ntok[j]);
            rows += t_.ntok[j];
        }
        if (rows == hi - lo) {
            done(stream, gdn_slots_split(q, t_, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, commit, row0, y_q8_1, lo, rows),
                 "gdn_step_norm_slots");
            return;
        }
    }
    const auto launch = [&](auto kernel) {
        return q.submit([&](sycl::handler& hd) {
            kernel.sk = sycl::local_accessor<float, 1>(sycl::range<1>(S), hd);
            kernel.sq = sycl::local_accessor<float, 1>(sycl::range<1>(S), hd);
            kernel.red = sycl::local_accessor<float, 1>(sycl::range<1>(RG * S), hd);
            kernel.wsum = sycl::local_accessor<float, 1>(sycl::range<1>(std::max(S * RG / WARP, 2 * S)), hd);
            hd.parallel_for(sycl::nd_range<1>((size_t) t_.n * h_v * S * RG, (size_t) S * RG), xe::fits("gdn_step_norm_slots", kernel));
        });
    };
    const GdnSlotsArgs a{t_, hbuf, gate, beta, z, gamma, y, y_q8_1, eps, C, h_k, h_v, row0, commit, gdn_silu_gate()};
    const auto e = form == kSlotsLarge ? launch(GdnSlotsPlain<false, true>{a})
                   : form == kSlotsBlock ? launch(GdnSlotsPlain<true, false>{a})
                   : form == kSlotsBlockLarge ? launch(GdnSlotsPlain<true, true>{a})
                   : form == kSlotsBlock16 ? launch(GdnSlotsPlain<true, false, LANES_SG>{a})
                                              : launch(GdnSlotsPlain<false, false>{a});
    done(stream, e, "gdn_step_norm_slots");
}

bool gdn_commit_layers(float* state, int64_t state_stride, int64_t conv_off, const float* qkv, int64_t qkv_stride,
                       const float* h, int64_t h_stride, const float* gate, const float* beta, int64_t gb_stride,
                       int conv_channels, int h_k, int h_v, int n_max, int n_layers, const int32_t* n_keep, void* stream) {
    // only the per-layer commit's state-only kernel is batched (gdn_step_norm_multi above): its other choices keep the loop
    static const bool per_layer = [] {
        const char* cs = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        const char* dbg = std::getenv("STRATA_DBG_GDN");
        return (cs != nullptr && std::strtol(cs, nullptr, 10) == 0) || (dbg != nullptr && dbg[0] == '1');
    }();
    if (per_layer || n_layers <= 0 || n_max < 1 || n_max > kVerifyMaxT || n_keep == nullptr || h_v <= 0 || h_k <= 0 ||
        h_v % h_k != 0)
        return false;
    gdn_conv_commit_layers(state + conv_off, state_stride, qkv, qkv_stride, conv_channels, n_layers, n_keep, stream);
    done(stream, gdn_state_commit(Q(stream), state, h, conv_channels, gate, beta, h_k, h_v, n_keep, n_layers, state_stride,
                                  h_stride, gb_stride),
         "gdn_commit_layers");
    return true;
}

bool graph_branch_pays(void* stream, void* side0, void* side1) {
    namespace sx = sycl::ext::oneapi::experimental;
    auto& q = Q(stream);
    sycl::queue* side[2] = {&Q(side0), &Q(side1)};
    float* buf = sycl::malloc_device<float>((size_t) 3 * 16 * 256, q);
    if (buf == nullptr) return false;
    // a kernel of a small projection's size (16 work-groups, a few microseconds)
    auto small = [](sycl::queue& on, float* p) {
        on.parallel_for(sycl::nd_range<1>((size_t) 16 * 256, 256), [=](sycl::nd_item<1> it) {
            float v = (float) it.get_global_id(0);
            for (int i = 0; i < 256; ++i) v = sycl::fma(v, 0.999f, 0.5f);
            p[it.get_global_id(0)] = v;
        });
    };
    std::vector<sx::command_graph<sx::graph_state::executable>> exec;
    for (int branched = 0; branched < 2; ++branched) {
        sx::command_graph<sx::graph_state::modifiable> graph(q.get_context(), q.get_device());
        graph.begin_recording(q);
        if (branched) {
            graph.begin_recording(*side[0]);
            graph.begin_recording(*side[1]);
        }
        for (int r = 0; r < 16; ++r) {
            // the rest of a layer's launches, in a row in both graphs (a graph with branches may run all its nodes
            // differently)
            for (int i = 0; i < 8; ++i) small(q, buf);
            if (branched) {
                for (int i = 0; i < 2; ++i) {
                    const sycl::event e = q.ext_oneapi_submit_barrier();
                    side[i]->ext_oneapi_submit_barrier({e});
                    small(*side[i], buf + (size_t) (1 + i) * 16 * 256);
                }
                small(q, buf);
                for (int i = 0; i < 2; ++i) {
                    const sycl::event e = side[i]->ext_oneapi_submit_barrier();
                    q.ext_oneapi_submit_barrier({e});
                }
            } else {
                for (int i = 0; i < 3; ++i) small(q, buf + (size_t) i * 16 * 256);
            }
        }
        if (branched) {
            graph.end_recording(*side[0]);
            graph.end_recording(*side[1]);
        }
        graph.end_recording(q);
        exec.push_back(graph.finalize());
    }
    double t_min[2] = {};
    for (int round = 0; round < 6; ++round)
        for (int v = 0; v < 2; ++v) {
            const auto t0 = std::chrono::steady_clock::now();
            q.ext_oneapi_graph(exec[v]);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            if (round > 0 && (t_min[v] == 0.0 || us < t_min[v])) t_min[v] = us;
        }
    sycl::free(buf, q);
    std::fprintf(stderr, "strata: graph side branches: %.1f us in a row, %.1f us branched: %s\n", t_min[0], t_min[1],
                 t_min[1] <= 0.95 * t_min[0] ? "used" : "not used");
    return t_min[1] <= 0.95 * t_min[0];
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    Q(stream).single_task([=] {
        while (read_host(flag) < value) {
        }
    });
}

bool doorbell_visible(void* stream) {
    auto& q = Q(stream);
    auto* m = sycl::malloc_host<uint32_t>(4, q);
    if (m == nullptr) return false;
    volatile uint32_t* hm = m;
    hm[0] = 0;
    hm[1] = 0;
    hm[2] = 0;
    uint32_t* ring = m;
    uint32_t* flag = m + 1;
    uint32_t* seen = m + 2;
    // at most 2^21 reads (well under a second on the UHD 770, a few microseconds where the flag arrives)
    q.single_task([=] {   // the ring as doorbell_ring stores it: volatile, not atomic (elementwise.cpp says why)
        *reinterpret_cast<volatile uint32_t*>(ring) = 1u;
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        for (uint32_t i = 0; i < (1u << 21); ++i)
            if (read_host(flag) == 1u) { *seen = 1u; break; }
    });
    const auto t0 = std::chrono::steady_clock::now();
    bool rang = false;
    while (!(rang = hm[0] == 1u) && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {}
    std::atomic_thread_fence(std::memory_order_seq_cst);
    hm[1] = 1u;
    q.wait();
    const bool ok = rang && hm[2] == 1u;
    sycl::free(m, q);
    return ok;
}

void resident_plan(const int32_t* ids, int n, int k, const int32_t* res, int n_expert, const uint8_t* cache_base,
                   const unsigned long long* slot_off, long long blob, int32_t* pl, long long capx, uint32_t* skip,
                   uint32_t ring, void* stream) {
    // One work-item per entry (at most kVerifyMaxT * 10 of them; one work-item alone grouped them in ~70 us a layer,
    // upstream 882764d).  The plan is the host loop's exactly: groups in order of first occurrence, entries in a group
    // in index order.  Entry i: L = its expert's first index; its group = the first occurrences before L; the group
    // starts at the number of entries whose leader is before L; its place in it = the earlier entries with leader L.
    constexpr int WG = 256;
    constexpr int CAP = kVerifyMaxT * 10;
    if (n < 1 || n > CAP || k < 1 || n > capx)
        throw std::invalid_argument("resident_plan: entries outside the window capacity");
    const auto e = Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> s_ids(sycl::range<1>(CAP), h);
        sycl::local_accessor<int32_t, 1> s_excl(sycl::range<1>(CAP), h);
        sycl::local_accessor<int32_t, 1> s_leader(sycl::range<1>(CAP), h);
        sycl::local_accessor<int32_t, 1> s_pos(sycl::range<1>(CAP), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int t = (int) it.get_local_id(0);
            bool bad = false;
            for (int i = t; i < n; i += WG) {
                const int32_t ex = ids[i];
                s_ids[i] = ex;
                bad = bad || ex < 0 || ex >= n_expert || res[ex] < 0;
            }
            if (sycl::any_of_group(grp, bad)) {
                if (t == 0) *skip = 0;
                return;
            }
            // Group predicates converge lanes but are not a local-memory publication contract.
            // Every chunk's ids must be visible before any lane searches another lane's entries.
            sycl::group_barrier(grp);
            // an entry's leader L (its expert's first index) and, for a leader, its entry count; then one scan of
            // (entries << 16 | 1) over the leaders numbers the groups and gives their starts (upstream fe4de5de)
            int total = 0;
            // Scan successive WG-entry chunks, retaining the packed prefix. This
            // keeps the same first-occurrence order without requiring a 512-item
            // work-group on devices that only support the established 256.
            for (int first = 0; first < n; first += WG) {
                const int i = first + t;
                int L = i, cnt = 0, pos = 0;
                if (i < n) {
                    for (int j = 0; j < i; ++j)
                        if (s_ids[j] == s_ids[i]) { L = j; break; }
                    for (int j = 0; j < n; ++j) {
                        if (s_ids[j] != s_ids[i]) continue;
                        if (L == i) ++cnt;
                        if (j < i) ++pos;
                    }
                    s_leader[i] = L;
                    s_pos[i] = pos;
                }
                const int pack = L == i && i < n ? (cnt << 16) | 1 : 0;
                const int excl = sycl::exclusive_scan_over_group(grp, pack, sycl::plus<int>());
                if (i < n) s_excl[i] = total + excl;
                total += sycl::reduce_over_group(grp, pack, sycl::plus<int>());
            }
            sycl::group_barrier(grp);
            int32_t* counts = pl;
            int32_t* start = pl + 4;
            int32_t* dst = start + capx + 1;
            int32_t* tok = dst + capx;
            const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
            unsigned long long* ptr = reinterpret_cast<unsigned long long*>(pl + ptr_off);
            int32_t* start2 = pl + ptr_off + 4 * capx;
            for (int i = t; i < n; i += WG) {
                const int L = s_leader[i], pos = s_pos[i], le = s_excl[L];
                const int g = le & 0xffff, gstart = le >> 16;
                dst[gstart + pos] = i;
                tok[gstart + pos] = i / k;
                if (L == i) {
                    const int32_t slot = res[s_ids[i]];
                    ptr[g] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
                    start[g] = gstart;
                }
            }
            sycl::group_barrier(grp);
            if (t != 0) return;
            const int groups = total & 0xffff;
            start[groups] = n;
            start2[0] = n;
            counts[0] = groups;
            counts[1] = n;
            counts[2] = 0;
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
            *skip = ring;
        });
    });
    done(stream, e, "resident_plan");
}

void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    Q(stream).single_task([=] {
        if (read_host(skip) == value) return;
        while (read_host(flag) < value) {
        }
    });
}

void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    const int ni = (int) n;
    Q(stream).parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> it) {
        if (*reinterpret_cast<const volatile uint32_t*>(skip) == value) return;
        for (int i = (int) it.get_local_id(0); i < ni; i += 128) dst[i] = reinterpret_cast<const volatile int32_t*>(src)[i];
    });
}

void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    // 16-byte reads, and `skip` read once a group: read by every item it crossed PCIe for each, and the RTX 4070
    // took 824 us for 10 rows of 2560 against 37
    using f32x4 = float __attribute__((ext_vector_type(4)));
    const long long n4 = n / 4;   // the CUDA kernel moved whole float4s: n is a multiple of 4
    const size_t groups = (size_t) std::min<long long>((n4 + 255) / 256, 64);
    Q(stream).parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) {
        uint32_t flag = 0;
        if (it.get_local_id(0) == 0) flag = *reinterpret_cast<const volatile uint32_t*>(skip);
        const bool zero = sycl::group_broadcast(it.get_group(), flag) == value;
        const long long step = (long long) it.get_global_range(0);
        for (long long i = (long long) it.get_global_id(0); i < n4; i += step)
            reinterpret_cast<f32x4*>(dst)[i] = zero ? f32x4(0.0f) : reinterpret_cast<const volatile f32x4*>(src)[i];
    });
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        const uint64_t token = (uint64_t) tokens[t];
        const uint8_t* c = codes + token * row_codes;
        const float* sc = scales + token * row_groups;
        const float* of = offsets ? offsets + token * row_groups : nullptr;
        const int per_byte = 8 / code_bits;
        const unsigned mask = (1u << code_bits) - 1u;
        const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
        const int64_t group = i / group_elems;
        const float product = (float) (code + code_bias) * sc[group];
        out[(size_t) t * n + i] = product + (of ? of[group] : 0.0f);
    });
    done(stream, e, "embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) (n_embd * hc)), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        R[(size_t) t * n_embd * hc + i] = x[(size_t) t * n_embd + i % n_embd];
    });
    done(stream, e, "broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int idx = *index;
        if (idx < 0) return;
        dst[id] = src[(size_t) idx * stride + id];
    });
    done(stream, e, "copy_indexed");
}

namespace {
// fetch_blobs' work-groups (of 256): the loads wait on the link, not the GPU, so a few work-groups keep enough of them
// in flight (the RTX 4070's PCIe 4.0 x4 is full at 6 of them, the B70's Gen5 x16 at 20), and the rest of the GPU is
// left to the VRAM hits that run beside this kernel (verify.cpp's fetch branch): with upstream's fixed 384 the hits
// took half again as long beside it (RTX 4070, IQ3_XXS 32K: the window 58.6 ms either way, 56.4 with 6).
// fetch_blobs_tune measures the device once; until then two a compute unit.  STRATA_FETCH_GROUPS=N fixes the count.
core::PerDevice<int>& fetch_groups() {
    static core::PerDevice<int> groups;
    return groups;
}
int fetch_groups_fixed() {
    static const int fixed = [] {
        const char* v = std::getenv("STRATA_FETCH_GROUPS");
        const long f = v == nullptr ? 0 : std::strtol(v, nullptr, 10);
        return f > 0 ? (int) f : 0;
    }();
    return fixed;
}
int fetch_groups_default(sycl::queue& q) {
    return std::max(16, 2 * (int) q.get_device().get_info<sycl::info::device::max_compute_units>());
}
sycl::event fetch_blobs_on(sycl::queue& q, const unsigned long long* src, const int32_t* n, uint8_t* dst, long long per,
                           size_t groups) {
    // a 16-byte clang vector: through sycl::uint4, NVPTX moved each 16 bytes as two 8-byte loads, and these loads
    // cross PCIe (RTX 4070: 858 us a call against upstream's 801)
    using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
    auto* d = reinterpret_cast<u32x4*>(dst);
    const size_t items = groups * 256;
    return q.parallel_for(sycl::nd_range<1>(items, 256), [=](sycl::nd_item<1> it) {
        const long long total = (long long) *n * per;
        for (long long i = (long long) it.get_global_id(0); i < total; i += (long long) items) {
            const long long k = i / per, off = i - k * per;
            d[i] = reinterpret_cast<const u32x4*>(src[k])[off];
        }
    });
}
}  // namespace

void fetch_blobs_tune(void* stream) {
    auto& q = Q(stream);
    fetch_groups().get(q.get_device(), [&] {
        if (const int fixed = fetch_groups_fixed(); fixed > 0) return fixed;
        // two blobs of 2 MiB from host memory, as a layer's PCIe share; each count 8 launches in a row, the best of
        // 4 rounds after a warm-up; the smallest count within 2% of the fastest is kept
        constexpr long long blob = 2ll << 20, per = blob / 16;
        constexpr int nb = 2;
        const int most = fetch_groups_default(q);
        auto* host = static_cast<uint8_t*>(sycl::malloc_host((size_t) nb * blob, q.get_context()));
        auto* dst = sycl::malloc_device<uint8_t>((size_t) nb * blob, q);
        auto* src = sycl::malloc_device<unsigned long long>(nb, q);
        auto* n = sycl::malloc_device<int32_t>(1, q);
        if (host == nullptr || dst == nullptr || src == nullptr || n == nullptr) {
            sycl::free(host, q); sycl::free(dst, q); sycl::free(src, q); sycl::free(n, q);
            return most;
        }
        std::memset(host, 1, (size_t) nb * blob);
        unsigned long long tab[nb];
        for (int k = 0; k < nb; ++k) tab[k] = (unsigned long long) (host + (size_t) k * blob);
        q.memcpy(src, tab, sizeof tab);
        q.memcpy(n, &nb, sizeof nb).wait();
        std::vector<int> counts;
        for (int g = 4; g < most; g = g < 8 ? g + 2 : g + g / 2) counts.push_back(g);
        counts.push_back(most);
        std::vector<double> t_min(counts.size(), 0.0);
        for (int round = 0; round < 5; ++round)
            for (size_t c = 0; c < counts.size(); ++c) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 8; ++i) fetch_blobs_on(q, src, n, dst, per, (size_t) counts[c]);
                q.wait();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (round > 0 && (t_min[c] == 0.0 || us < t_min[c])) t_min[c] = us;
            }
        sycl::free(host, q); sycl::free(dst, q); sycl::free(src, q); sycl::free(n, q);
        const double best = *std::min_element(t_min.begin(), t_min.end());
        size_t pick = counts.size() - 1;
        for (size_t c = 0; c < counts.size(); ++c)
            if (t_min[c] <= best * 1.02) { pick = c; break; }
        std::fprintf(stderr, "strata: the PCIe share's fetch runs %d work-groups (%.1f GB/s; %d: %.1f GB/s)\n", counts[pick],
                     (double) nb * blob * 8 / t_min[pick] / 1e3, most, (double) nb * blob * 8 / t_min.back() / 1e3);
        return counts[pick];
    });
}

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) fail("fetch_blobs: blob size must be a multiple of 16");
    auto& q = Q(stream);
    const int fixed = fetch_groups_fixed();
    const int groups = fixed > 0 ? fixed : fetch_groups().get(q.get_device(), [&] { return fetch_groups_default(q); });
    done(stream, fetch_blobs_on(q, src, n, dst, (long long) (blob_bytes / 16), (size_t) groups), "fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    const unsigned long long b = (unsigned long long) base;
    const long long bytes = (long long) blob_bytes;
    const auto e = Q(stream).parallel_for(sycl::range<1>(128), [=](sycl::id<1> id) {
        const int k = (int) id[0];
        if (k < *n) ptr[k] = b + (unsigned long long) k * (unsigned long long) bytes;
    });
    done(stream, e, "rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* ex, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) (n_embd * hc)), [=](sycl::id<2> id) {
        const int t = (int) id[0];
        const int64_t i = (int64_t) id[1];
        R[(size_t) t * n_embd * hc + i] = h[(size_t) t * n_embd * hc + i] + ex[(size_t) t * n_embd + i % n_embd];
    });
    done(stream, e, "add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) fail("ident_hits: n out of range");
    const auto e = Q(stream).parallel_for(sycl::range<1>(1024), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i < n) { slot[i] = ids[i]; dst[i] = i; }
        if (i == 0) *count = n;
    });
    done(stream, e, "ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    const size_t items = 16 * 256;
    const auto e = Q(stream).parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
        const int row = *row_dev;
        for (int64_t i = (int64_t) id[0]; i < R_stride; i += (int64_t) items) R_dst[i] = R_src[(size_t) row * R_stride + i];
        if (id[0] == 0) {
            const int32_t tok = ids[row];
            *tok_dst = tok;
            // the token last: the drafter reads the probability once the token is there (MtpDrafter::draft)
            if (probs != nullptr && out_p != nullptr) reinterpret_cast<volatile float*>(out_p)[j] = probs[row];
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
            if (out != nullptr) reinterpret_cast<volatile int32_t*>(out)[j] = tok;
        }
    });
    done(stream, e, "mtp_select");
}

void force_token(int32_t* tok, const int32_t* force, int j, void* stream) {
    const auto e = Q(stream).single_task([=] {
        const int32_t f = reinterpret_cast<const volatile int32_t*>(force)[j];
        if (f >= 0) *tok = f;
    });
    done(stream, e, "force_token");
}

void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
                       void* stream) {
    const size_t items = (size_t) 16 * 256;
    const auto e = Q(stream).parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
        const int64_t row = *row_dev;
        if (row == 0) return;   // already there
        for (int64_t i = (int64_t) id[0]; i < a_n + b_n + c_n; i += (int64_t) items) {
            if (i < a_n) a[i] = a[row * a_n + i];
            else if (i < a_n + b_n) b[i - a_n] = b[row * b_n + (i - a_n)];
            else c[i - a_n - b_n] = c[row * c_n + (i - a_n - b_n)];
        }
    });
    done(stream, e, "copy_row_to_first");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    // the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
    auto launch = [&](auto elem) {
        using E = decltype(elem);
        const long long row_e = (long long) row_bytes / (long long) sizeof(E);
        const E* s = reinterpret_cast<const E*>(src);
        E* d = reinterpret_cast<E*>(dst);
        const long long total = (long long) n * row_e;
        return Q(stream).parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> id) {
            const long long i = (long long) id[0], r = i / row_e, o = i - r * row_e;
            d[i] = s[(long long) ids[r] * row_e + o];
        });
    };
    if (n <= 0) return;
    sycl::event e;
    if (row_bytes % 16 == 0) e = launch(sycl::uint4{});
    else if (row_bytes % 4 == 0) e = launch(uint32_t{});
    else e = launch(uint8_t{});
    done(stream, e, "gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<1>(64), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i < n) ids[i] = table[ids[i]];
    });
    done(stream, e, "map_ids");
}

namespace {
template <int WG>
sycl::event row_top_prob_launch(sycl::queue& q, const float* logits, int n_rows, int n_vocab, const int32_t* ids,
                                float* probs) {
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(WG / 32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_rows * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_group(0), tid = (int) it.get_local_id(0);
            const float* l = logits + (size_t) t * n_vocab;
            const float m = l[ids[t]];
            float s = 0.0f;
            for (int i = tid; i < n_vocab; i += WG) s += sycl::exp(l[i] - m);
            s = warp_sum(sg, s);
            if ((tid & 31) == 0) part[tid >> 5] = s;
            sycl::group_barrier(it.get_group());
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < WG / 32; ++w) tot += part[w];
                probs[t] = 1.0f / tot;
            }
        });
    });
}
}  // namespace

namespace {
// row_top_prob over SPLIT work-groups a row (upstream 3e4aa1d6: one group a row left most of the GPU idle on the draft
// head's 40,525 tokens).  Group b holds the one-group kernel's work-items [b * WG / SPLIT, (b + 1) * WG / SPLIT) with
// their stride WG, so each sub-group's sum is the one-group kernel's; they go to the row's slots in a device scratch,
// and the row's last group (a counter it resets) adds the WG / 32 sums in the same order: the same bits.
constexpr int TOP_SPLIT = 8;
struct TopScratch { float* part; uint32_t* count; };
TopScratch top_scratch(sycl::queue& q) {
    static core::PerDevice<TopScratch> per_device;
    return per_device.get(q.get_device(), [&q] {
        const size_t rows = kVerifyMaxT;
        TopScratch t{};
        t.part = sycl::malloc_device<float>(rows * 32, q.get_device(), q.get_context());
        t.count = sycl::malloc_device<uint32_t>(rows, q.get_device(), q.get_context());
        if (!t.part || !t.count) throw core::DeviceError("row_top_prob: device allocation failed");
        sycl::queue init(q.get_context(), q.get_device());   // not the caller's queue: it may be recording a graph
        init.memset(t.count, 0, rows * sizeof(uint32_t)).wait();
        return t;
    });
}
template <int WG>
sycl::event row_top_prob_split(sycl::queue& q, const float* logits, int n_rows, int n_vocab, const int32_t* ids,
                               float* probs, TopScratch sc) {
    constexpr int LW = WG / TOP_SPLIT;
    static_assert(LW % 32 == 0, "whole sub-groups a group");
    return q.parallel_for(sycl::nd_range<1>((size_t) n_rows * TOP_SPLIT * LW, LW), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int grp = (int) it.get_group(0), t = grp / TOP_SPLIT, b = grp % TOP_SPLIT;
        const int tid = b * LW + (int) it.get_local_id(0);   // the one-group kernel's work-item
        const float* l = logits + (size_t) t * n_vocab;
        const float m = l[ids[t]];
        float s = 0.0f;
        for (int i = tid; i < n_vocab; i += WG) s += sycl::exp(l[i] - m);
        s = warp_sum(sg, s);
        if ((tid & 31) == 0) sc.part[t * 32 + (tid >> 5)] = s;
        // the last of the row's groups to finish adds the sums
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        sycl::group_barrier(it.get_group());
        if (it.get_local_id(0) != 0) return;
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> cnt(sc.count[t]);
        if (cnt.fetch_add(1u) != TOP_SPLIT - 1) return;
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
        float tot = 0.0f;
        for (int w = 0; w < WG / 32; ++w) tot += sc.part[t * 32 + w];
        probs[t] = 1.0f / tot;
        cnt.store(0u);
    });
}
}  // namespace

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    auto& q = Q(stream);
    const int wg = xe::work_group_upto_1024(q);   // the draft probabilities sum in another order below 1024
    static const bool split = [] {   // STRATA_TOP_PROB_SPLIT=0: one work-group a row
        const char* v = std::getenv("STRATA_TOP_PROB_SPLIT");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    // the split for the main model's head (248,320 tokens: RTX 4070 15.1 -> 10.2 us a row, B70 38.95 -> 38.79), not
    // for a draft head's subset (40,525: the RTX 4070 3.98 -> 3.52 us, the B70 7.60 -> 7.85)
    constexpr int kSplitMinVocab = 65536;
    if (split && n_rows <= kVerifyMaxT && n_vocab >= kSplitMinVocab) {
        const TopScratch sc = top_scratch(q);
        const auto e = wg == 1024 ? row_top_prob_split<1024>(q, logits, n_rows, n_vocab, ids, probs, sc)
                       : wg == 512 ? row_top_prob_split<512>(q, logits, n_rows, n_vocab, ids, probs, sc)
                                   : row_top_prob_split<256>(q, logits, n_rows, n_vocab, ids, probs, sc);
        done(stream, e, "row_top_prob");
        return;
    }
    const auto e = wg == 1024 ? row_top_prob_launch<1024>(q, logits, n_rows, n_vocab, ids, probs)
                   : wg == 512 ? row_top_prob_launch<512>(q, logits, n_rows, n_vocab, ids, probs)
                               : row_top_prob_launch<256>(q, logits, n_rows, n_vocab, ids, probs);
    done(stream, e, "row_top_prob");
}

void logits_nonfinite_rows(const float* logits, int n_rows, int n, uint32_t* flags, void* stream) {
    if (n_rows <= 0 || n <= 0) return;
    constexpr int WG = 256;
    const int per_block = (n + kNonfiniteBlocks - 1) / kNonfiniteBlocks;
    // work-group (b, row) scans [b * per_block, +per_block) of its row and writes its own word.  The test is on the
    // bits (exponent all ones), so a fast-math build cannot fold it away.
    const auto e = Q(stream).parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) n_rows, (size_t) kNonfiniteBlocks * WG), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) {
            const int row = (int) it.get_group(0), b = (int) it.get_group(1);
            const float* l = logits + (size_t) row * (size_t) n;
            const int lo = b * per_block, hi = sycl::min(n, lo + per_block);
            bool bad = false;
            for (int v = lo + (int) it.get_local_id(1); v < hi; v += WG)
                bad |= (sycl::bit_cast<uint32_t>(l[v]) & 0x7f800000u) == 0x7f800000u;
            bad = sycl::any_of_group(it.get_group(), bad);
            if (it.get_local_id(1) == 0) flags[row * kNonfiniteBlocks + b] = bad ? 1u : 0u;
        });
    done(stream, e, "logits_nonfinite_rows");
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    const long long stride = (long long) ids_stride;
    const auto e = Q(stream).parallel_for(sycl::nd_range<2>({(size_t) n, 8 * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
        const int q = (int) it.get_group(0);
        int32_t* st = steps + q * 4;
        const int n_kv = st[1];
        const int start = n_kv > window ? n_kv - window : 0;
        const int width = n_kv - start;
        for (int jj = (int) it.get_global_id(1); jj < width; jj += (int) it.get_global_range(1))
            ids[q * stride + jj] = start + jj;
        sycl::group_barrier(it.get_group());
        if (it.get_group(1) == 0 && it.get_local_id(1) == 0) st[3] = width;
    });
    done(stream, e, "window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    const auto e = Q(stream).parallel_for(sycl::range<1>(64), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i >= n) return;
        const int c = cells[i];
        steps[i * 4 + 0] = c;
        steps[i * 4 + 1] = c + 1;
        steps[i * 4 + 2] = (c + 1) / 4;
        steps[i * 4 + 3] = c + 1;
    });
    done(stream, e, "dense_steps");
}

// The stage stamps' clock: STRATA_VERIFY_PROFILE_CLOCK (device, sub_group or work_group; default device) names the
// sycl_ext_oneapi_clock scope, and a device without it falls back to the first of device, sub_group and work_group it
// has.  The B70 has only the sub-group clock, which tools/clock_probe.cpp measured (2026-10-23) as a timeline the stage
// stamps can use: one 64-bit counter across the EUs and the kernels, counting GPU core cycles (so its rate follows the
// GPU's frequency, 0.5 to 2.8 ticks/ns), restarting only after the GPU idles for some 10 ms - longer than any gap
// inside a window.  The profile converts ticks to time with each window's own rate (Verifier::accumulate_profile).
namespace {
enum class StampClock { none, device, sub_group, work_group };
constexpr const char* kStampClockDefault = "device";

StampClock stamp_clock() {
    static const StampClock c = [] {
        const sycl::device d = core::Runtime::get().compute().get_device();
        if (d.get_backend() == sycl::backend::ext_oneapi_cuda) return StampClock::device;   // %globaltimer
#ifdef SYCL_EXT_ONEAPI_CLOCK
        const struct { const char* name; StampClock c; sycl::aspect a; } scopes[] = {
            {"device", StampClock::device, sycl::aspect::ext_oneapi_clock_device},
            {"sub_group", StampClock::sub_group, sycl::aspect::ext_oneapi_clock_sub_group},
            {"work_group", StampClock::work_group, sycl::aspect::ext_oneapi_clock_work_group}};
        const char* v = std::getenv("STRATA_VERIFY_PROFILE_CLOCK");
        const std::string want = v != nullptr && v[0] != '\0' ? v : kStampClockDefault;
        for (const auto& s : scopes)
            if (want == s.name && d.has(s.a)) return s.c;
        for (const auto& s : scopes)
            if (d.has(s.a)) return s.c;
#endif
        return StampClock::none;
    }();
    return c;
}
}  // namespace

// The stage stamps: %globaltimer (nanoseconds) where the kernel is compiled for NVPTX, whose devices report no
// device-scope clock; elsewhere the stamp_clock() scope's clock, which counts device ticks.  Each scope is its own
// kernel: a kernel that names a clock needs its aspect, so one kernel switching between scopes would need them all.
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    const StampClock c = stamp_clock();
    if (c == StampClock::none) fail("gpu_stamp: the device has no clock a kernel reads");
#if defined(SYCL_EXT_ONEAPI_CLOCK)
    namespace sx = sycl::ext::oneapi::experimental;
    if (c == StampClock::sub_group) {
        Q(stream).single_task([=] { buf[i] = (unsigned long long) sx::clock<sx::clock_scope::sub_group>(); });
        return;
    }
    if (c == StampClock::work_group) {
        Q(stream).single_task([=] { buf[i] = (unsigned long long) sx::clock<sx::clock_scope::work_group>(); });
        return;
    }
#endif
    Q(stream).single_task([=] {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        buf[i] = t;
#elif defined(SYCL_EXT_ONEAPI_CLOCK)
        buf[i] = (unsigned long long) sycl::ext::oneapi::experimental::clock<
            sycl::ext::oneapi::experimental::clock_scope::device>();
#else
        buf[i] = 0;
#endif
    });
}

bool gpu_stamp_available() { return stamp_clock() != StampClock::none; }

const char* gpu_stamp_clock() {
    static const char* names[] = {"none", "device", "sub_group", "work_group"};
    return names[(int) stamp_clock()];
}

}  // namespace strata::kernels
