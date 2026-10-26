// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/batch_window_ubench.cpp - homelab: what each kernel of one layer of a batch window
// (Verifier::record_window's pre(l) and post(l), a batch of slots one row each) costs at 1..16 rows.
//
//     build/batch_window_ubench --spec SPEC [--ctx 16384] [--max-ctx 131072] [--resident-cells 4096] [--reps 25]
//                               [--warm 50] [--flush-mib 128] [--rows 1,2,4,8,16] [--mmq-min 4] [--seed 353]
//
// SPEC is the model's geometry and its layers' tensor types, read from the GGUF header (tools: one `dims` line of
// key=value, then one `layer L key=type ...` line per layer the stage runs; a layer with attn_q is QSA, else GDN).
// Every kernel runs on synthetic weights of the real shapes and types, through the engine's own entry points, with the
// standing batch configuration's paths (STRATA_BATCH_DENSE_MMQ_MIN, GDN_SLOTS, SELF_COMMIT, QSA_ROWS, the Q8_0
// hyper-connection read, the resident expert plan, int8 KV streamed).  Each measurement: the L2 flushed by a fill,
// then a profiling tag, the kernel's launches, a profiling tag; the median over --reps of the tags' gap (the empty gap
// is printed as the floor of the method and subtracted).  A distinct (kernel, type) pair is measured once; the stage
// roll-up multiplies them over the SPEC's layers.  No model file, no engine session: the live server keeps its VRAM.
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/hc_mmq.hpp"
#include "strata/kernels/hc_xmx.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/dmmq_dispatch.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

namespace {

struct Opt {
    std::string spec;
    int ctx = 16384, max_ctx = 131072, resident_cells = 4096, reps = 25, warm = 50, flush_mib = 128, mmq_min = 4;
    // warm 50 (homelab hc_mode_probe): a shape's first 25-50 calls in a process run up to 1.3x slow, so with 3 the
    // first rows measured (the hc read) came out bimodal from run to run
    int spin_ms = 200, warmup_ms = 3000;   // the card busy before each measurement: its clocks fall when it idles
    uint32_t seed = 353;
    std::vector<int> rows{1, 2, 4, 8, 16};
};

struct Layer {
    int index = 0;
    bool qsa = false;
    std::map<std::string, int> type;
};

struct Spec {
    std::map<std::string, int64_t> dim;
    std::vector<Layer> layers;
    int64_t at(const char* key) const {
        const auto it = dim.find(key);
        if (it == dim.end()) throw std::runtime_error(std::string("spec: no dims key ") + key);
        return it->second;
    }
};

Spec read_spec(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("spec: cannot open " + path);
    Spec s;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string head, kv;
        ls >> head;
        if (head == "dims") {
            while (ls >> kv) s.dim[kv.substr(0, kv.find('='))] = std::stoll(kv.substr(kv.find('=') + 1));
        } else if (head == "layer") {
            Layer l;
            ls >> l.index;
            while (ls >> kv) l.type[kv.substr(0, kv.find('='))] = std::stoi(kv.substr(kv.find('=') + 1));
            l.qsa = l.type.count("attn_q") != 0;
            s.layers.push_back(l);
        }
    }
    return s;
}

sycl::queue& Q() { return strata::core::Runtime::get().compute(); }
void* CS() { return &Q(); }

// a phase's buffers, freed when the phase ends (the live server leaves ~1 GiB of VRAM)
struct Arena {
    std::vector<void*> ptrs;
    size_t dev_bytes = 0;
    template <class T> T* dev(size_t n) {
        auto& rt = strata::core::Runtime::get();
        void* p = sycl::malloc_device(std::max<size_t>(n * sizeof(T), 64) + 64, rt.device(), rt.context());
        if (p == nullptr) throw std::runtime_error("ubench: device allocation of " + std::to_string(n * sizeof(T)) + " B failed");
        ptrs.push_back(p);
        dev_bytes += n * sizeof(T);
        return static_cast<T*>(p);
    }
    template <class T> T* host(size_t n) {
        auto& rt = strata::core::Runtime::get();
        void* p = sycl::malloc_host(std::max<size_t>(n * sizeof(T), 64) + 64, rt.context());
        if (p == nullptr) throw std::runtime_error("ubench: host allocation failed");
        std::memset(p, 0, n * sizeof(T));
        ptrs.push_back(p);
        return static_cast<T*>(p);
    }
    ~Arena() {
        Q().wait();
        for (void* p : ptrs) sycl::free(p, strata::core::Runtime::get().context());
    }
};

inline uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
// bytes in [0, 0x3c): every fp16 / bf16 field read from them is finite and below 1, every code and index in range
void fill_bytes(void* p, size_t n, uint32_t seed) {
    auto* b = static_cast<uint8_t*>(p);
    Q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = (uint8_t) (mix((uint32_t) i[0] * 2654435761u ^ seed) % 0x3c); });
}
void fill_f32(float* p, size_t n, uint32_t seed, float lo, float hi) {
    Q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        p[i] = lo + (hi - lo) * (float) (mix((uint32_t) i[0] * 2654435761u ^ seed) & 0xffffff) / 16777216.0f;
    });
}
void fill_bf16(uint16_t* p, size_t n, uint32_t seed, float scale) {
    Q().parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float v = scale * ((float) (mix((uint32_t) i[0] * 2654435761u ^ seed) & 0xffff) / 32768.0f - 1.0f);
        p[i] = (uint16_t) (sycl::bit_cast<uint32_t>(v) >> 16);
    });
}
template <class T> T* put(Arena& a, const std::vector<T>& h) {
    T* d = a.dev<T>(h.size());
    Q().memcpy(d, h.data(), h.size() * sizeof(T)).wait();
    return d;
}

// ---- timing: L2 flushed, then tag | launches | tag on the in-order compute queue; the median gap
struct Timer {
    float* flush = nullptr;
    size_t flush_n = 0;
    int reps = 25, warm = 3;
    double floor_us = 0;
    double once(const std::function<void(int)>& f, int rep) {
        auto& q = Q();
        q.fill(flush, (float) (rep & 7), flush_n);
        sycl::event a = sx::submit_profiling_tag(q);
        f(rep);
        sycl::event b = sx::submit_profiling_tag(q);
        b.wait_and_throw();
        const auto ta = a.get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto tb = b.get_profiling_info<sycl::info::event_profiling::command_end>();
        return (double) (int64_t) (tb - ta) / 1e3;
    }
    double median(const std::function<void(int)>& f) {
        for (int i = 0; i < warm; ++i) once(f, i);
        std::vector<double> v;
        for (int i = 0; i < reps; ++i) v.push_back(once(f, warm + i));
        std::sort(v.begin(), v.end());
        return std::max(0.0, v[v.size() / 2] - floor_us);
    }
};

Timer g_timer;

// the card kept busy (fills of the flush buffer) for `ms`: the measurements below then run at its loaded clocks
void spin(int ms) {
    if (ms <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    int i = 0;
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < ms) {
        Q().fill(g_timer.flush, (float) (i++ & 7), g_timer.flush_n);
        if ((i & 7) == 0) Q().wait();
    }
    Q().wait();
}

struct Result {
    std::map<int, double> us;   // rows -> median microseconds
    double bytes16 = 0;         // bytes a 16-row call must move: weights once + state + activations (the floor)
};
std::map<std::string, Result> g_res;
Opt g_opt;

void bench(const std::string& key, const std::function<double(int)>& bytes, const std::function<void(int)>& setup,
           const std::function<void(int, int)>& run) {
    if (g_res.count(key)) return;
    spin(g_opt.spin_ms);
    Result& r = g_res[key];
    std::printf("%-44s", key.c_str());
    for (int T : g_opt.rows) {
        setup(T);
        Q().wait_and_throw();
        r.us[T] = g_timer.median([&](int rep) { run(T, rep); });
        std::printf(" %9.1f", r.us[T]);
        std::fflush(stdout);
    }
    r.bytes16 = bytes(16);
    std::printf("   | %.2f MB @16\n", r.bytes16 / 1e6);
}

double wbytes(int type, int64_t n_in, int64_t n_out) {
    return (double) k::native_mmvq_weight_bytes(type == k::kNativeQ6KRows ? 14 : type, (int) n_in, (int) n_out);
}
size_t q81(int64_t cols, int rows) { return (size_t) rows * (size_t) (cols / 32) * 36; }

}  // namespace

int main(int argc, char** argv) {
    Opt& o = g_opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--spec") o.spec = next();
        else if (a == "--ctx") o.ctx = std::stoi(next());
        else if (a == "--max-ctx") o.max_ctx = std::stoi(next());
        else if (a == "--resident-cells") o.resident_cells = std::stoi(next());
        else if (a == "--reps") o.reps = std::stoi(next());
        else if (a == "--warm") o.warm = std::stoi(next());
        else if (a == "--flush-mib") o.flush_mib = std::stoi(next());
        else if (a == "--mmq-min") o.mmq_min = std::stoi(next());
        else if (a == "--spin-ms") o.spin_ms = std::stoi(next());
        else if (a == "--warmup-ms") o.warmup_ms = std::stoi(next());
        else if (a == "--seed") o.seed = (uint32_t) std::stoul(next());
        else if (a == "--rows") {
            o.rows.clear();
            std::stringstream ss(next());
            std::string t;
            while (std::getline(ss, t, ',')) o.rows.push_back(std::stoi(t));
        } else throw std::runtime_error("unknown argument " + a);
    }
    if (o.spec.empty()) { std::fprintf(stderr, "usage: batch_window_ubench --spec SPEC [options]\n"); return 2; }
    const Spec sp = read_spec(o.spec);
    const int MT = *std::max_element(o.rows.begin(), o.rows.end());
    if (MT > k::kVerifyMaxT || k::verify_max_t() < MT)
        throw std::runtime_error("rows past STRATA_BATCH_MAX_T: set STRATA_BATCH_MAX_T=16");

    // the engine's switches as `--native` sets them (generate.cpp) and the HC_Q8 read the live server chose
    k::shared_expert_set_native_bf16(true);
    k::native_moe_combine_set_enabled(true);
    k::native_gdn_set_enabled(true);
    k::native_router_set_enabled(true);
    k::native_qsa_set_enabled(true);
    k::native_qsa_indexer_set_enabled(true);
    k::native_rope_set_enabled(true);
    k::fused_gr_set_q8_fuse(false);

    const int64_t N = sp.at("n_embd"), FF = sp.at("n_ff"), NE = sp.at("n_expert"), K = sp.at("k"), HC = sp.at("hc"),
                  LR = sp.at("hc_lr"), NH = sp.at("n_head"), NKV = sp.at("n_head_kv"), HD = sp.at("head_dim"),
                  IQ = sp.at("idx_heads"), ID = sp.at("idx_dim"), DS = sp.at("ssm_d_state"), HK = sp.at("ssm_groups"),
                  HV = sp.at("ssm_heads"), ZV = sp.at("ssm_inner"), DC = sp.at("ssm_d_conv");
    const int64_t C = 2 * HK * DS + ZV, QKV = C;
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = NH; s.n_head_kv = NKV; s.head_dim = HD; s.idx_n_head = IQ; s.idx_dim = ID;
    const float eps = k::qsa_rms_eps();

    auto& rt = strata::core::Runtime::get();
    // homelab's small-kernel lane (each default off): the forms the engine would take, read from the environment
    std::printf("forms: STRATA_HC_Q8_WIN %d, STRATA_GDN_AB_FORM %d (in the conv launch: %d), STRATA_MMVF_WIDE %d, "
                "STRATA_ROUTER_FUSE %d\n", k::fused_gr_q8_win(), k::gdn_ab_form(), (int) k::gdn_ab_fused_in_conv(),
                k::bf16_gemv_wide(), (int) k::bf16_router_fuse());
    std::printf("device: %s, %u compute units, mmq on XMX: %d, rows %d..%d, ctx %d (max %d), %d resident cells, "
                "reps %d, flush %d MiB\n",
                rt.device().get_info<sycl::info::device::name>().c_str(),
                rt.device().get_info<sycl::info::device::max_compute_units>(), (int) k::iq_mmq_usable(Q()),
                o.rows.front(), MT, o.ctx, o.max_ctx, o.resident_cells, o.reps, o.flush_mib);

    Arena base;
    g_timer.reps = o.reps;
    g_timer.warm = o.warm;
    g_timer.flush_n = (size_t) o.flush_mib << 18;
    g_timer.flush = base.dev<float>(g_timer.flush_n);

    // ---- the card's bandwidth: a copy kernel over twice the flush size, read + write bytes (again at the end)
    double bw_gbs = 0;
    spin(o.warmup_ms);
    auto copy_test = [&](const char* when) {
        Arena a;
        const size_t n = g_timer.flush_n;   // floats each side
        float* x = a.dev<float>(n);
        float* y = a.dev<float>(n);
        fill_f32(x, n, 1, -1, 1);
        Q().wait();
        auto* x4 = reinterpret_cast<sycl::float4*>(x);
        auto* y4 = reinterpret_cast<sycl::float4*>(y);
        const double us_k = g_timer.median([&](int) {
            Q().parallel_for(sycl::range<1>(n / 4), [=](sycl::id<1> i) { y4[i] = x4[i]; });
        });
        const double us_m = g_timer.median([&](int) { Q().memcpy(y, x, n * 4); });
        const double us_r = g_timer.median([&](int) {
            Q().parallel_for(sycl::range<1>(n / 4), [=](sycl::id<1> i) {
                const sycl::float4 v = x4[i];
                if (v.x() == 12345.f) y4[i] = v;   // read-only stream
            });
        });
        bw_gbs = std::max(bw_gbs, 2.0 * n * 4 / us_k / 1e3);
        std::printf("%s: copy kernel %.1f MiB -> %.1f MiB: %.1f us = %.1f GB/s (read+write); memcpy %.1f GB/s; "
                    "read-only %.1f GB/s\n", when,
                    n * 4 / 1048576.0, n * 4 / 1048576.0, us_k, bw_gbs, 2.0 * n * 4 / us_m / 1e3,
                    1.0 * n * 4 / us_r / 1e3);
    };
    copy_test("start");
    g_timer.floor_us = 0;
    g_timer.floor_us = g_timer.median([](int) {});
    std::printf("method floor (flush, tag, tag): %.1f us, subtracted from every row below\n", g_timer.floor_us);

    // ---- per-row inputs shared by every phase (the window's 16 rows)
    float* R = base.dev<float>((size_t) MT * HC * N);
    float* bo = base.dev<float>((size_t) MT * N);
    float* inj = base.dev<float>((size_t) MT * HC);
    float* inj2 = base.dev<float>((size_t) MT * HC);
    float* lo = base.dev<float>((size_t) MT * LR);
    float* rs = base.dev<float>((size_t) MT * HC);
    float* xn = base.dev<float>((size_t) MT * HC * N);
    float* mixed = base.dev<float>((size_t) MT * N);
    const int64_t QMAX = std::max({N, ZV, NH * HD, FF});
    uint8_t* xq = base.dev<uint8_t>(q81(QMAX, MT));
    uint8_t* dq = base.dev<uint8_t>(2 * k::iq_mmq_act_bytes(MT, std::max({N, ZV, NH * HD})));
    std::vector<int32_t> hb, hi;
    for (int T = 0; T <= MT; ++T) { hb.push_back(0); hb.push_back(T); }
    for (int t = 0; t < MT; ++t) hi.push_back(t);
    int32_t* dbounds = put(base, hb);
    int32_t* dids = put(base, hi);
    fill_f32(R, (size_t) MT * HC * N, 11, -1, 1);
    fill_f32(bo, (size_t) MT * N, 12, -1, 1);
    fill_f32(inj, (size_t) MT * HC, 13, -1, 1);
    fill_f32(inj2, (size_t) MT * HC, 14, -1, 1);
    fill_f32(mixed, (size_t) MT * N, 15, -1, 1);
    Q().wait();
    k::native_quantize_q8_1(mixed, xq, (int) N, MT, CS());

    // the BF16 GEMVs' and the GDN step's forms, as Verifier::tune_device picks them
    k::gdn_step_tune((int) C, (int) HK, (int) HV, CS());
    for (const int64_t n_out : {NE, ID, IQ * ID, (int64_t) 10240, N}) k::bf16_gemv_fp32_mmvf_multi_tune(N, n_out, CS());
    Q().wait_and_throw();

    std::printf("\n%-44s", "kernel (median us; rows ->)");
    for (int T : o.rows) std::printf(" %9d", T);
    std::printf("\n");

    // ======================= the hyper-connection reads (each layer twice) and mixed's q8_1
    {
        Arena a;
        const int64_t D = HC * N;
        const size_t q8b_down = (size_t) LR * (D + D / 32 * 2), q8b_up = (size_t) D * (LR + LR / 32 * 2);
        uint8_t* q8d = a.dev<uint8_t>(q8b_down);
        uint8_t* q8u = a.dev<uint8_t>(q8b_up);
        float* wn = a.dev<float>(D);
        uint16_t* wi = a.dev<uint16_t>((size_t) HC * D);
        fill_bytes(q8d, q8b_down, 21);
        fill_bytes(q8u, q8b_up, 22);
        fill_f32(wn, D, 23, 0.5f, 1.5f);
        fill_bf16(wi, (size_t) HC * D, 24, 0.05f);
        std::vector<k::FusedGrArgs> fa(MT);
        for (int t = 0; t < MT; ++t) {
            k::FusedGrArgs& x = fa[t];
            x.R = R + (size_t) t * D; x.R_out = R + (size_t) t * D; x.apply = true;
            x.bo_prev = bo + (size_t) t * N; x.inj_prev = inj2 + (size_t) t * HC;
            x.w_norm = wn; x.w_inject = wi; x.q8_down = q8d; x.q8_up = q8u; x.eps = eps;
            x.lo = lo + (size_t) t * LR; x.rs = rs + (size_t) t * HC; x.inject_out = inj + (size_t) t * HC;
            x.mixed = mixed + (size_t) t * N;
        }
        k::HcMmqScratch hs;
        if (k::hc_mmq_min() > 0 || k::hc_xmx_min() > 0) {
            hs.up = a.dev<float>((size_t) MT * D);
            hs.xq = a.dev<uint8_t>(k::hc_mmq_xq_bytes(MT));
            hs.ids = dids;
        }
        // STRATA_BATCH_HCX (hc_xmx.hpp) takes a window before the MMQ opt-ins, as in Verifier::record_window
        const bool hcx_engine = k::hcx_engine();
        const k::HcxScratch xs = k::hcx_carve(a.dev<uint8_t>(k::hcx_scratch_bytes(MT)), MT);
        // Resolve before timing, using the same weights/scratch/capability/row policy as production.
        std::map<int, k::dmmq::HcPath> paths;
        std::map<int, bool> hcx;
        for (int T : o.rows) {
            hs.bounds = dbounds + 2 * T;
            const auto path = k::hc_mmq_path(fa[0], lo, hs, T);
            paths[T] = path;
            hcx[T] = k::hcx_take(fa.data(), T);
            std::printf("hc Q8 rows=%d path=%s MMQ-min=%d decode-min=%d hcx-min=%d (slices %d, XMX %d)\n", T,
                        hcx[T] ? "hcx" : path == k::dmmq::HcPath::decode ? "decode" :
                        path == k::dmmq::HcPath::mmq ? "mmq" : "fused", k::hc_mmq_min(), k::hc_xmx_min(),
                        k::hcx_min(), k::hcx_slices(), (int) hcx_engine);
        }
        bench("hc read (Q8_0, x2 a layer)",
              [&](int T) { return (double) (q8b_down + q8b_up) + (double) HC * D * 2 + T * (D * 4.0 * 4 + N * 8.0); },
              [](int) {}, [&](int T, int) {
                  hs.bounds = dbounds + 2 * T;
                  if (hcx.at(T)) k::hcx_read(fa.data(), T, xs, CS(), k::hcx_slices());
                  else if (paths.at(T) != k::dmmq::HcPath::fused) k::fused_gr_read_mmq(fa.data(), T, xn, hs, CS());
                  else k::fused_gr_read_multi(fa.data(), T, xn, CS());
              });
        // hcx's own rows beside it (whatever the settings), one and two activation slices, where the card has XMX
        for (int S = 1; hcx_engine && S <= k::kHcxMaxSlices; ++S)
            bench("hc read hcx S" + std::to_string(S) + " (one read)",
                  [&](int T) { return (double) (q8b_down + q8b_up) + (double) HC * D * 2 + T * (D * 4.0 * 3 + N * 8.0); },
                  [](int) {}, [&, S](int T, int) { k::hcx_read(fa.data(), T, xs, CS(), S); });
        bench("quantize mixed q8_1", [&](int T) { return T * (N * 4.0 + N / 32 * 36.0); }, [](int) {},
              [&](int T, int) { k::native_quantize_q8_1(mixed, xq, (int) N, T, CS()); });
    }

    // ======================= the dense products (both paths: q8_1 GEMV, and XMX iq_mmq with its quantize)
    // role: the engine's MmvqRoleScope for the product (verify and layer tag gdn qkv and ssm_out), which the
    // STRATA_MMVQ_SLM rules key on; the rest stay MmvqRole::other there and here
    using Role = k::MmvqRole;
    struct DenseOp { const char* name; const char* tensor; int64_t n_in, n_out; bool qsa; bool mmq_ok; Role role; };
    const std::vector<DenseOp> dense_ops = {
        {"qkv", "attn_qkv", N, QKV, false, true, Role::qkv},  {"z (attn_gate)", "attn_gate", N, ZV, false, true, Role::other},
        {"ssm_out", "ssm_out", ZV, N, false, true, Role::ssm_out}, {"q|gate", "attn_q", N, NH * 2 * HD, true, true, Role::other},
        {"k", "attn_k", N, NKV * HD, true, false, Role::other},  {"v", "attn_v", N, NKV * HD, true, false, Role::other},
        {"o (attn_output)", "attn_output", NH * HD, N, true, true, Role::other},
    };
    auto native_type = [](int t, int64_t n_in) {
        if (t == 14 && k::native_q6_k_rows_ok((int) n_in)) return k::kNativeQ6KRows;
        if (t == 8 && k::native_q8_0_rows_ok((int) n_in)) return k::kNativeQ8Rows;
        return t;
    };
    auto dense_key = [](const DenseOp& d, int type, int path) {   // path 0 mmvq, 1 mmq, 2 the decode XMX product
        return std::string("dense ") + d.name + " t" + std::to_string(type) + (path == 2 ? " xmx" : path ? " mmq" : " mmvq");
    };
    // STRATA_BATCH_DENSE_XMX (homelab K1), as the engine reads it: the decode XMX product from that many rows
    const int xmx_min = [] {
        const char* e = std::getenv("STRATA_BATCH_DENSE_XMX");
        return k::dmmq::dispatch_min(e && *e ? std::atoi(e) : 0, std::getenv("STRATA_DMMQ_MIN_ROWS"));
    }();
    auto xmx_ok = [&](int t, int64_t n_in, int64_t n_out) { return k::iq_mmq_decode_ok(Q(), t, n_out, n_in, MT); };
    {
        std::set<std::pair<int, int>> done;   // (op, gguf type)
        for (const Layer& L : sp.layers)
            for (size_t oi = 0; oi < dense_ops.size(); ++oi) {
                const DenseOp& d = dense_ops[oi];
                if (d.qsa != L.qsa || !L.type.count(d.tensor) || !done.insert({(int) oi, L.type.at(d.tensor)}).second) continue;
                Arena a;
                const int gt = L.type.at(d.tensor), t = native_type(gt, d.n_in);
                const size_t wb = (size_t) wbytes(gt, d.n_in, d.n_out);
                void* w = a.dev<uint8_t>(wb);
                fill_bytes(w, wb, 100 + (uint32_t) oi * 7 + (uint32_t) gt);
                if (t == k::kNativeQ6KRows) k::native_q6_k_to_rows(w, (int) d.n_in, (int) d.n_out, CS());
                if (t == k::kNativeQ8Rows) k::native_q8_0_to_rows(w, (int) d.n_in, (int) d.n_out, CS());
                float* x = a.dev<float>((size_t) MT * d.n_in);
                float* y = a.dev<float>((size_t) MT * d.n_out);
                uint8_t* xqi = a.dev<uint8_t>(q81(d.n_in, MT));
                fill_f32(x, (size_t) MT * d.n_in, 31, -1, 1);
                Q().wait();
                k::native_quantize_q8_1(x, xqi, (int) d.n_in, MT, CS());
                const double act = d.n_in * 4.0 + d.n_out * 4.0;
                bench(dense_key(d, gt, 0), [&](int T) { return (double) wb + T * act; }, [](int) {},
                      [&](int T, int) {
                          k::MmvqRoleScope role(d.role);
                          k::native_mmvq(t, w, xqi, y, (int) d.n_in, (int) d.n_out, T, CS());
                      });
                if (d.mmq_ok && k::iq_mmq_type_ok(t) && d.n_out % 128 == 0 && d.n_in % 32 == 0) {
                    const bool sums = k::iq_mmq_type_sums(t), dup = k::iq_mmq_type_dup(t);
                    const int64_t cols = dup ? 2 * d.n_in : d.n_in;
                    bench(dense_key(d, gt, 1), [&](int T) { return (double) wb + T * act; }, [](int) {},
                          [&](int T, int) {
                              k::iq_mmq_quantize(Q(), x, nullptr, dq, d.n_in, d.n_in, T, sums, dup);
                              k::iq_mmq_grouped(Q(), t, w, 0, d.n_out, cols, 1, dq, T, dbounds + 2 * T, T, dids, y,
                                                d.n_out, nullptr, 0, T);
                          });
                }
                if (xmx_min > 0 && xmx_ok(t, d.n_in, d.n_out)) {
                    const bool sums = k::iq_mmq_type_sums(t);
                    bench(dense_key(d, gt, 2), [&](int T) { return (double) wb + T * act; }, [](int) {},
                          [&](int T, int) {
                              k::iq_mmq_quantize(Q(), x, nullptr, dq, d.n_in, d.n_in, T, sums, false);
                              k::iq_mmq_decode(Q(), t, w, d.n_out, d.n_in, dq, T, y, d.n_out);
                          });
                }
            }
    }

    // ======================= GDN: the conv, alpha/beta, the recurrence (every slot its own state, self-commit)
    {
        Arena a;
        const size_t sf = (size_t) DS * HV * DS, hf = (size_t) C * (DC - 1);
        float* state = a.dev<float>((sf + hf) * MT);
        float* qkv = a.dev<float>((size_t) MT * C);
        float* h = a.dev<float>((size_t) MT * C);
        float* conv_w = a.dev<float>((size_t) C * DC);
        float* gate = a.dev<float>((size_t) MT * HV);
        float* beta = a.dev<float>((size_t) MT * HV);
        float* z = a.dev<float>((size_t) MT * ZV);
        float* y = a.dev<float>((size_t) MT * ZV);
        float* gamma = a.dev<float>(DS);
        uint16_t* wa = a.dev<uint16_t>((size_t) N * HV);
        uint16_t* wbt = a.dev<uint16_t>((size_t) N * HV);
        float* dt = a.dev<float>(HV);
        float* ssm_a = a.dev<float>(HV);
        fill_f32(state, (sf + hf) * MT, 41, -0.01f, 0.01f);
        fill_f32(qkv, (size_t) MT * C, 42, -1, 1);
        fill_f32(conv_w, (size_t) C * DC, 43, -0.5f, 0.5f);
        fill_f32(gate, (size_t) MT * HV, 44, -1.0f, -0.1f);
        fill_f32(beta, (size_t) MT * HV, 45, 0.1f, 0.9f);
        fill_f32(z, (size_t) MT * ZV, 46, -1, 1);
        fill_f32(gamma, DS, 47, 0.5f, 1.5f);
        fill_bf16(wa, (size_t) N * HV, 48, 0.05f);
        fill_bf16(wbt, (size_t) N * HV, 49, 0.05f);
        fill_f32(dt, HV, 50, -1, 1);
        fill_f32(ssm_a, HV, 51, -1, -0.1f);
        Q().wait();
        k::GdnSlotTable tab;
        for (int j = 0; j < MT; ++j) {
            tab.state[j] = state + (sf + hf) * j;
            tab.hist[j] = tab.state[j] + sf;
            tab.first[j] = j;
            tab.ntok[j] = 1;
        }
        auto tab_n = [&](int T) { k::GdnSlotTable t = tab; t.n = T; return t; };
        k::gdn_conv_l2_slots(tab_n(MT), qkv, conv_w, h, (int) C, (int) (2 * HK), eps, CS(), true);
        bench("gdn conv+l2 (slots, commit)", [&](int T) { return T * (C * 4.0 * 2 + hf * 4.0 * 2) + C * DC * 4.0; },
              [](int) {}, [&](int T, int) { k::gdn_conv_l2_slots(tab_n(T), qkv, conv_w, h, (int) C, (int) (2 * HK), eps, CS(), true); });
        bench("gdn alpha/beta (bf16 multi)", [&](int T) { return 2.0 * N * HV * 2 + T * (N * 4.0 + HV * 8.0); }, [](int) {},
              [&](int T, int) { k::gdn_ab_multi(mixed, wa, wbt, dt, ssm_a, gate, beta, (int) N, (int) HV, T, CS()); });
        if (k::gdn_ab_fused_in_conv())   // STRATA_GDN_AB_FORM 4..6: alpha/beta inside the conv's launch
            bench("gdn conv+l2+alpha/beta (one launch)",
                  [&](int T) { return T * (C * 4.0 * 2 + hf * 4.0 * 2) + C * DC * 4.0 + 2.0 * N * HV * 2 + T * (N * 4.0 + HV * 8.0); },
                  [](int) {}, [&](int T, int) {
                      const k::GdnAbArgs ab{mixed, wa, wbt, dt, ssm_a, gate, beta, (int) N, (int) HV, T};
                      k::gdn_conv_l2_slots(tab_n(T), qkv, conv_w, h, (int) C, (int) (2 * HK), eps, CS(), true, &ab);
                  });
        fill_f32(gate, (size_t) MT * HV, 44, -1.0f, -0.1f);
        bench("gdn step+norm (slots, commit, q8_1 out)", [&](int T) { return T * (sf * 8.0 + C * 4.0 + ZV * 8.0 + q81(ZV, 1)); },
              [](int) {}, [&](int T, int) {
                  k::gdn_step_norm_slots(tab_n(T), h, (int) C, gate, beta, z, gamma, eps, y, (int) HK, (int) HV, true,
                                         CS(), 0, xq);
              });
    }

    // ======================= QSA: the rows' own streamed int8 K/V, indexer and selection
    {
        Arena a;
        const int P = (int) s.page_size;
        const int ctx_alloc = o.ctx + 64;
        const int64_t n_blocks = ctx_alloc / P, n_slots = o.resident_cells / P;
        const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
        const int64_t max_blocks = o.max_ctx / s.idx_block + 2;
        const int64_t HDK = NKV * HD, G8 = HD / k::KV_Q8_GROUP;
        const size_t TS = (size_t) (s.idx_block - 1) * ID, PR = (size_t) (ctx_alloc / s.idx_block + 2) * ID;
        struct Slot {
            int8_t *kq, *vq;
            uint16_t *ks, *vs;
            k::KvHostPools host;
            k::KvStreamMap map;
            float *tail, *dead, *pooled, *snap;
            int32_t* bpos;
        };
        std::vector<Slot> sl(MT);
        std::vector<int32_t> hstep((size_t) MT * k::kStepCount), hpos((size_t) MT * (NH + NKV + IQ));
        for (int r = 0; r < MT; ++r) {
            const int pos = o.ctx - 1 - 5 * r;
            k::qsa_step_fill(hstep.data() + (size_t) r * k::kStepCount, pos, s);
            for (int64_t j = 0; j < NH; ++j) hpos[(size_t) (r * NH + j)] = pos;
            for (int64_t j = 0; j < NKV; ++j) hpos[(size_t) (MT * NH + r * NKV + j)] = pos;
            for (int64_t j = 0; j < IQ; ++j) hpos[(size_t) (MT * (NH + NKV) + r * IQ + j)] = pos;
        }
        int32_t* step = put(a, hstep);
        int32_t* pos = put(a, hpos);
        const int32_t* pos_k = pos + MT * NH;
        const int32_t* pos_i = pos + MT * (NH + NKV);
        for (int r = 0; r < MT; ++r) {
            Slot& x = sl[r];
            x.kq = a.dev<int8_t>((size_t) n_slots * P * HDK); x.vq = a.dev<int8_t>((size_t) n_slots * P * HDK);
            x.ks = a.dev<uint16_t>((size_t) n_slots * P * NKV * G8); x.vs = a.dev<uint16_t>((size_t) n_slots * P * NKV * G8);
            x.host.k_q = a.host<int8_t>((size_t) n_blocks * P * HDK); x.host.v_q = a.host<int8_t>((size_t) n_blocks * P * HDK);
            x.host.k_scale = a.host<uint16_t>((size_t) n_blocks * P * NKV * G8);
            x.host.v_scale = a.host<uint16_t>((size_t) n_blocks * P * NKV * G8);
            fill_bytes(x.host.k_q, (size_t) n_blocks * P * HDK, 60 + r);
            fill_bytes(x.host.v_q, (size_t) n_blocks * P * HDK, 61 + r);
            fill_bytes(x.host.k_scale, (size_t) n_blocks * P * NKV * G8 * 2, 62 + r);
            fill_bytes(x.host.v_scale, (size_t) n_blocks * P * NKV * G8 * 2, 63 + r);
            x.map.page_table = a.dev<int32_t>(n_blocks);
            x.map.slot_block = a.dev<int32_t>(n_slots); x.map.slot_stamp = a.dev<int32_t>(n_slots);
            x.map.slot_ref = a.dev<int32_t>(n_slots); x.map.ctl = a.dev<int32_t>(k::kKvCtlInts);
            x.map.miss_block = a.dev<int32_t>(n_slots); x.map.miss_slot = a.dev<int32_t>(n_slots);
            x.map.n_blocks = n_blocks; x.map.n_slots = n_slots;
            k::kv_stream_reset(x.map, CS());
            x.tail = a.dev<float>(TS); x.dead = a.dev<float>(ID); x.pooled = a.dev<float>(PR); x.snap = a.dev<float>(TS);
            x.bpos = a.dev<int32_t>(1);
            fill_f32(x.tail, TS, 64 + r, -1, 1); fill_f32(x.dead, ID, 65 + r, -1, 1); fill_f32(x.pooled, PR, 66 + r, -1, 1);
        }
        auto pools = [&](int r) {
            k::QsaAttnPools p;
            p.k_q = sl[r].kq; p.v_q = sl[r].vq; p.k_scale = sl[r].ks; p.v_scale = sl[r].vs; p.page_table = sl[r].map.page_table;
            return p;
        };
        // the layer's weights (BF16 indexer projections, norms) and the rows' activations
        uint16_t* wik = a.dev<uint16_t>((size_t) N * ID);
        uint16_t* wiq = a.dev<uint16_t>((size_t) N * IQ * ID);
        float* nk = a.dev<float>(HD); float* nq = a.dev<float>(HD); float* nik = a.dev<float>(ID); float* niq = a.dev<float>(ID);
        float* idx_raw = a.dev<float>((size_t) MT * ID);
        float* qidx = a.dev<float>((size_t) MT * IQ * ID);
        float* kcur = a.dev<float>((size_t) MT * HDK);
        float* vcur = a.dev<float>((size_t) MT * HDK);
        float* qfull = a.dev<float>((size_t) MT * NH * 2 * HD);
        float* qcur = a.dev<float>((size_t) MT * NH * HD);
        float* scores = a.dev<float>((size_t) MT * max_blocks);
        int32_t* sel = a.dev<int32_t>((size_t) MT * cap);
        const uint64_t asf = k::qsa_decode_attn_scratch_floats(cap, s);
        float* ascr = a.dev<float>((size_t) MT * asf);
        float* attn = a.dev<float>((size_t) MT * NH * HD);
        float* attn32 = a.dev<float>((size_t) MT * NH * HD);
        fill_bf16(wik, (size_t) N * ID, 70, 0.05f); fill_bf16(wiq, (size_t) N * IQ * ID, 71, 0.05f);
        fill_f32(nk, HD, 72, 0.5f, 1.5f); fill_f32(nq, HD, 73, 0.5f, 1.5f); fill_f32(nik, ID, 74, 0.5f, 1.5f);
        fill_f32(niq, ID, 75, 0.5f, 1.5f);
        fill_f32(kcur, (size_t) MT * HDK, 76, -1, 1); fill_f32(vcur, (size_t) MT * HDK, 77, -2, 2);
        fill_f32(qfull, (size_t) MT * NH * 2 * HD, 78, -1, 1); fill_f32(idx_raw, (size_t) MT * ID, 79, -1, 1);
        fill_f32(qidx, (size_t) MT * IQ * ID, 80, -1, 1);
        Q().wait();
        const k::RopeScaling& sc = k::rope_scaling();
        k::QsaScoreSlots stab;
        k::QsaAttnSlots atab;
        for (int r = 0; r < MT; ++r) {
            stab.pooled[r] = sl[r].pooled; stab.dead[r] = sl[r].dead;
            if (!k::qsa_attn_slots_set(atab, r, pools(r))) throw std::runtime_error("attention table refused a row");
        }
        auto itab_n = [&](int T) {
            k::QsaIndexerSlots it;
            for (int r = 0; r < T; ++r) {
                const int j = it.n++;
                it.snap[j] = sl[r].snap; it.tail[j] = sl[r].tail; it.dead[j] = sl[r].dead; it.pooled[j] = sl[r].pooled;
                it.block_pos[j] = sl[r].bpos; it.first[j] = r; it.ntok[j] = 1; it.max_cells[j] = ctx_alloc;
            }
            return it;
        };
        // a real selection for every row (scores, top-k), and its blocks resident (the resolve's misses, untimed)
        k::qsa_block_scores_slots(stab, qidx, step, MT, max_blocks, s, scores, CS());
        k::qsa_block_topk(scores, step, MT, max_blocks, cap, s, sel, CS());
        for (int r = 0; r < MT; ++r)
            k::kv_stream_resolve(sl[r].map, pools(r), sl[r].host, k::kKvInt8, sel + (size_t) r * cap,
                                 step + (size_t) r * k::kStepCount, 1, cap, s, CS());
        Q().wait_and_throw();
        const double width = (double) std::min<int64_t>(o.ctx, cap);
        bench("qsa idx k_proj (bf16 multi)", [&](int T) { return N * ID * 2.0 + T * (N + ID) * 4.0; }, [](int) {},
              [&](int T, int) {
                  if (T > 1) k::bf16_gemv_fp32_mmvf_multi(mixed, N, wik, idx_raw, ID, N, ID, T, CS());
                  else k::bf16_gemv_fp32_mmvf(mixed, wik, idx_raw, N, ID, CS());
              });
        bench("qsa k norm+rope", [&](int T) { return T * HDK * 8.0; }, [](int) {}, [&](int T, int) {
            k::native_qsa_rms_norm_rope(kcur, (int) HD, nk, kcur, (int) (T * NKV), (int) HD, (int) s.n_rot, eps, sc, pos_k, CS());
        });
        bench("qsa K/V append int8 (a launch a row, host copy)", [&](int T) { return T * HDK * 2 * 5.0; }, [](int) {},
              [&](int T, int) {
                  for (int r = 0; r < T; ++r)
                      k::kv_append_q8_step(sl[r].kq, sl[r].vq, sl[r].ks, sl[r].vs, sl[r].map.page_table,
                                           step + (size_t) r * k::kStepCount, kcur + (size_t) r * HDK,
                                           vcur + (size_t) r * HDK, s, CS(), &sl[r].host);
              });
        // the same rows' stores in one launch (STRATA_BATCH_QSA_APPEND_ROWS / _STREAMED: kv_append_q8_slots)
        k::KvAppendSlots kvtab;
        for (int r = 0; r < MT; ++r) {
            kvtab.k[r] = sl[r].kq; kvtab.v[r] = sl[r].vq; kvtab.k_scale[r] = sl[r].ks; kvtab.v_scale[r] = sl[r].vs;
            kvtab.page_table[r] = sl[r].map.page_table;
            kvtab.host_k[r] = sl[r].host.k_q; kvtab.host_v[r] = sl[r].host.v_q;
            kvtab.host_k_scale[r] = sl[r].host.k_scale; kvtab.host_v_scale[r] = sl[r].host.v_scale;
        }
        bench("qsa K/V append int8 (slots, host copy)", [&](int T) { return T * HDK * 2 * 5.0; }, [](int) {},
              [&](int T, int) { k::kv_append_q8_slots(kvtab, step, (int) k::kStepCount, kcur, vcur, (int) HDK, T, 2, s, CS()); });
        bench("qsa indexer append (slots)", [&](int T) { return T * (TS * 12.0 + ID * 12.0); }, [](int) {},
              [&](int T, int) {
                  k::native_qsa_indexer_append_slots(idx_raw, step + k::kStepPos, k::kStepCount, 0, nik, eps, itab_n(T), s, sc, CS());
              });
        bench("qsa q split+norm+rope", [&](int T) { return T * NH * HD * 8.0; }, [](int) {}, [&](int T, int) {
            k::native_qsa_rms_norm_rope(qfull, (int) (2 * HD), nq, qcur, (int) (T * NH), (int) HD, (int) s.n_rot, eps, sc, pos, CS());
        });
        bench("qsa idx q_proj (bf16 multi)", [&](int T) { return N * IQ * ID * 2.0 + T * (N + IQ * ID) * 4.0; }, [](int) {},
              [&](int T, int) {
                  if (T > 1) k::bf16_gemv_fp32_mmvf_multi(mixed, N, wiq, qidx, IQ * ID, N, IQ * ID, T, CS());
                  else k::bf16_gemv_fp32_mmvf(mixed, wiq, qidx, N, IQ * ID, CS());
              });
        bench("qsa idx q norm+rope", [&](int T) { return T * IQ * ID * 8.0; }, [](int) {}, [&](int T, int) {
            k::native_qsa_rms_norm_rope(qidx, (int) ID, niq, qidx, (int) (T * IQ), (int) ID, (int) s.n_rot, eps, sc, pos_i, CS());
        });
        bench("qsa block scores (slots)", [&](int T) { return T * ((double) o.ctx / s.idx_block * ID * 4 + max_blocks * 4.0); },
              [](int) {}, [&](int T, int) { k::qsa_block_scores_slots(stab, qidx, step, T, max_blocks, s, scores, CS()); });
        bench("qsa block top-k", [&](int T) { return T * ((double) o.ctx / s.idx_block * 4 + cap * 4.0); }, [](int) {},
              [&](int T, int) { k::qsa_block_topk(scores, step, T, max_blocks, cap, s, sel, CS()); });
        bench("qsa kv resolve (a launch a row, all hits)", [&](int T) { return T * cap * 8.0; }, [](int) {},
              [&](int T, int) {
                  for (int r = 0; r < T; ++r)
                      k::kv_stream_resolve(sl[r].map, pools(r), sl[r].host, k::kKvInt8, sel + (size_t) r * cap,
                                           step + (size_t) r * k::kStepCount, 1, cap, s, CS());
              });
        // the same rows in two launches (STRATA_BATCH_QSA_RESOLVE_ROWS / _STREAMED: kv_stream_resolve_slots)
        k::KvResolveSlots rtab;
        for (int r = 0; r < MT; ++r) { rtab.map[r] = sl[r].map; rtab.slots[r] = pools(r); rtab.host[r] = sl[r].host; }
        bench("qsa kv resolve (slots, all hits)", [&](int T) { return T * cap * 8.0; }, [](int) {},
              [&](int T, int) { k::kv_stream_resolve_slots(rtab, k::kKvInt8, sel, step, T, cap, s, CS()); });
        bench("qsa decode attention (slots, int8)",
              [&](int T) { return T * width * (double) k::kv_q8_bytes_per_cell(s) + T * (NH * HD * 8.0 + asf * 4.0); },
              [](int) {}, [&](int T, int) {
                  k::qsa_decode_attn_slots(qcur, atab, sel, step, cap, s, ascr, attn, T, CS());
              });
        bench("qsa gate apply", [&](int T) { return T * NH * HD * 16.0; }, [](int) {},
              [&](int T, int) { k::native_qsa_gate_apply(attn, qfull, attn32, (int) (T * NH), (int) HD, CS()); });
        bench("qsa quantize attn q8_1", [&](int T) { return T * (NH * HD * 4.0 + q81(NH * HD, 1)); }, [](int) {},
              [&](int T, int) { k::native_quantize_q8_1(attn32, xq, (int) (NH * HD), T, CS()); });
    }

    // ======================= the router, the plan, the shared expert, the experts, the combine
    {
        Arena a;
        uint16_t* wr = a.dev<uint16_t>((size_t) N * NE);
        uint16_t* wsg = a.dev<uint16_t>(N);
        float* logits = a.dev<float>((size_t) MT * NE);
        int32_t* ids = a.dev<int32_t>((size_t) MT * K);
        float* w = a.dev<float>((size_t) MT * K);
        float* sh_g = a.dev<float>(MT);
        uint8_t* nat_xq = a.dev<uint8_t>(q81(N, MT));
        fill_bf16(wr, (size_t) N * NE, 90, 0.05f);
        fill_bf16(wsg, N, 91, 0.05f);
        fill_f32(logits, (size_t) MT * NE, 92, -4, 4);
        fill_f32(w, (size_t) MT * K, 93, 0.05f, 0.15f);
        Q().wait();
        k::quantize_q8_1_rows(mixed, MT, N, nat_xq, CS());
        bench("router gemv (+shexp gate, bf16 multi)", [&](int T) { return N * (NE + 1) * 2.0 + T * (N + NE) * 4.0; },
              [](int) {}, [&](int T, int) {
                  if (T == 1) { k::bf16_gemv_fp32_mmvf(mixed, wr, logits, N, NE, CS()); return; }
                  if (!k::bf16_gemv_fp32_mmvf_multi_aux(mixed, N, wr, logits, NE, N, NE, T, wsg, sh_g, 1, CS()))
                      k::bf16_gemv_fp32_mmvf_multi(mixed, N, wr, logits, NE, N, NE, T, CS());
              });
        bench("router top-10", [&](int T) { return T * NE * 4.0; }, [](int) {},
              [&](int T, int) { k::native_router_top10_multi(logits, ids, w, T, CS()); });
        if (k::bf16_router_fuse())   // STRATA_ROUTER_FUSE: the GEMV, the gate and the top-10 in one launch (2+ rows)
            bench("router fused (gemv+gate+top-10)", [&](int T) { return N * (NE + 1) * 2.0 + T * (N + NE) * 4.0; },
                  [](int) {}, [&](int T, int) {
                      if (T > 1 && k::bf16_router_fused(mixed, N, wr, logits, N, T, wsg, sh_g, 1, ids, w, CS())) return;
                      k::bf16_gemv_fp32_mmvf(mixed, wr, logits, N, NE, CS());
                      k::native_router_top10_multi(logits, ids, w, T, CS());
                  });
        bench("quantize xm q8_1 (experts)", [&](int T) { return T * (N * 4.0 + q81(N, 1)); }, [](int) {},
              [&](int T, int) { k::quantize_q8_1_rows(mixed, T, N, nat_xq, CS()); });

        // the shared expert per (gate, up, down) type triple of the stage's layers
        {
            float* sg = a.dev<float>((size_t) MT * FF);
            float* su = a.dev<float>((size_t) MT * FF);
            float* sh_out = a.dev<float>((size_t) MT * N);
            uint8_t* sq = a.dev<uint8_t>(q81(std::max(N, FF), MT));
            std::set<std::tuple<int, int, int>> done;
            for (const Layer& L : sp.layers) {
                const int g = L.type.at("ffn_gate_shexp"), u = L.type.at("ffn_up_shexp"), d = L.type.at("ffn_down_shexp");
                if (!done.insert({g, u, d}).second) continue;
                Arena b;
                k::NativeSharedWeights nw;
                const size_t gb = (size_t) wbytes(g, N, FF), ub = (size_t) wbytes(u, N, FF), db = (size_t) wbytes(d, FF, N);
                void* wg = b.dev<uint8_t>(gb); void* wu = b.dev<uint8_t>(ub); void* wd = b.dev<uint8_t>(db);
                fill_bytes(wg, gb, 94); fill_bytes(wu, ub, 95); fill_bytes(wd, db, 96);
                nw.gate_type = native_type(g, N); nw.up_type = native_type(u, N); nw.down_type = native_type(d, FF);
                if (nw.gate_type == k::kNativeQ6KRows) k::native_q6_k_to_rows(wg, (int) N, (int) FF, CS());
                if (nw.up_type == k::kNativeQ6KRows) k::native_q6_k_to_rows(wu, (int) N, (int) FF, CS());
                if (nw.down_type == k::kNativeQ6KRows) k::native_q6_k_to_rows(wd, (int) FF, (int) N, CS());
                nw.gate_data = wg; nw.up_data = wu; nw.down_data = wd; nw.q8_1 = sq;
                const std::string key = "shared expert t" + std::to_string(g) + "/" + std::to_string(u) + "/" + std::to_string(d);
                bench(key, [&](int T) { return (double) (gb + ub + db) + T * (N * 8.0 + FF * 12.0); },
                      [](int) {}, [&](int T, int) {
                          k::shared_expert_multi(T, mixed, nullptr, nw, wsg, sg, su, sh_g, sh_out, N, FF, CS(), T > 1,
                                                 T > 1, nat_xq);
                      });
            }
        }

        // the routed experts: per (gate/up, down) pair, `random` (K distinct of n_expert a row, as decode routes) and
        // `shared` (every row the same K: the reuse ceiling); kSets routings in rotation, each its own plan
        constexpr int kSets = 8;
        const int64_t capx = (int64_t) MT * K;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const int64_t plan_i32 = ptr_off + 6 * capx + 16;
        float* parts = a.dev<float>((size_t) capx * N);
        float* hit_out = a.dev<float>((size_t) capx * N);
        float* shared = a.dev<float>((size_t) MT * N);
        float* ymiss = a.host<float>((size_t) capx * N);
        uint32_t* skip = a.dev<uint32_t>(1);
        uint32_t* skip_scr = a.dev<uint32_t>(1);
        int32_t* plans = a.dev<int32_t>((size_t) kSets * plan_i32);
        int32_t* plan_scr = a.dev<int32_t>((size_t) plan_i32);
        int32_t* ids_sets = a.dev<int32_t>((size_t) kSets * capx);
        int32_t* res_sets = a.dev<int32_t>((size_t) kSets * NE);
        const uint32_t one = 1;
        Q().memcpy(skip, &one, 4).wait();
        fill_f32(shared, (size_t) MT * N, 97, -1, 1);
        std::mt19937 rng(o.seed);
        std::vector<double> distinct16(2, 0);
        std::map<int, double> distinct_at;   // random routing: distinct experts in the first T rows, mean over the sets
        auto routing = [&](bool same) {
            std::vector<int32_t> hid((size_t) kSets * capx), hres((size_t) kSets * NE, -1);
            for (int st = 0; st < kSets; ++st) {
                std::vector<int32_t> pick;
                int nslot = 0;
                for (int t = 0; t < MT; ++t) {
                    if (!same || t == 0) {
                        std::vector<int32_t> all(NE);
                        for (int e = 0; e < NE; ++e) all[e] = e;
                        std::shuffle(all.begin(), all.end(), rng);
                        pick.assign(all.begin(), all.begin() + K);
                    }
                    for (int j = 0; j < K; ++j) {
                        const int e = pick[j];
                        hid[(size_t) st * capx + t * K + j] = e;
                        if (hres[(size_t) st * NE + e] < 0) hres[(size_t) st * NE + e] = nslot++;
                    }
                }
                distinct16[same] += (double) nslot / kSets;
                if (!same)
                    for (int T : o.rows) {
                        std::set<int32_t> seen(hid.begin() + (ptrdiff_t) st * capx, hid.begin() + (ptrdiff_t) st * capx + T * K);
                        distinct_at[T] += (double) seen.size() / kSets;
                    }
            }
            Q().memcpy(ids_sets, hid.data(), hid.size() * 4);
            Q().memcpy(res_sets, hres.data(), hres.size() * 4).wait();
        };
        std::set<std::pair<int, int>> done;
        bool sharing_printed = false;
        for (const Layer& L : sp.layers) {
            const int gu = L.type.at("ffn_gate_exps"), dn = L.type.at("ffn_down_exps");
            if (!done.insert({gu, dn}).second) continue;
            if (!k::native_expert_supported(gu, dn, N, FF)) throw std::runtime_error("expert pair not supported");
            const k::NativeExpertLayout Lx = k::native_expert_layout(gu, dn, N, FF);
            Arena b;
            uint8_t* pool = b.dev<uint8_t>(Lx.bytes * (size_t) capx);
            void* scr = b.dev<uint8_t>(k::native_expert_scratch_bytes(capx, FF));
            fill_bytes(pool, Lx.bytes * (size_t) capx, 98);
            // STRATA_EXPERT_XMX's path, forced (kNativeExpertFastXmx) where this device and pair take it: measured
            // beside the configured path as "<key> xmx" (run without STRATA_EXPERT_XMX for the A/B)
            k::native_expert_fast_override(k::kNativeExpertFastXmx);
            const bool xmx_here = k::native_expert_xmx_taken(Lx, capx, CS());
            k::native_expert_fast_override(-1);
            for (int same = 0; same < 2; ++same) {
                distinct16[same] = 0;
                if (!same) distinct_at.clear();
                routing(same);
                const std::string key = "experts t" + std::to_string(gu) + "/" + std::to_string(dn) + (same ? " shared" : " random");
                // plans built by the engine's own resident_plan (untimed) for the measured row count
                auto setup = [&](int T) {
                    for (int st = 0; st < kSets; ++st)
                        k::resident_plan(ids_sets + (size_t) st * capx, (int) (T * K), (int) K, res_sets + (size_t) st * NE,
                                         (int) NE, pool, nullptr, (long long) Lx.bytes, plans + (size_t) st * plan_i32,
                                         capx, skip_scr, 1, CS());
                };
                const double blob = (double) Lx.bytes;
                bench(key, [&](int) { return blob * distinct16[same] + capx * (N * 4.0 + 64); }, setup, [&](int T, int rep) {
                    const int32_t* pl = plans + (size_t) (rep % kSets) * plan_i32;
                    const int32_t* p_start = pl + 4;
                    const int32_t* p_dst = p_start + capx + 1;
                    const int32_t* p_tok = p_dst + capx;
                    const auto* p_ptr = reinterpret_cast<const unsigned long long*>(pl + ptr_off);
                    const int64_t cap = (int64_t) T * K;
                    k::native_expert_grouped(Lx, p_ptr, p_start, pl, p_dst, p_tok, cap, cap, nat_xq, scr, hit_out, CS(), cap);
                });
                if (xmx_here)
                    bench(key + " xmx", [&](int) { return blob * distinct16[same] + capx * (N * 4.0 + 64); }, setup,
                          [&](int T, int rep) {
                              const int32_t* pl = plans + (size_t) (rep % kSets) * plan_i32;
                              const int32_t* p_start = pl + 4;
                              const int32_t* p_dst = p_start + capx + 1;
                              const int32_t* p_tok = p_dst + capx;
                              const auto* p_ptr = reinterpret_cast<const unsigned long long*>(pl + ptr_off);
                              const int64_t cap = (int64_t) T * K;
                              k::native_expert_fast_override(k::kNativeExpertFastXmx);
                              k::native_expert_grouped(Lx, p_ptr, p_start, pl, p_dst, p_tok, cap, cap, nat_xq, scr,
                                                       hit_out, CS(), cap);
                              k::native_expert_fast_override(-1);
                          });
                if (!same && !sharing_printed) {   // the routing's sharing, which the XMX path's passes depend on
                    sharing_printed = true;
                    std::printf("%-44s entries an expert (random routing, %lld of %lld):", "", (long long) K, (long long) NE);
                    for (int T : o.rows) std::printf(" %d rows %.2f;", T, T * K / std::max(distinct_at[T], 1.0));
                    std::printf("\n");
                }
            }
            std::printf("%-44s distinct experts a 16-row window: random %.1f, shared %.1f; blob %.2f MB\n", "",
                        distinct16[0], distinct16[1], Lx.bytes / 1e6);
            if (done.size() == 1) {   // the plan and the combine do not depend on the types
                routing(false);
                bench("resident plan", [&](int T) { return T * K * 8.0 + NE * 4.0; }, [](int) {}, [&](int T, int rep) {
                    k::resident_plan(ids_sets + (size_t) (rep % kSets) * capx, (int) (T * K), (int) K,
                                     res_sets + (size_t) (rep % kSets) * NE, (int) NE, pool, nullptr, (long long) Lx.bytes,
                                     plan_scr, capx, skip_scr, 1, CS());
                });
                bench("parts copy-or-zero (resident: zeros)", [&](int T) { return T * K * N * 4.0; }, [](int) {},
                      [&](int T, int) { k::copy_or_zero_from_mapped(parts, ymiss, (long long) T * K * N, skip, 1, CS()); });
                bench("moe combine (+hits, shexp gate)", [&](int T) { return T * (K * N * 8.0 + N * 12.0); },
                      [&](int T) {
                          for (int st = 0; st < kSets; ++st)
                              k::resident_plan(ids_sets + (size_t) st * capx, (int) (T * K), (int) K, res_sets + (size_t) st * NE,
                                               (int) NE, pool, nullptr, (long long) Lx.bytes, plans + (size_t) st * plan_i32,
                                               capx, skip_scr, 1, CS());
                      },
                      [&](int T, int) {
                          const int32_t* pl = plans;
                          const int32_t* p_dst = pl + 4 + capx + 1;
                          k::native_moe_combine_multi_hits(parts, hit_out, p_dst, pl + 1, w, shared, bo, N, K, T, CS(),
                                                           T > 1 ? sh_g : nullptr);
                      });
            }
        }
    }

    spin(o.spin_ms);
    copy_test("end");

    // ======================= the stage: the SPEC's layers, each the kernels its type and the row count take
    struct Use { std::string row; std::string key; double times; };
    auto layer_uses = [&](const Layer& L, int T) {
        std::vector<Use> u;
        auto dense = [&](const char* row, size_t oi, bool may_mmq) {
            const DenseOp& d = dense_ops[oi];
            const int gt = L.type.at(d.tensor), t = native_type(gt, d.n_in);
            const bool xmx = may_mmq && xmx_min > 0 && T >= xmx_min && xmx_ok(t, d.n_in, d.n_out);
            const bool mmq = may_mmq && T >= o.mmq_min && k::iq_mmq_type_ok(t) && d.n_out % 128 == 0 && d.n_in % 32 == 0;
            u.push_back({row, dense_key(d, gt, xmx ? 2 : mmq ? 1 : 0), 1});
        };
        u.push_back({"hc read (Q8_0) x2", "hc read (Q8_0, x2 a layer)", 2});
        u.push_back({"quantize mixed q8_1", "quantize mixed q8_1", 1});
        if (!L.qsa) {
            dense("gdn qkv", 0, true);
            if (k::gdn_ab_fused_in_conv()) {
                u.push_back({"gdn conv+l2 (+alpha/beta)", "gdn conv+l2+alpha/beta (one launch)", 1});
            } else {
                u.push_back({"gdn conv+l2", "gdn conv+l2 (slots, commit)", 1});
                u.push_back({"gdn alpha/beta", "gdn alpha/beta (bf16 multi)", 1});
            }
            dense("gdn z", 1, true);
            u.push_back({"gdn step+norm", "gdn step+norm (slots, commit, q8_1 out)", 1});
            dense("gdn ssm_out", 2, true);
        } else {
            u.push_back({"qsa idx k_proj", "qsa idx k_proj (bf16 multi)", 1});
            dense("qsa k", 4, false);
            dense("qsa v", 5, false);
            u.push_back({"qsa k norm+rope", "qsa k norm+rope", 1});
            u.push_back({"qsa K/V append (per row)", "qsa K/V append int8 (a launch a row, host copy)", 1});
            u.push_back({"qsa indexer append", "qsa indexer append (slots)", 1});
            dense("qsa q|gate", 3, true);
            u.push_back({"qsa q norm+rope", "qsa q split+norm+rope", 1});
            u.push_back({"qsa idx q_proj", "qsa idx q_proj (bf16 multi)", 1});
            u.push_back({"qsa idx q norm+rope", "qsa idx q norm+rope", 1});
            u.push_back({"qsa block scores", "qsa block scores (slots)", 1});
            u.push_back({"qsa block top-k", "qsa block top-k", 1});
            u.push_back({"qsa kv resolve (per row)", "qsa kv resolve (a launch a row, all hits)", 1});
            u.push_back({"qsa attention", "qsa decode attention (slots, int8)", 1});
            u.push_back({"qsa gate apply", "qsa gate apply", 1});
            u.push_back({"qsa quantize attn", "qsa quantize attn q8_1", 1});
            dense("qsa o", 6, true);
        }
        if (k::bf16_router_fuse()) {
            u.push_back({"router fused (+top-10)", "router fused (gemv+gate+top-10)", 1});
        } else {
            u.push_back({"router gemv", "router gemv (+shexp gate, bf16 multi)", 1});
            u.push_back({"router top-10", "router top-10", 1});
        }
        u.push_back({"resident plan", "resident plan", 1});
        u.push_back({"quantize xm q8_1", "quantize xm q8_1 (experts)", 1});
        u.push_back({"shared expert", "shared expert t" + std::to_string(L.type.at("ffn_gate_shexp")) + "/" +
                                          std::to_string(L.type.at("ffn_up_shexp")) + "/" +
                                          std::to_string(L.type.at("ffn_down_shexp")), 1});
        u.push_back({"routed experts (random)", "experts t" + std::to_string(L.type.at("ffn_gate_exps")) + "/" +
                                                    std::to_string(L.type.at("ffn_down_exps")) + " random", 1});
        u.push_back({"parts copy-or-zero", "parts copy-or-zero (resident: zeros)", 1});
        u.push_back({"moe combine", "moe combine (+hits, shexp gate)", 1});
        return u;
    };
    std::vector<std::string> order;
    std::map<std::string, std::map<int, double>> stage_us;
    std::map<std::string, double> stage_bytes16;
    std::map<std::string, int> uses;
    std::map<int, double> total, shared_total;
    for (int T : o.rows)
        for (const Layer& L : sp.layers)
            for (const Use& x : layer_uses(L, T)) {
                if (!g_res.count(x.key)) throw std::runtime_error("no measurement for " + x.key);
                if (!stage_us.count(x.row)) order.push_back(x.row);
                stage_us[x.row][T] += x.times * g_res[x.key].us[T];
                total[T] += x.times * g_res[x.key].us[T];
                if (T == 16) { stage_bytes16[x.row] += x.times * g_res[x.key].bytes16; uses[x.row] += 1; }
                std::string sk = x.key;
                if (sk.size() > 7 && sk.compare(sk.size() - 7, 7, " random") == 0) sk = sk.substr(0, sk.size() - 7) + " shared";
                shared_total[T] += x.times * g_res[sk].us[T];
            }
    const int n_gdn = (int) std::count_if(sp.layers.begin(), sp.layers.end(), [](const Layer& l) { return !l.qsa; });
    std::printf("\nSTAGE ROLL-UP: %zu layers (%d GDN, %zu QSA), sum of isolated kernel times, ms; slope = (t16 - t1) / 15 "
                "per layer it runs in; floor = bytes at 16 rows / %.0f GB/s (the copy kernel)\n",
                sp.layers.size(), n_gdn, sp.layers.size() - n_gdn, bw_gbs);
    std::printf("%-28s %6s", "kernel", "layers");
    for (int T : o.rows) std::printf(" %8d", T);
    std::printf(" %10s %10s %10s %8s\n", "us/row/lyr", "MB@16/lyr", "floor us", "x floor");
    for (const std::string& row : order) {
        const auto& m = stage_us[row];
        const int nl = uses[row] ? uses[row] : 1;
        std::printf("%-28s %6d", row.c_str(), nl);
        for (int T : o.rows) std::printf(" %8.3f", m.at(T) / 1e3);
        const double t1 = m.count(1) ? m.at(1) : m.begin()->second, t16 = m.count(16) ? m.at(16) : m.rbegin()->second;
        const double mb = stage_bytes16[row] / nl / 1e6, floor_us = stage_bytes16[row] / nl / (bw_gbs * 1e3);
        std::printf(" %10.1f %10.2f %10.1f %8.1f\n", (t16 - t1) / 15 / nl, mb, floor_us, (t16 / nl) / std::max(floor_us, 1e-9));
    }
    std::printf("%-28s %6s", "TOTAL (random routing)", "");
    for (int T : o.rows) std::printf(" %8.3f", total[T] / 1e3);
    std::printf("\n%-28s %6s", "TOTAL (rows share experts)", "");
    for (int T : o.rows) std::printf(" %8.3f", shared_total[T] / 1e3);
    // the routed experts on XMX (STRATA_EXPERT_XMX), where every layer's pair has its "xmx" row: the stage's sum and
    // the random-routing total with them in place of the configured experts
    {
        std::map<int, double> xmx_us, cur_us;
        bool all = true;
        for (const Layer& L : sp.layers) {
            const std::string key = "experts t" + std::to_string(L.type.at("ffn_gate_exps")) + "/" +
                                    std::to_string(L.type.at("ffn_down_exps")) + " random";
            all = all && g_res.count(key + " xmx");
            if (!all) break;
            for (int T : o.rows) {
                xmx_us[T] += g_res[key + " xmx"].us[T];
                cur_us[T] += g_res[key].us[T];
            }
        }
        if (all) {
            std::printf("\n%-28s %6s", "routed experts XMX (random)", "");
            for (int T : o.rows) std::printf(" %8.3f", xmx_us[T] / 1e3);
            std::printf("\n%-28s %6s", "TOTAL (random, XMX experts)", "");
            for (int T : o.rows) std::printf(" %8.3f", (total[T] - cur_us[T] + xmx_us[T]) / 1e3);
        } else {
            std::printf("\nrouted experts XMX: not measured (no XMX on this device, or a layer's pair it does not take)");
        }
    }
    std::printf("\nnot measured: the window's embedding gather (stage 0), layer 1's PLE block (a per-row loop), the "
                "hand-off copies, the last layer's write; the T=1 row uses the multi-row forms where the engine's "
                "one-row window takes the single-row calls (router, combine)\n");
    return 0;
}
