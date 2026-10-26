#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model-free host controls of the real primitive, layout bridge and decode consumers.

The temporary SYCL shim maps only the three scalar math calls used by the actual
q8_1_finite.hpp to host math. It does not emulate a GPU or prove SYCL lowering.
Dense/shared branches and format decoders are extracted from production source,
not rewritten as Python arithmetic or a second dispatch implementation.
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def braced(source: str, marker: str) -> str:
    """Return the production declaration/branch through its balanced body."""
    begin = source.index(marker)
    opening = source.index("{", begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


PRELUDE = r'''
#include "strata/kernels/dmmq_contract.hpp"
#include "q8_1_finite.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
namespace d = strata::kernels::dmmq;
namespace xe = strata::kernels::xe;
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
struct alignas(4) Native { _Float16 scale, raw_sum; int8_t qs[32]; };
static_assert(sizeof(Native) == 36 && offsetof(Native, qs) == 4);
using half = _Float16;
struct block_q5_0 { half d; uint8_t qh[4], qs[16]; };
struct block_q4_K { half dm[2]; uint8_t scales[12], qs[128]; };
struct block_q5_K { half dm[2]; uint8_t scales[12], qh[32], qs[128]; };
int get_int_b2(const uint8_t* p, int i) { int v; std::memcpy(&v, p + i * 4, 4); return v; }
int get_int_b4(const uint8_t* p, int i) { return get_int_b2(p, i); }
struct Sub { uint32_t v[8]; float s0, s1; float m = 0; };
template<int TY> struct Dec;
'''

NUMERIC = r'''
void numeric_contract() {
    check(!d::native_q8_1_requested(nullptr) && !d::native_q8_1_requested("") &&
          !d::native_q8_1_requested("0"), "setting absent/empty/zero is OFF");
    check(d::native_q8_1_requested("1") && d::native_q8_1_requested("2"), "explicit nonzero setting is ON");
    check(xe::q8_1_quant(0, 0, 0) == 0 && xe::q8_1_quant(-0.0f, 0, 0) == 0, "zero block never divides by zero");
    check(xe::q8_1_quant(0.5f, 1, 127) == 1 && xe::q8_1_quant(-0.5f, 1, 127) == -1,
          "native ties round away from zero");
    check(xe::q8_1_quant(std::nextafter(0.5f, 0.0f), 1, 127) == 0 &&
          xe::q8_1_quant(std::nextafter(-0.5f, 0.0f), 1, 127) == 0, "rounding edge distinguishes neighbor from tie");
    check(xe::q8_1_quant(200, 1, 200) == 127 && xe::q8_1_quant(-200, 1, 200) == -127,
          "clamp is symmetric [-127,127], not wrapping int8");
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    check(xe::q8_1_finite(inf) == 65504 && xe::q8_1_finite(-inf) == -65504 &&
          xe::q8_1_finite(65505) == 65504 && xe::q8_1_finite(-65505) == -65504, "half finite clamp both signs");
    check(std::isnan(xe::q8_1_finite(nan)), "native finite primitive preserves NaN, not an invented sanitization");
    check(xe::q8_1_finite(1.2345f) == 1.2345f && xe::q8_1_finite(65504) == 65504,
          "ordinary finite values and boundary unchanged");
    check(xe::q8_1_quant(inf, 65504, inf) == 127 && xe::q8_1_quant(-inf, 65504, inf) == -127,
          "native infinite codes clamp before integer conversion");
    int witnesses = 0;
    for (int i = 1; i <= 1024; ++i) {
        const float scale = (float) i / 127.0f;
        for (int n = 0; n < 127; ++n) {
            const float x = scale * ((float) n + 0.5f);
            const int divide = (int) std::round(x / scale), reciprocal = (int) std::round(x * (1.0f / scale));
            if (divide != reciprocal) {
                ++witnesses;
                check(xe::q8_1_quant(x, scale, scale * 127) == divide,
                      "actual code uses divide, not reciprocal multiply");
            }
        }
    }
    check(witnesses > 0, "reciprocal counterfactual has nonempty discriminating input");
    constexpr int blocks = 6;  // three contiguous rows, two blocks per row
    std::array<Native, blocks> native{};
    std::array<float, blocks * 10 + 2> storage;
    storage.fill(-9999.0f);
    void* planar = storage.data() + 1;
    auto* codes = static_cast<int8_t*>(planar);
    auto* scales = reinterpret_cast<float*>(codes + blocks * 32);
    const float unrounded = 1.0001f;
    for (int b = 0; b < blocks; ++b) {
        native[b].scale = (half) (b == 0 ? unrounded : b == 1 ? 0.0f : b == 2 ? 65504.0f : b == 3 ? 1e-10f : 0.25f);
        native[b].raw_sum = (half) (b == 4 ? nan : 123.0f);  // must NOT feed K-quant correction
        for (int j = 0; j < 32; ++j) native[b].qs[j] = b == 1 ? 0 : (int8_t) ((b * 31 + j * 7) % 255 - 127);
    }
    native[0].qs[0] = xe::q8_1_quant(unrounded * 0.49999f, unrounded, unrounded * 127);
    check(native[0].qs[0] == 0 && xe::q8_1_quant(unrounded * 0.49999f, (float) native[0].scale, 127) == 1,
          "codes precede half rounding, not quantized using the stored scale");
    for (int b = 0; b < blocks; ++b) d::native_q8_1_planar_block<half>(native.data(), planar, blocks, b, true);
    for (int b = 0; b < blocks; ++b) {
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            check(codes[b * 32 + j] == native[b].qs[j], "32-value blocks and row layout copy exact signed codes");
            sum += native[b].qs[j];
        }
        check(scales[b] == (float) native[b].scale, "planar FP32 scale expands stored binary16");
        check(scales[blocks + b] == (float) native[b].scale * sum,
              "correction uses integer-code sum, ignoring raw half sum");
    }
    check(scales[0] != unrounded && scales[0] == 1 && scales[2] == 65504 && scales[3] == 0,
          "half rounding, maximum finite and underflow storage survive adapter");
    check(storage.front() == -9999 && storage.back() == -9999, "planar writes remain inside exact allocation");
    std::fill(scales + blocks, scales + blocks * 2, -1234.0f);
    for (int b = 0; b < blocks; ++b) d::native_q8_1_planar_block<half>(native.data(), planar, blocks, b, false);
    for (int b = 0; b < blocks; ++b)
        check(scales[blocks + b] == -1234, "sum plane untouched when consumer does not request it");
    check(d::native_q8_1_storage_ok(native.data(), planar, 64, 3), "nonoverlapping complete rows accepted");
    for (int rows : {0, -1, 17})
        check(!d::native_q8_1_storage_ok(native.data(), planar, 64, rows), "invalid rows rejected");
    for (int cols : {0, -32, 31, 33, std::numeric_limits<int>::max()})
        check(!d::native_q8_1_storage_ok(native.data(), planar, cols, 3), "invalid or overflowing columns rejected");
    check(!d::native_q8_1_storage_ok(nullptr, planar, 64, 3) &&
          !d::native_q8_1_storage_ok(native.data(), nullptr, 64, 3), "null storage rejected");
    auto* bytes = reinterpret_cast<uint8_t*>(storage.data());
    check(!d::native_q8_1_storage_ok(bytes, bytes, 32, 1) &&
          !d::native_q8_1_storage_ok(bytes, bytes + 32, 32, 1) &&
          !d::native_q8_1_storage_ok(bytes + 32, bytes, 32, 1),
          "equal and both directions of overlapping ranges rejected");
    check(d::native_q8_1_storage_ok(bytes, bytes + 36, 32, 1) &&
          d::native_q8_1_storage_ok(bytes + 40, bytes, 32, 1), "adjacent ranges accepted without extra padding");
    check(!d::native_q8_1_storage_ok(native.data(), bytes + 1, 32, 1), "misaligned planar FP32 plane rejected");
}

int code(const Sub& decoded, int j) { return (int8_t) ((decoded.v[j / 4] >> (8 * (j % 4))) & 255); }
void formats_contract() {
    block_q5_0 q50{}; q50.d = (half) 0.5f;
    block_q4_K q4{}; q4.dm[0] = (half) 0.125f; q4.dm[1] = (half) 0.25f;
    block_q5_K q5{}; q5.dm[0] = q4.dm[0]; q5.dm[1] = q4.dm[1];
    // Uniform 6-bit scale/min fields include nonzero high bits in subblocks 4..7.
    for (int j = 0; j < 4; ++j) { q4.scales[j] = q5.scales[j] = 0x91; q4.scales[j + 4] = q5.scales[j + 4] = 0x63; }
    for (int j = 8; j < 12; ++j) q4.scales[j] = q5.scales[j] = 0x31;
    for (int j = 0; j < 128; ++j) q4.qs[j] = q5.qs[j] = (uint8_t) (j * 13 + 7);
    for (int j = 0; j < 32; ++j) q5.qh[j] = (uint8_t) (j * 17 + 3);
    for (int j = 0; j < 16; ++j) q50.qs[j] = (uint8_t) (j * 19 + 5);
    for (int j = 0; j < 4; ++j) q50.qh[j] = (uint8_t) (j * 37 + 11);
    Native native{}; native.scale = (half) 0.3333f; native.raw_sum = (half) 121.0f;
    int sum = 0;
    for (int j = 0; j < 32; ++j) { native.qs[j] = (int8_t) (j * 5 - 73); sum += native.qs[j]; }
    std::array<float, 10> planar{};
    d::native_q8_1_planar_block<half>(&native, planar.data(), 1, 0, true);
    const float scale = planar[8], correction = planar[9];
    check(correction != (float) native.raw_sum, "raw-sum counterfactual discriminates format corrections");
    for (int b = 0; b < 8; ++b) {
        const Sub a = Dec<12>::get(reinterpret_cast<uint8_t*>(&q4), b, sizeof(q4));
        const Sub c = Dec<13>::get(reinterpret_cast<uint8_t*>(&q5), b, sizeof(q5));
        const int sc = b < 4 ? 17 : 33, mn = b < 4 ? 35 : 19;
        int dot4 = 0, dot5 = 0;
        double expected4 = 0, expected5 = 0;
        for (int j = 0; j < 32; ++j) {
            const int low = (q4.qs[32 * (b / 2) + j] >> (4 * (b % 2))) & 15;
            const int high = low | (((q5.qh[j] >> b) & 1) << 4);
            check(code(a, j) == low && code(c, j) == high,
                  "actual Q4_K/Q5_K decoder preserves raw unsigned weight codes");
            dot4 += code(a, j) * native.qs[j]; dot5 += code(c, j) * native.qs[j];
            expected4 += ((float) q4.dm[0] * sc * low - (float) q4.dm[1] * mn) * scale * native.qs[j];
            expected5 += ((float) q5.dm[0] * sc * high - (float) q5.dm[1] * mn) * scale * native.qs[j];
        }
        check(a.s0 == (float) q4.dm[0] * sc && a.m == (float) q4.dm[1] * mn &&
              c.s0 == a.s0 && c.m == a.m, "actual packed six-bit scales/mins decode for all eight subblocks");
        check(std::fabs((a.s0 * scale) * dot4 - a.m * correction - expected4) < 0.01 &&
              std::fabs((c.s0 * scale) * dot5 - c.m * correction - expected5) < 0.01,
              "Q4_K/Q5_K affine dot requires d*integer-code-sum correction");
    }
    const Sub centered = Dec<6>::get(reinterpret_cast<uint8_t*>(&q50), 0, sizeof(q50));
    int raw_dot = 0, centered_dot = 0;
    for (int j = 0; j < 32; ++j) {
        const int raw = ((q50.qs[j % 16] >> (4 * (j / 16))) & 15) | (((q50.qh[j / 8] >> (j % 8)) & 1) << 4);
        check(code(centered, j) == raw - 16, "actual Q5_0 decoder centers once, never twice");
        raw_dot += raw * native.qs[j]; centered_dot += code(centered, j) * native.qs[j];
    }
    check(centered_dot == raw_dot - 16 * sum,
          "centered Q5_0 equals raw-code integer correction, no raw-sum substitution");
    check(std::fabs(centered_dot * scale - (raw_dot * scale - 16 * (float) native.raw_sum)) > 1,
          "native raw-half-sum Q5_0 expression is not misrepresented as centered-XMX parity");
}
'''

DISPATCH = r'''
namespace sycl { struct queue {}; }
std::vector<std::string> calls;
void call(const char* s) { calls.emplace_back(s); }
bool native_on = false, capable = true, batch_rec_ = true;
int dense_xmx_min = 7, shared_min = 7, mmq_min = 7;
alignas(4) uint8_t buf[131072];
uint8_t* dq_ = buf; uint8_t* xq_ = buf + 8192;
void* cs = buf; int32_t dbounds_[64]{}, dids_[16]{};
int batch_dense_xmx_min() { return dense_xmx_min; }
int batch_dense_mmq_min() { return mmq_min; }
bool batch_dense_gemv_narrow(int, int64_t, int64_t) { return false; }
struct WeightRef { int native_type; const void* native_data; };
namespace strata::core {
struct Runtime {
    static Runtime& get() { static Runtime r; return r; }
    sycl::queue& stream(void*) { static sycl::queue q; return q; }
};
}
namespace strata::prefill::mmq {
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*) { call("old_quant"); }
struct Product {
    const void* w; int type; int64_t w_rows, w_cols, n; const void* xq; const int32_t* bounds; const int32_t* ids;
    int64_t total_rows, max_rows; float* dst; int64_t ld_dst;
};
struct Context { void run(Product&, void*) { call("old_mmq"); } };
}
namespace strata::kernels {
bool iq_mmq_decode_native_q8_1() { call("setting"); return native_on; }
// ACTUAL_TYPE_SUMS
bool iq_mmq_type_ok(int t) { return d::block_cols(t) > 0; }
bool iq_mmq_decode_ok(sycl::queue&, int t, int64_t n, int64_t k, int64_t r) {
    return capable && d::shape_ok(t, n, k, r);
}
void iq_mmq_decode_from_q8_1(sycl::queue&, const void* in, void* out, int64_t k, int64_t r, bool) {
    check(d::native_q8_1_storage_ok(in, out, k, r), "real consumer submits nonoverlapping adapter storage");
    call("adapter");
}
void iq_mmq_decode(sycl::queue&, int, const void*, int64_t, int64_t, const void*, int64_t, float*, int64_t) {
    call("decode");
}
void iq_mmq_quantize(sycl::queue&, const float*, const int32_t*, void*, int64_t, int64_t, int64_t, bool, bool) {
    call("old_quant");
}
void iq_mmq_swiglu_quantize(sycl::queue&, const float*, const float*, void*, int64_t, int64_t, bool) {
    call("old_swiglu");
}
void native_quantize_q8_1(const float*, void*, int, int, void*) { call("native_quant"); }
void native_swiglu_q8_1(const float*, const float*, void*, int, int, void*) { call("native_swiglu"); }
struct NativeSharedWeights {
    void* q8_1; const void* gate_data; const void* up_data; const void* down_data; int gate_type, up_type, down_type;
};
int shared_expert_xmx_min() { return shared_min; }
sycl::queue& queue_for(void* p) { return strata::core::Runtime::get().stream(p); }
bool shared_gate_tail(sycl::queue&, int, const float*, const uint16_t*, const uint16_t*, float*, float*, int64_t,
                      void*, bool, bool) { call("tail"); return true; }
// ACTUAL_SHARED_BRANCH
}
bool run_dense(int64_t rows, int type = 12) {
    // ACTUAL_DENSE_LAMBDA
    WeightRef w{type, buf};
    return dense(&w, reinterpret_cast<float*>(buf + 32768), reinterpret_cast<float*>(buf + 49152), 256, 256, rows);
}
void trace(std::initializer_list<const char*> want, const char* what) {
    std::vector<std::string> expected;
    for (const char* s : want) expected.emplace_back(s);
    check(calls == expected, what); calls.clear();
}
void dispatch_contract() {
    native_on = false;
    check(run_dense(7), "dense decode off branch fires");
    trace({"setting", "old_quant", "decode"}, "dense OFF keeps exact old quantize/decode sequence");
    native_on = true;
    check(run_dense(7), "dense decode on branch fires");
    trace({"setting", "adapter", "decode"}, "dense ON reuses prepared xq_, never requantizes");
    dense_xmx_min = 0;
    check(run_dense(7), "disabled dense XMX retains established MMQ");
    trace({"old_quant", "old_mmq"}, "new switch never enables dense XMX or changes prompt-style MMQ");
    dense_xmx_min = 7;
    check(!run_dense(6), "dense row floor unchanged"); trace({}, "below floor never consults native switch");
    capable = false;
    check(run_dense(7), "unsupported decode device retains old MMQ");
    trace({"old_quant", "old_mmq"}, "decode capability rejection never adapts operands"); capable = true;
    batch_rec_ = false;
    check(!run_dense(7), "non-batch dense path untouched");
    trace({}, "non-batch never consults native switch"); batch_rec_ = true;
    dense_xmx_min = 1;
    for (int rows : {1, 16}) {
        check(run_dense(rows), "dense adapter supports decode endpoints one and sixteen");
        trace({"setting", "adapter", "decode"}, "decode endpoint keeps native adapter order");
    }
    check(run_dense(17), "seventeen rows uses old MMQ, not decode");
    trace({"old_quant", "old_mmq"}, "decode window boundary respected");
    dense_xmx_min = 7;
    using namespace strata::kernels;
    NativeSharedWeights nw{buf + 8192, buf, buf, buf, 12, 13, 6};
    auto shared = [&](int rows, const void* xq, int64_t n = 512, int64_t ff = 256, bool native_input = false) {
        return run_shared(rows, reinterpret_cast<float*>(buf + 32768), nullptr, nw, nullptr,
                          reinterpret_cast<float*>(buf + 16384), reinterpret_cast<float*>(buf + 24576),
                          reinterpret_cast<float*>(buf + 57344), reinterpret_cast<float*>(buf + 49152), n, ff, cs,
                          false, false, xq, native_input);
    };
    native_on = false;
    check(shared(7, buf + 4096), "shared off branch fires");
    trace({"setting", "old_quant", "decode", "decode", "old_swiglu", "decode", "tail"},
          "shared OFF preserves exact old launch order");
    check(shared(7, buf + 4096, 512, 256, false), "shared OFF accepts existing canonical expert input");
    trace({"setting", "old_quant", "decode", "decode", "old_swiglu", "decode", "tail"},
          "provenance does not change old OFF behavior");
    native_on = true;
    check(shared(7, buf + 4096, 512, 256, false), "shared ON rejects non-native quantization provenance");
    trace({"setting", "native_quant", "adapter", "decode", "decode", "native_swiglu", "adapter", "decode", "tail"},
          "canonical expert per-256 image cannot masquerade as native32 codes");
    check(shared(7, buf + 4096), "shared ON unknown provenance uses conservative default");
    trace({"setting", "native_quant", "adapter", "decode", "decode", "native_swiglu", "adapter", "decode", "tail"},
          "same layout with unspecified provenance cannot take reuse fastpath");
    check(shared(7, buf + 4096, 512, 256, true), "shared explicit native32 branch fires");
    trace({"setting", "adapter", "decode", "decode", "native_swiglu", "adapter", "decode", "tail"},
          "shared ON reuses caller x_q8_1, then native SwiGLU in retired out storage");
    check(shared(7, nullptr), "shared missing native input quantizes once");
    trace({"setting", "native_quant", "adapter", "decode", "decode", "native_swiglu", "adapter", "decode", "tail"},
          "shared ON missing input uses canonical quantizer with existing scratch");
    for (int rows : {0, 6, 17}) {
        check(!shared(rows, buf + 4096), "shared row boundary falls back");
        trace({"fallback"}, "shared boundary never consults switch");
    }
    shared_min = 0;
    check(!shared(7, buf + 4096), "native switch does not enable shared XMX");
    trace({"fallback"}, "shared default disabled remains disabled"); shared_min = 7;
    check(!shared(7, buf + 4096, 256, 256), "existing shared scratch-size guard retained");
    trace({"fallback"}, "undersized scratch takes established fallback");
    capable = false;
    check(!shared(7, buf + 4096), "shared capability gate retained");
    trace({"fallback"}, "shared no device capability does not adapt"); capable = true;
    nw.q8_1 = nullptr;
    check(!shared(7, buf + 4096), "shared null scratch gate retained");
    trace({"fallback"}, "null scratch takes established fallback"); nw.q8_1 = buf + 8192;
    shared_min = 1;
    for (int rows : {1, 16}) {
        check(shared(rows, nullptr), "shared decode endpoints supported");
        trace({"setting", "native_quant", "adapter", "decode", "decode", "native_swiglu", "adapter", "decode", "tail"},
              "shared endpoints preserve scratch ownership and launch order");
    }
    for (int type : {12, 13}) {
        check(iq_mmq_type_sums(type), "affine K formats request integer sums");
    }
    check(!iq_mmq_type_sums(6), "centered Q5_0 never requests raw-sum correction");
}
int main() {
    numeric_contract(); formats_contract(); dispatch_contract();
    std::printf("native_q8_1_contract: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
'''


class NativeQ81ContractTest(unittest.TestCase):
    def test_real_primitive_adapter_decoders_and_dispatch(self) -> None:
        source = (ROOT / "src/kernels/xe/iq_mmq.cpp").read_text()
        decoders = "\n".join(braced(source, f"template<> struct Dec<{t}>") + ";" for t in (6, 12, 13))
        decoders = braced(source, "inline void scale_min_k4") + "\n" + decoders
        verify = (ROOT / "src/core/verify.cpp").read_text()
        dense = braced(verify, "auto dense =") + ";"
        shared = (ROOT / "src/kernels/xe/shared_expert.cpp").read_text()
        begin = shared.index("    const int xmx_min =", shared.index("bool shared_expert_multi("))
        end = shared.index("    if (const int tile =", begin)
        signature = braced(shared, "bool shared_expert_multi(").split("{", 1)[0]
        branch = signature.replace("shared_expert_multi", "run_shared") + "{\n" + shared[begin:end]
        branch += '\ncall("fallback"); return false;\n}'
        dispatch = DISPATCH.replace("// ACTUAL_SHARED_BRANCH", branch).replace("// ACTUAL_DENSE_LAMBDA", dense)
        dispatch = dispatch.replace("// ACTUAL_TYPE_SUMS", braced(source, "bool iq_mmq_type_sums("))
        program = PRELUDE + decoders + NUMERIC + dispatch
        with tempfile.TemporaryDirectory(prefix="dmmq-native-contract-") as tmp:
            work = Path(tmp)
            (work / "sycl").mkdir()
            (work / "sycl/sycl.hpp").write_text(
                "#pragma once\n#include <cmath>\n"
                "namespace sycl { using std::fabs; using std::copysign; using std::round; }\n"
            )
            cpp = work / "contract.cpp"
            cpp.write_text(program)
            compiler = shlex.split(os.environ.get("CXX", "c++"))
            result = subprocess.run(
                compiler + ["-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror", "-I", str(work),
                            "-I", str(ROOT / "include"), "-I", str(ROOT / "src/kernels/xe"),
                            str(cpp), "-o", str(work / "contract")],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / "contract")], text=True, capture_output=True, check=False)
            print(result.stdout, end="")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("0 failures", result.stdout)

    def test_no_global_prompt_or_hc_quantization_switch(self) -> None:
        for name in ("src/kernels/xe/hc_mmq.cpp", "src/kernels/xe/native_mmvq.cpp", "src/prefill/moe_mmq.cpp"):
            source = (ROOT / name).read_text()
            self.assertNotIn("iq_mmq_decode_native_q8_1", source)
            self.assertNotIn("iq_mmq_decode_from_q8_1", source)
        source = (ROOT / "src/kernels/xe/iq_mmq.cpp").read_text()
        for marker in ("sycl::event iq_mmq_quantize(", "sycl::event iq_mmq_swiglu_quantize(",
                       "sycl::event iq_mmq_quantize_scatter(", "struct DecodeKernel"):
            body = braced(source, marker)
            self.assertNotIn("native_q8_1", body)
        self.assertIn('std::getenv("STRATA_DMMQ_NATIVE_Q8_1")', braced(source, "bool iq_mmq_decode_native_q8_1("))
        adapter = braced(source, "sycl::event iq_mmq_decode_from_q8_1(")
        self.assertIn("native_q8_1_storage_ok", adapter)
        self.assertIn("native_q8_1_planar_block<sycl::half>", adapter)
        self.assertNotIn("malloc", adapter)
        # The real XMX consumer still subtracts integer sums and adds K slices in the original order.
        kernel = braced(source, "struct DecodeKernel")
        self.assertIn("acc[t][i] -= d[u].m * xu[(int64_t) r * nb + kb]", kernel)
        self.assertIn("for (int j = 0; j < ks; ++j) sum += red[(j * 16 + r) * IN + c]", kernel)

    def test_dense_prepared_inputs_have_native32_producers(self) -> None:
        source = (ROOT / "src/core/verify.cpp").read_text()
        for producer in (
            "if (!mixed_q8) native_quantize_q8_1(xm, xq_, (int) N, n, cs)",
            "if (!qfuse) native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs)",
            "native_quantize_q8_1(attn32_ + tb * NH * HD, xq_, (int) (NH * HD), n, cs)",
            "qfuse ? (void*) xq_ : nullptr",
        ):
            self.assertIn(producer, source)
        # Fused alternatives actually store native32 codes/scales at the producer, not canonical expert q8_1.
        for name in ("fused_gr.cpp", "fused_gr_win.cpp", "hc_mmq.cpp", "verify_kernels.cpp"):
            producer = (ROOT / "src/kernels/xe" / name).read_text()
            self.assertIn("xe::q8_1_quant(", producer)
            self.assertIn("xe::q8_1_finite(amax / 127.0f)", producer)
        self.assertEqual(source.count("!dense("), 5, "new dense callers need a producer audit")

    def test_verifier_preserves_expert_quantization_provenance(self) -> None:
        source = (ROOT / "src/core/verify.cpp").read_text()
        begin = source.index("gate_later_[grp] = shared_expert_multi(")
        end = source.index(";", begin)
        self.assertIn(": nullptr, false)", source[begin:end])
        producer = braced((ROOT / "src/kernels/xe/iq_kernels.cpp").read_text(), "void quantize_q8_1_rows(")
        self.assertIn("if (n_cols % 256 == 0)", producer)
        self.assertIn("canon::q8_scale(amax)", producer)
        header = (ROOT / "include/strata/kernels/shared_expert.hpp").read_text()
        self.assertIn("bool x_q8_1_native = false", header)
        # Every existing source caller: only dmmq_parity attests native32; others stay conservative.
        parity = (ROOT / "src/kernels/dmmq_parity.cpp").read_text()
        self.assertIn("native_quantize_q8_1(dx, xq,", parity)
        self.assertIn("!gated, !gated, xq, true)", parity)
        for name in ("batch_window_ubench.cpp", "mmvq_slm_parity.cpp"):
            caller = (ROOT / "src/kernels" / name).read_text()
            self.assertIn("quantize_q8_1_rows(", caller)
            self.assertIn("shared_expert_multi(", caller)
        shared = (ROOT / "src/kernels/xe/shared_expert.cpp").read_text()
        self.assertIn("x_q8_1_native ? x_q8_1 : nullptr", shared)
        # Tiling must carry provenance to its children, not silently use the native default.
        tiled = braced(shared, "if (const int tile =")
        self.assertIn("x_q8_1_native);", tiled)

    def test_canonical_quantizers_still_own_codes_and_half_storage(self) -> None:
        source = (ROOT / "src/kernels/xe/native_mmvq.cpp").read_text()
        for marker in ("void native_quantize_q8_1(", "void native_swiglu_q8_1("):
            body = braced(source, marker)
            self.assertIn("xe::q8_1_finite(amax / 127.0f)", body)
            self.assertIn("xe::q8_1_quant(xi, d, amax)", body)
            self.assertIn("half2(half(d), half(xe::q8_1_finite(sum)))", body)
        self.assertIn("sizeof(Q81Block) == 36", source)
        self.assertIn("offsetof(Q81Block, qs) == 4", source)


if __name__ == "__main__":
    unittest.main()
