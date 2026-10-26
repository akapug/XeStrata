// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/kernel_spill_audit.cpp - builds every kernel linked into it (strata_kernels, whole) for
// the device, one kernel id at a time, so that the GPU compiler's own reports cover all of them: run it with
// IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir> and each kernel's ISA header there gives its register file
// (numGRF) and its spill size.  Prints each kernel id's name and whether it built.
//
//     build/kernel_spill_audit
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <exception>

int main() try {
    auto& q = strata::core::Runtime::get().compute();
    int built = 0, failed = 0;
    for (const sycl::kernel_id& id : sycl::get_kernel_ids()) {
        try {
            (void) sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()}, {id});
            ++built;
            std::printf("built %s\n", id.get_name());
        } catch (const sycl::exception& e) {
            ++failed;
            std::printf("FAILED %s: %s\n", id.get_name(), e.what());
        }
    }
    std::printf("kernel_spill_audit: %d kernels built, %d failed\n", built, failed);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "kernel_spill_audit: %s\n", e.what());
    return 1;
}
