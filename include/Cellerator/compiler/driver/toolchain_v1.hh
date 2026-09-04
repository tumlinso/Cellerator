#pragma once

// Stable Part One toolchain discovery and identity contract. All accelerator
// routes remain optional so host-only and ordinary C++ compilation stay valid.
#include <Cellerator/compiler/driver/define_toolchain_override_precedence_v1.hh>
#include <Cellerator/compiler/driver/discover_clang_cuda_and_llvm_nvptx_toolchains_v1.hh>
#include <Cellerator/compiler/driver/discover_host_clang_toolchains_v1.hh>
#include <Cellerator/compiler/driver/discover_host_gcc_toolchains_v1.hh>
#include <Cellerator/compiler/driver/discover_nvcc_toolchains_v1.hh>
#include <Cellerator/compiler/driver/fingerprint_toolchains_for_artifacts_and_resumption_v1.hh>
