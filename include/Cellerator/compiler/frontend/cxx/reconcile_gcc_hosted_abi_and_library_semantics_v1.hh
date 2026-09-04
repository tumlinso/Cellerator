#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t gcc_hosted_abi_schema_version_v1 = 1;

enum class cxx_calling_convention_v1 : std::uint8_t {
    system_v_amd64 = 1,
    microsoft_x64,
    aarch64_aapcs,
    unknown,
};

enum class gcc_hosted_abi_status_v1 : std::uint8_t {
    compatible = 0,
    schema_mismatch,
    unsupported_host,
    target_mismatch,
    language_mismatch,
    standard_library_mismatch,
    abi_macro_mismatch,
    layout_mismatch,
    calling_convention_mismatch,
};

struct cxx_abi_observation_v1 {
    std::string target;
    std::uint64_t language_standard = 0;
    std::string standard_library;
    std::uint64_t standard_library_version = 0;
    std::uint64_t gxx_abi_version = 0;
    int glibcxx_cxx11_abi = -1;
    std::uint32_t pointer_bytes = 0;
    std::uint32_t pointer_alignment = 0;
    std::uint32_t long_double_bytes = 0;
    std::uint32_t long_double_alignment = 0;
    cxx_calling_convention_v1 calling_convention = cxx_calling_convention_v1::unknown;
};

struct gcc_hosted_abi_result_v1 {
    gcc_hosted_abi_status_v1 status = gcc_hosted_abi_status_v1::compatible;
    std::vector<std::string> diagnostics;
};

cxx_abi_observation_v1 observe_gcc_hosted_abi_v1() noexcept;

gcc_hosted_abi_status_v1 reconcile_gcc_hosted_abi_v1(
    std::uint32_t schema_version,
    const cxx_abi_observation_v1& clang_assumptions,
    const cxx_abi_observation_v1& gcc_observations,
    gcc_hosted_abi_result_v1* result) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
