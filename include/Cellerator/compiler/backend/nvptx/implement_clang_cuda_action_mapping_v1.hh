#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

enum class clang_cuda_action_kind_v1 : std::uint8_t {
    device_compile = 1u,
    host_compile,
    offload_bundle,
    link,
};

struct clang_cuda_toolchain_v1 {
    std::string clang_path;
    std::string bundler_path;
    std::string cuda_root;
    std::string libdevice_path;
};

struct clang_cuda_mapping_request_v1 {
    std::string source_path;
    std::string output_stem;
    std::uint32_t compute_major = 0u;
    std::uint32_t compute_minor = 0u;
    std::vector<std::string> include_paths;
    std::vector<std::string> libraries;
};

struct clang_cuda_action_v1 {
    clang_cuda_action_kind_v1 kind = clang_cuda_action_kind_v1::device_compile;
    std::string executable;
    std::vector<std::string> arguments;
    std::string output_path;
};

enum class clang_cuda_mapping_status_v1 : std::uint8_t {
    success = 0u,
    invalid_toolchain,
    invalid_request,
};

struct clang_cuda_action_plan_v1 {
    clang_cuda_mapping_status_v1 status = clang_cuda_mapping_status_v1::invalid_request;
    std::vector<clang_cuda_action_v1> actions;

    explicit operator bool() const noexcept {
        return status == clang_cuda_mapping_status_v1::success;
    }
};

[[nodiscard]] clang_cuda_action_plan_v1 map_clang_cuda_actions_v1(
    const clang_cuda_toolchain_v1& toolchain,
    const clang_cuda_mapping_request_v1& request);

}  // namespace Cellerator::compiler::backend::nvptx
