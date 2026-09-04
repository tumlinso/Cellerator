#pragma once
#include <functional>
#include <string>
#include <string_view>
#include <vector>
namespace cellerator::compiler::driver {
struct cuda_installation_v1 { std::string root, version_identity; int minimum_host_major = 0, maximum_host_major = 0; std::vector<int> architectures; };
struct nvcc_discovery_input_v1 { std::string explicit_root; std::vector<cuda_installation_v1> installations; int host_compiler_major = 0; int requested_architecture = 0; };
struct nvcc_toolchain_v1 { std::string nvcc, toolkit_root, ptxas, nvlink, fatbinary, version_identity, diagnostic; explicit operator bool() const noexcept { return diagnostic.empty(); } };
using nvcc_executable_probe_v1 = std::function<bool(std::string_view)>;
nvcc_toolchain_v1 discover_nvcc_v1(const nvcc_discovery_input_v1&, const nvcc_executable_probe_v1&);
}  // namespace cellerator::compiler::driver
