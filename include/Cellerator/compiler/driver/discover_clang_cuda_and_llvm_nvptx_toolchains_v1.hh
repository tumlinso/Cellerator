#pragma once
#include <functional>
#include <string>
#include <string_view>
namespace cellerator::compiler::driver {
struct clang_cuda_discovery_input_v1 { std::string clang_root, llvm_root, cuda_root, target; };
struct clang_cuda_toolchain_v1 { bool host_available = false, cuda_route_available = false, nvptx_route_available = false; std::string clang_cxx, llvm_config, cuda_resource, libdevice, ptxas, diagnostic; };
using clang_cuda_probe_v1 = std::function<bool(std::string_view)>;
clang_cuda_toolchain_v1 discover_clang_cuda_and_nvptx_v1(const clang_cuda_discovery_input_v1&, const clang_cuda_probe_v1&);
}  // namespace cellerator::compiler::driver
