#pragma once
#include <functional>
#include <string>
#include <string_view>
#include <vector>
namespace cellerator::compiler::driver {
struct gcc_discovery_input_v1 { std::vector<std::string> roots; std::string target_triple, version_identity, libstdcxx_abi_mode; };
struct gcc_toolchain_v1 { std::string cxx, cc, linker, include_root, target_triple, version_identity, libstdcxx_abi_mode, diagnostic; explicit operator bool() const noexcept { return diagnostic.empty(); } };
using gcc_executable_probe_v1 = std::function<bool(std::string_view)>;
gcc_toolchain_v1 discover_host_gcc_v1(const gcc_discovery_input_v1&, const gcc_executable_probe_v1&);
}  // namespace cellerator::compiler::driver
