#pragma once
#include <functional>
#include <string>
#include <string_view>
#include <vector>
namespace cellerator::compiler::driver {
enum class discovery_source_v1 { explicit_override, environment, configured_resource, path, platform_default, unavailable };
struct clang_discovery_input_v1 {
    std::string explicit_root, environment_root, configured_root;
    std::vector<std::string> path_roots, platform_roots;
};
struct clang_toolchain_v1 {
    discovery_source_v1 source = discovery_source_v1::unavailable;
    std::string compiler, linker, resource_directory, target_triple, version_identity;
    explicit operator bool() const noexcept { return source != discovery_source_v1::unavailable; }
};
using executable_probe_v1 = std::function<bool(std::string_view)>;
clang_toolchain_v1 discover_host_clang_v1(const clang_discovery_input_v1&, const executable_probe_v1&);
}  // namespace cellerator::compiler::driver
