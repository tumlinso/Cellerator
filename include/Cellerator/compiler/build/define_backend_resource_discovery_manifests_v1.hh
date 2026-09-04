#pragma once

#include <array>
#include <string_view>

namespace Cellerator::compiler::build {

inline constexpr std::array<std::string_view, 8> backend_resource_keys_v1{{
    "host_cxx", "nvcc", "clang_cuda", "llvm_config", "nvptx", "ptxas",
    "linker", "resource_dir",
}};

[[nodiscard]] constexpr bool backend_resource_manifest_is_complete_v1() {
    for (const auto key : backend_resource_keys_v1) {
        if (key.empty()) return false;
    }
    return true;
}

}  // namespace Cellerator::compiler::build
