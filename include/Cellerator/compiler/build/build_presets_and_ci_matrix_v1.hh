#pragma once

#include <array>
#include <string_view>

namespace Cellerator::compiler::build {
inline constexpr std::array<std::string_view, 7> compiler_ci_presets_v1{{
    "host-clang", "host-gcc", "cuda-nvcc-sm70", "cuda-clang",
    "installed-consumer", "sanitizer", "language-server",
}};
inline constexpr std::array<std::string_view, 5> non_hardware_presets_v1{{
    "host-clang", "host-gcc", "installed-consumer", "sanitizer",
    "language-server",
}};
}  // namespace Cellerator::compiler::build
