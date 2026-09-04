#pragma once

#include <array>
#include <string_view>

namespace cellerator::compiler::acceptance::v1 {

inline constexpr std::array<std::string_view, 12> component_registry = {
    "source", "frontend", "sema", "profiles", "semantic-ir", "planning-ir",
    "realization-ir", "reflection", "passes", "lto", "tooling", "sdk"};

}  // namespace cellerator::compiler::acceptance::v1
