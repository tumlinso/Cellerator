#pragma once

#include <array>
#include <string_view>

namespace Cellerator::compiler::build {
inline constexpr std::array<std::string_view, 5> host_smoke_components_v1{{
    "ceir", "profile", "source", "diagnostics", "daemon_protocol",
}};
inline constexpr bool host_smokes_link_cuda_v1 = false;
}  // namespace Cellerator::compiler::build
