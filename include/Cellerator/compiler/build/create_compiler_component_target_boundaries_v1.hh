#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace Cellerator::compiler::build {

enum class compiler_component_v1 : std::uint8_t {
    core,
    diagnostics,
    ceir,
    profiles,
    frontend,
    planning,
    realization,
    backends,
    tooling,
};

struct component_edge_v1 {
    compiler_component_v1 consumer;
    compiler_component_v1 dependency;
};

inline constexpr std::array<std::string_view, 9> compiler_component_names_v1{{
    "CompilerCore", "CompilerDiagnostics", "CEIR", "CompilerProfiles",
    "CompilerFrontend", "CompilerPlanning", "CompilerRealization",
    "CompilerBackends", "CompilerTooling",
}};

inline constexpr std::array<component_edge_v1, 11> compiler_component_edges_v1{{
    {compiler_component_v1::diagnostics, compiler_component_v1::core},
    {compiler_component_v1::ceir, compiler_component_v1::core},
    {compiler_component_v1::ceir, compiler_component_v1::diagnostics},
    {compiler_component_v1::profiles, compiler_component_v1::ceir},
    {compiler_component_v1::frontend, compiler_component_v1::ceir},
    {compiler_component_v1::frontend, compiler_component_v1::diagnostics},
    {compiler_component_v1::planning, compiler_component_v1::ceir},
    {compiler_component_v1::planning, compiler_component_v1::profiles},
    {compiler_component_v1::realization, compiler_component_v1::planning},
    {compiler_component_v1::backends, compiler_component_v1::realization},
    {compiler_component_v1::tooling, compiler_component_v1::backends},
}};

[[nodiscard]] constexpr bool compiler_component_graph_is_acyclic_v1() noexcept {
    for (const auto edge : compiler_component_edges_v1) {
        if (static_cast<unsigned>(edge.dependency) >=
            static_cast<unsigned>(edge.consumer)) {
            return false;
        }
    }
    return true;
}

}  // namespace Cellerator::compiler::build
