#pragma once

#include <cstdint>
#include <string_view>

namespace cellerator::compiler::api::v1 {

enum class component_v1 : std::uint8_t {
    ir = 0,
    profile,
    planning,
    realization,
    backend,
    diagnostics,
    runtime_execution,
    compiler,
    count,
};

struct component_contract_v1 {
    component_v1 component = component_v1::ir;
    std::string_view link_name;
    std::string_view abi_owner;
    std::uint32_t dependency_mask = 0;
};

inline constexpr std::uint32_t component_count_v1 =
    static_cast<std::uint32_t>(component_v1::count);
[[nodiscard]] component_contract_v1 component_contract(component_v1 component) noexcept;
[[nodiscard]] bool component_link_graph_is_acyclic_v1() noexcept;

}  // namespace cellerator::compiler::api::v1
