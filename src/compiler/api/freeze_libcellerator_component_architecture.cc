#include <Cellerator/compiler/api/freeze_libcellerator_component_architecture_v1.hh>

#include <array>

namespace cellerator::compiler::api::v1 {
namespace {
constexpr std::uint32_t bit(component_v1 value) {
    return 1U << static_cast<unsigned>(value);
}
constexpr std::array<component_contract_v1, component_count_v1> contracts{{
    {component_v1::ir, "CelleratorIR", "cellerator::compiler::ir", 0},
    {component_v1::profile, "CelleratorProfile", "cellerator::compiler::profile", bit(component_v1::ir)},
    {component_v1::planning, "CelleratorPlanning", "cellerator::compiler::planning", bit(component_v1::ir) | bit(component_v1::profile)},
    {component_v1::realization, "CelleratorRealization", "cellerator::compiler::realization", bit(component_v1::ir) | bit(component_v1::planning)},
    {component_v1::backend, "CelleratorBackend", "cellerator::compiler::backend", bit(component_v1::realization)},
    {component_v1::diagnostics, "CelleratorDiagnostics", "cellerator::compiler::diagnostics", bit(component_v1::ir)},
    {component_v1::runtime_execution, "CelleratorRuntime", "Cellerator::execution", 0},
    {component_v1::compiler, "CelleratorCompiler", "cellerator::compiler", bit(component_v1::backend) | bit(component_v1::diagnostics) | bit(component_v1::runtime_execution)},
}};
}

component_contract_v1 component_contract(component_v1 component) noexcept {
    const auto index = static_cast<std::size_t>(component);
    return index < contracts.size() ? contracts[index] : component_contract_v1{};
}

bool component_link_graph_is_acyclic_v1() noexcept {
    for (std::size_t index = 0; index < contracts.size(); ++index) {
        if ((contracts[index].dependency_mask & ~((1U << index) - 1U)) != 0) return false;
    }
    return true;
}

}  // namespace cellerator::compiler::api::v1
