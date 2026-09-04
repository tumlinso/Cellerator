#pragma once

#include <Cellerator/compiler/composition/basis_v1.hh>
#include <Cellerator/compiler/composition/grammar_v1.hh>
#include <Cellerator/compiler/discovery/atom_v1.hh>
#include <Cellerator/compiler/discovery/discovery_v1.hh>
#include <Cellerator/compiler/ir/planning/planning_ir_v1.hh>
#include <Cellerator/compiler/planning/adapt_candidate_catalog_v3_providers_v1.hh>
#include <Cellerator/compiler/planning/adapt_external_global_cost_exchange_v1.hh>
#include <Cellerator/compiler/planning/expose_complete_planning_reports_v1.hh>
#include <Cellerator/compiler/planning/implement_candidate_inclusion_exclusion_and_forcing_v1.hh>
#include <Cellerator/compiler/planning/implement_custom_candidate_registration_v1.hh>
#include <Cellerator/compiler/planning/implement_planner_portfolio_dispatch_v1.hh>
#include <Cellerator/compiler/planning/implement_planning_cache_and_invalidation_v1.hh>
#include <Cellerator/compiler/program/ruleset_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::planning {

inline constexpr std::uint32_t public_planning_compiler_interface_version_v1 = 1u;

enum public_planning_capability_v1 : std::uint32_t {
    public_planning_provider_v1 = 1u << 0u,
    public_planning_planner_v1 = 1u << 1u,
    public_planning_cache_v1 = 1u << 2u,
    public_planning_report_v1 = 1u << 3u,
    public_planning_custom_candidate_v1 = 1u << 4u,
    public_planning_external_cost_v1 = 1u << 5u,
    public_planning_force_control_v1 = 1u << 6u,
};

inline constexpr std::uint32_t public_planning_capabilities_v1 =
    public_planning_provider_v1 | public_planning_planner_v1 |
    public_planning_cache_v1 | public_planning_report_v1 |
    public_planning_custom_candidate_v1 | public_planning_external_cost_v1 |
    public_planning_force_control_v1;

struct public_planning_dependency_versions_v1 {
    std::uint32_t planning_ir =
        cellerator::compiler::ir::planning::v1::planning_ir_contract_version_v1;
    std::uint32_t discovery = discovery::discovery_contract_version_v1;
    std::uint32_t atom = discovery::atom_compiler_contract_version_v1;
    std::uint32_t grammar = composition::grammar_contract_version_v1;
    std::uint32_t basis = composition::basis_contract_version_v1;
    std::uint32_t ruleset = program::ruleset_contract_version_v1;
};

struct public_planning_compiler_interface_v1 {
    std::uint32_t interface_version = public_planning_compiler_interface_version_v1;
    std::uint32_t capabilities = public_planning_capabilities_v1;
    public_planning_dependency_versions_v1 dependencies{};
};

enum class public_planning_interface_status_v1 : std::uint8_t {
    ready = 0u,
    unsupported_interface,
    unsupported_planning_ir,
    unsupported_discovery,
    unsupported_atom,
    unsupported_grammar,
    unsupported_basis,
    unsupported_ruleset,
    incomplete_capabilities,
};

// This is the narrow, versioned handoff consumed by Realization IR. The types
// above remain ordinary public C++ contracts; freezing validates their shared
// compiler dependencies without installing callbacks or owning runtime state.
[[nodiscard]] public_planning_interface_status_v1
freeze_public_planning_compiler_interface_v1(
    const public_planning_compiler_interface_v1& interface) noexcept;

}  // namespace Cellerator::compiler::planning
