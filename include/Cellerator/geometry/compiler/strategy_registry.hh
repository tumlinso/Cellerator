#pragma once

#include <Cellerator/geometry/compiler/strategy.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::compiler {

struct geometry_strategy_registry_v1 {
    const geometry_strategy_descriptor_v1 *strategies = nullptr;
    u32 strategy_count = 0u;
};

geometry_strategy_status validate_geometry_strategy_registry(
    geometry_strategy_registry_v1 registry) noexcept;

const geometry_strategy_descriptor_v1 *find_geometry_strategy(
    geometry_strategy_registry_v1 registry,
    u64 strategy_id,
    geometry_search_tier tier) noexcept;

geometry_strategy_status query_geometry_strategy_requirements(
    geometry_strategy_registry_v1 registry,
    const geometry_problem_v1 &problem,
    const geometry_search_policy_v1 &policy,
    geometry_strategy_requirements_v1 *requirements) noexcept;

geometry_strategy_status run_geometry_strategy(
    geometry_strategy_registry_v1 registry,
    const geometry_problem_v1 &problem,
    const geometry_search_policy_v1 &policy,
    geometry_strategy_workspace_v1 workspace,
    geometry_solution_buffers_v1 buffers,
    geometry_solution_v1 *solution) noexcept;

static_assert(std::is_trivially_copyable<geometry_strategy_registry_v1>::value,
    "geometry strategy registries must remain pointer-copyable");

} // namespace cellerator::geometry::compiler
