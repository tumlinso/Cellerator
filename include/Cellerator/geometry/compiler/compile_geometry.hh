#pragma once

#include <Cellerator/geometry/compiler/strategy_registry.hh>

namespace cellerator::geometry::compiler {

// The built-in identity strategy is the allocation-free permissive baseline.
// It preserves the selected work-window order and emits one exact
// unstructured relation component.
const geometry_strategy_descriptor_v1 &identity_geometry_strategy() noexcept;

// Compile through a source-linked strategy and independently validate its
// portable work layout and exact logical-edge cover before publishing it.
geometry_strategy_status compile_geometry(
    geometry_strategy_registry_v1 registry,
    const geometry_problem_v1 &problem,
    const geometry_search_policy_v1 &policy,
    geometry_strategy_workspace_v1 strategy_workspace,
    geometry_solution_buffers_v1 buffers,
    relation_cover_validation_workspace validation_workspace,
    geometry_solution_v1 *solution) noexcept;

} // namespace cellerator::geometry::compiler
