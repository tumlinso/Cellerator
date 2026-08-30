#include <Cellerator/geometry/admissibility.hh>
#include <Cellerator/geometry/compiler/strategy_registry.hh>

namespace cellerator::geometry::compiler {
namespace {

inline constexpr u64 identity_geometry_strategy_id =
    0x4944454e54495459ull;

geometry_strategy_status query_identity_requirements(
    const geometry_problem_v1 &problem,
    const geometry_search_policy_v1 &policy,
    geometry_strategy_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr
        || policy.tier != geometry_search_tier::instant
        || (problem.admissibility != nullptr
            && !admissibility_is_permissive(*problem.admissibility)))
        return geometry_strategy_status::unsupported_policy;
    geometry_strategy_requirements_v1 result{};
    result.work_item_capacity = problem.work_window.member_count;
    result.component_capacity =
        problem.primary_relation.logical_edge_count == 0u ? 0u : 1u;
    result.logical_edge_capacity =
        problem.primary_relation.logical_edge_count;
    *requirements = result;
    return geometry_strategy_status::ok;
}

geometry_strategy_status execute_identity_strategy(
    const geometry_problem_v1 &problem,
    const geometry_search_policy_v1 &policy,
    geometry_strategy_workspace_v1,
    geometry_solution_buffers_v1 buffers,
    geometry_solution_v1 *solution) noexcept {
    if (solution == nullptr || policy.tier != geometry_search_tier::instant)
        return geometry_strategy_status::invalid_argument;

    const u32 work_count = problem.work_window.member_count;
    for (u32 index = 0u; index < work_count; ++index)
        buffers.execution_to_window[index] = index;
    work_layout_view_v1 work_layout{};
    if (!build_work_layout(problem.work_window, buffers.execution_to_window,
            work_count, buffers.window_to_execution, work_count, &work_layout))
        return geometry_strategy_status::strategy_failed;

    const u64 logical_edge_count =
        problem.primary_relation.logical_edge_count;
    for (u64 edge = 0u; edge < logical_edge_count; ++edge)
        buffers.logical_edge_ids[edge] = edge;
    if (logical_edge_count != 0u) {
        semantic_component_v1 component{};
        component.component_id = 1u;
        component.kind = semantic_component_kind::unstructured;
        component.logical_edge_count = logical_edge_count;
        buffers.components[0] = component;
    }

    relation_cover_view_v1 relation_cover{};
    relation_cover.structure = problem.primary_relation.structure;
    relation_cover.structure_epoch = problem.primary_relation.epoch;
    relation_cover.source_axis = problem.primary_relation.source_axis;
    relation_cover.destination_axis =
        problem.primary_relation.destination_axis;
    relation_cover.logical_edge_count = logical_edge_count;
    relation_cover.component_count = logical_edge_count == 0u ? 0u : 1u;
    relation_cover.components =
        logical_edge_count == 0u ? nullptr : buffers.components;
    relation_cover.logical_edge_ids =
        logical_edge_count == 0u ? nullptr : buffers.logical_edge_ids;

    geometry_solution_v1 result{};
    result.work_layout = work_layout;
    result.relation_cover = relation_cover;
    *solution = result;
    return geometry_strategy_status::ok;
}

} // namespace

const geometry_strategy_descriptor_v1 &identity_geometry_strategy() noexcept {
    static constexpr geometry_strategy_descriptor_v1 strategy = {
        geometry_strategy_schema_version,
        geometry_search_tier_instant,
        identity_geometry_strategy_id,
        query_identity_requirements,
        execute_identity_strategy
    };
    return strategy;
}

} // namespace cellerator::geometry::compiler
