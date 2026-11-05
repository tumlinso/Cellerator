#include <Cellerator/geometry/planning/strategy_registry.hh>

#include <cassert>
#include <cstdint>

namespace {

namespace planning = cellerator::geometry::planning;

planning::geometry_strategy_status query_small(
    const planning::geometry_problem_v1 &problem,
    const planning::geometry_search_policy_v1 &,
    planning::geometry_strategy_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr || problem.work_window.member_count == 0u)
        return planning::geometry_strategy_status::invalid_argument;
    requirements->workspace_bytes = 32u;
    requirements->workspace_alignment = 16u;
    requirements->work_item_capacity = problem.work_window.member_count;
    requirements->component_capacity = 1u;
    requirements->logical_edge_capacity =
        problem.primary_relation.logical_edge_count;
    return planning::geometry_strategy_status::ok;
}

planning::geometry_strategy_status execute_small(
    const planning::geometry_problem_v1 &,
    const planning::geometry_search_policy_v1 &,
    planning::geometry_strategy_workspace_v1,
    planning::geometry_solution_buffers_v1,
    planning::geometry_solution_v1 *solution) noexcept {
    if (solution == nullptr)
        return planning::geometry_strategy_status::invalid_argument;
    *solution = {};
    return planning::geometry_strategy_status::ok;
}

planning::geometry_strategy_status query_external(
    const planning::geometry_problem_v1 &,
    const planning::geometry_search_policy_v1 &,
    planning::geometry_strategy_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr)
        return planning::geometry_strategy_status::invalid_argument;
    *requirements = {};
    return planning::geometry_strategy_status::ok;
}

planning::geometry_problem_v1 make_problem(const std::uint32_t *members) noexcept {
    planning::geometry_problem_v1 problem{};
    problem.primary_relation.logical_edge_count = 3u;
    problem.work_window.member_count = 2u;
    problem.work_window.members = members;
    return problem;
}

void test_resolves_source_linked_strategies_by_tier() {
    const planning::geometry_strategy_descriptor_v1 strategies[] = {
        {planning::geometry_strategy_schema_version,
            planning::geometry_search_tier_instant
                | planning::geometry_search_tier_bounded,
            11u, query_small, execute_small},
        {planning::geometry_strategy_schema_version,
            planning::geometry_search_tier_external,
            22u, query_external, execute_small}
    };
    const planning::geometry_strategy_registry_v1 registry{strategies, 2u};
    assert(planning::validate_geometry_strategy_registry(registry)
        == planning::geometry_strategy_status::ok);
    assert(planning::find_geometry_strategy(
        registry, 11u, planning::geometry_search_tier::instant));
    assert(planning::find_geometry_strategy(
        registry, 11u, planning::geometry_search_tier::external) == nullptr);
    assert(planning::find_geometry_strategy(
        registry, 22u, planning::geometry_search_tier::external));
}

void test_queries_before_running_and_checks_caller_buffers() {
    const planning::geometry_strategy_descriptor_v1 strategy = {
        planning::geometry_strategy_schema_version,
        planning::geometry_search_tier_instant,
        11u,
        query_small,
        execute_small
    };
    const planning::geometry_strategy_registry_v1 registry{&strategy, 1u};
    const std::uint32_t members[] = {0u, 1u};
    const planning::geometry_problem_v1 problem = make_problem(members);
    planning::geometry_search_policy_v1 policy{};
    policy.strategy_id = 11u;

    planning::geometry_strategy_requirements_v1 requirements{};
    assert(planning::query_geometry_strategy_requirements(
        registry, problem, policy, &requirements)
        == planning::geometry_strategy_status::ok);
    assert(requirements.workspace_bytes == 32u);
    assert(requirements.work_item_capacity == 2u);
    assert(requirements.logical_edge_capacity == 3u);

    alignas(16) std::uint8_t workspace_bytes[32]{};
    std::uint32_t forward[2]{};
    std::uint32_t inverse[2]{};
    cellerator::geometry::semantic_component_v1 component{};
    std::uint64_t edges[3]{};
    planning::geometry_solution_buffers_v1 buffers{};
    buffers.execution_to_window = forward;
    buffers.window_to_execution = inverse;
    buffers.work_item_capacity = 2u;
    buffers.components = &component;
    buffers.component_capacity = 1u;
    buffers.logical_edge_ids = edges;
    buffers.logical_edge_capacity = 3u;
    planning::geometry_solution_v1 solution{};
    assert(planning::run_geometry_strategy(registry, problem, policy,
        {workspace_bytes, 32u}, buffers, &solution)
        == planning::geometry_strategy_status::ok);
    assert(solution.strategy_id == 11u);
    assert(solution.tier == planning::geometry_search_tier::instant);

    buffers.logical_edge_capacity = 2u;
    assert(planning::run_geometry_strategy(registry, problem, policy,
        {workspace_bytes, 32u}, buffers, &solution)
        == planning::geometry_strategy_status::insufficient_output_capacity);
    buffers.logical_edge_capacity = 3u;
    assert(planning::run_geometry_strategy(registry, problem, policy,
        {workspace_bytes, 31u}, buffers, &solution)
        == planning::geometry_strategy_status::insufficient_workspace);
}

void test_registry_rejects_duplicates_and_invalid_tiers() {
    planning::geometry_strategy_descriptor_v1 strategies[] = {
        {planning::geometry_strategy_schema_version,
            planning::geometry_search_tier_instant,
            11u, query_small, execute_small},
        {planning::geometry_strategy_schema_version,
            planning::geometry_search_tier_bounded,
            11u, query_small, execute_small}
    };
    planning::geometry_strategy_registry_v1 registry{strategies, 2u};
    assert(planning::validate_geometry_strategy_registry(registry)
        == planning::geometry_strategy_status::invalid_registry);

    strategies[1].strategy_id = 12u;
    strategies[1].supported_tiers = 1u << 12u;
    assert(planning::validate_geometry_strategy_registry(registry)
        == planning::geometry_strategy_status::invalid_registry);
}

} // namespace

int main() {
    test_resolves_source_linked_strategies_by_tier();
    test_queries_before_running_and_checks_caller_buffers();
    test_registry_rejects_duplicates_and_invalid_tiers();
    return 0;
}
