#include <Cellerator/geometry/compiler/strategy_registry.hh>

#include <cassert>
#include <cstdint>

namespace {

namespace comp = cellerator::geometry::compiler;

comp::geometry_strategy_status query_small(
    const comp::geometry_problem_v1 &problem,
    const comp::geometry_search_policy_v1 &,
    comp::geometry_strategy_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr || problem.work_window.member_count == 0u)
        return comp::geometry_strategy_status::invalid_argument;
    requirements->workspace_bytes = 32u;
    requirements->workspace_alignment = 16u;
    requirements->work_item_capacity = problem.work_window.member_count;
    requirements->component_capacity = 1u;
    requirements->logical_edge_capacity =
        problem.primary_relation.logical_edge_count;
    return comp::geometry_strategy_status::ok;
}

comp::geometry_strategy_status execute_small(
    const comp::geometry_problem_v1 &,
    const comp::geometry_search_policy_v1 &,
    comp::geometry_strategy_workspace_v1,
    comp::geometry_solution_buffers_v1,
    comp::geometry_solution_v1 *solution) noexcept {
    if (solution == nullptr)
        return comp::geometry_strategy_status::invalid_argument;
    *solution = {};
    return comp::geometry_strategy_status::ok;
}

comp::geometry_strategy_status query_external(
    const comp::geometry_problem_v1 &,
    const comp::geometry_search_policy_v1 &,
    comp::geometry_strategy_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr)
        return comp::geometry_strategy_status::invalid_argument;
    *requirements = {};
    return comp::geometry_strategy_status::ok;
}

comp::geometry_problem_v1 make_problem(const std::uint32_t *members) noexcept {
    comp::geometry_problem_v1 problem{};
    problem.primary_relation.logical_edge_count = 3u;
    problem.work_window.member_count = 2u;
    problem.work_window.members = members;
    return problem;
}

void test_resolves_source_linked_strategies_by_tier() {
    const comp::geometry_strategy_descriptor_v1 strategies[] = {
        {comp::geometry_strategy_schema_version,
            comp::geometry_search_tier_instant
                | comp::geometry_search_tier_bounded,
            11u, query_small, execute_small},
        {comp::geometry_strategy_schema_version,
            comp::geometry_search_tier_external,
            22u, query_external, execute_small}
    };
    const comp::geometry_strategy_registry_v1 registry{strategies, 2u};
    assert(comp::validate_geometry_strategy_registry(registry)
        == comp::geometry_strategy_status::ok);
    assert(comp::find_geometry_strategy(
        registry, 11u, comp::geometry_search_tier::instant));
    assert(comp::find_geometry_strategy(
        registry, 11u, comp::geometry_search_tier::external) == nullptr);
    assert(comp::find_geometry_strategy(
        registry, 22u, comp::geometry_search_tier::external));
}

void test_queries_before_running_and_checks_caller_buffers() {
    const comp::geometry_strategy_descriptor_v1 strategy = {
        comp::geometry_strategy_schema_version,
        comp::geometry_search_tier_instant,
        11u,
        query_small,
        execute_small
    };
    const comp::geometry_strategy_registry_v1 registry{&strategy, 1u};
    const std::uint32_t members[] = {0u, 1u};
    const comp::geometry_problem_v1 problem = make_problem(members);
    comp::geometry_search_policy_v1 policy{};
    policy.strategy_id = 11u;

    comp::geometry_strategy_requirements_v1 requirements{};
    assert(comp::query_geometry_strategy_requirements(
        registry, problem, policy, &requirements)
        == comp::geometry_strategy_status::ok);
    assert(requirements.workspace_bytes == 32u);
    assert(requirements.work_item_capacity == 2u);
    assert(requirements.logical_edge_capacity == 3u);

    alignas(16) std::uint8_t workspace_bytes[32]{};
    std::uint32_t forward[2]{};
    std::uint32_t inverse[2]{};
    cellerator::geometry::semantic_component_v1 component{};
    std::uint64_t edges[3]{};
    comp::geometry_solution_buffers_v1 buffers{};
    buffers.execution_to_window = forward;
    buffers.window_to_execution = inverse;
    buffers.work_item_capacity = 2u;
    buffers.components = &component;
    buffers.component_capacity = 1u;
    buffers.logical_edge_ids = edges;
    buffers.logical_edge_capacity = 3u;
    comp::geometry_solution_v1 solution{};
    assert(comp::run_geometry_strategy(registry, problem, policy,
        {workspace_bytes, 32u}, buffers, &solution)
        == comp::geometry_strategy_status::ok);
    assert(solution.strategy_id == 11u);
    assert(solution.tier == comp::geometry_search_tier::instant);

    buffers.logical_edge_capacity = 2u;
    assert(comp::run_geometry_strategy(registry, problem, policy,
        {workspace_bytes, 32u}, buffers, &solution)
        == comp::geometry_strategy_status::insufficient_output_capacity);
    buffers.logical_edge_capacity = 3u;
    assert(comp::run_geometry_strategy(registry, problem, policy,
        {workspace_bytes, 31u}, buffers, &solution)
        == comp::geometry_strategy_status::insufficient_workspace);
}

void test_registry_rejects_duplicates_and_invalid_tiers() {
    comp::geometry_strategy_descriptor_v1 strategies[] = {
        {comp::geometry_strategy_schema_version,
            comp::geometry_search_tier_instant,
            11u, query_small, execute_small},
        {comp::geometry_strategy_schema_version,
            comp::geometry_search_tier_bounded,
            11u, query_small, execute_small}
    };
    comp::geometry_strategy_registry_v1 registry{strategies, 2u};
    assert(comp::validate_geometry_strategy_registry(registry)
        == comp::geometry_strategy_status::invalid_registry);

    strategies[1].strategy_id = 12u;
    strategies[1].supported_tiers = 1u << 12u;
    assert(comp::validate_geometry_strategy_registry(registry)
        == comp::geometry_strategy_status::invalid_registry);
}

} // namespace

int main() {
    test_resolves_source_linked_strategies_by_tier();
    test_queries_before_running_and_checks_caller_buffers();
    test_registry_rejects_duplicates_and_invalid_tiers();
    return 0;
}
