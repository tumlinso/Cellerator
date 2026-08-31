#include <Cellerator/geometry/compiler/workload_profile_v2.hh>
#include <Cellerator/compute/architecture/target_cover/strategy_registry.hh>
#include <Cellerator/compute/architecture/target_cover_strategy_v1.hh>

#include <cstdlib>

using namespace cellerator::geometry::compiler::v2;

namespace {
void require(bool condition) { if (!condition) std::abort(); }

workload_status query_semantic(const semantic_strategy_problem &,
    std::uint64_t *bytes, std::uint64_t *alignment) noexcept {
    *bytes = 1; *alignment = 1; return {};
}
workload_status solve_semantic(const semantic_strategy_problem &,
    void *, std::uint64_t, semantic_strategy_solution *) noexcept { return {}; }

void test_workload_profile() {
    workload_component component{};
    component.identity = {1, 2};
    component.dense_width_min = 16;
    component.dense_width_max = 128;
    component.dense_width_bucket = 64;
    component.frequency = (std::uint64_t{1} << 32) + 9;
    workload_profile profile{workload_profile_schema_version,
        sizeof(workload_profile), &component, 1};
    require(static_cast<bool>(validate_workload_profile(profile)));
    component.requirement_flags = canonical_output_required | packed_output_permitted;
    require(validate_workload_profile(profile).code
        == workload_status_code::invalid_requirements);
}

void test_original_groups_and_incremental_window() {
    const std::uint64_t offsets[] = {0, 2, 4};
    const std::uint64_t items[] = {1, (std::uint64_t{1} << 32) + 1, 3, 4};
    original_group_skeleton skeleton{{11, 12}, offsets, 2, items, 4};
    require(static_cast<bool>(validate_original_group_skeleton(skeleton)));
    const std::uint64_t active[] = {0, 1};
    work_window_change change{1, window_change_kind::add_group, {}};
    incremental_work_window window{{13, 14}, skeleton.identity, {}, active, 2, &change, 1};
    require(static_cast<bool>(validate_incremental_work_window(skeleton, window)));
    const std::uint64_t unsorted[] = {1, 0};
    window.active_original_group_ids = unsorted;
    require(validate_incremental_work_window(skeleton, window).code
        == workload_status_code::invalid_argument);
}

void test_separate_strategy_registries() {
    semantic_strategy semantic[2]{};
    semantic[0] = {{1, 1}, "portable-a", query_semantic, solve_semantic, true, false, {}};
    semantic[1] = {{2, 1}, "portable-b", query_semantic, solve_semantic, true, true, {}};
    require(static_cast<bool>(validate_semantic_strategy_registry({semantic, 2})));
    semantic[1].identity = semantic[0].identity;
    require(validate_semantic_strategy_registry({semantic, 2}).code
        == workload_status_code::invalid_argument);
}

void test_exact_evaluator_and_incremental_state() {
    exact_contribution entries[2]{};
    entries[0].logical_identity = 1;
    entries[0].cost = {2, 3, 4, 5, 6, 7};
    entries[1].logical_identity = (std::uint64_t{1} << 32) + 1;
    entries[1].cost = {11, 13, 17, 19, 23, 29};
    exact_evaluation result{};
    require(static_cast<bool>(evaluate_exact({{21, 22}, {23, 24}, {25, 26}, entries, 2}, &result)));
    require(result.total.persistent_bytes == 29);
    incremental_exact_state state{};
    require(static_cast<bool>(initialize_incremental_exact_state(result, {25, 26}, &state)));
    exact_delta delta{};
    delta.removed = entries[0].cost;
    delta.removed_contributions = 1;
    delta.next_work_window = {27, 28};
    require(static_cast<bool>(apply_exact_delta(delta, &state)));
    require(state.evaluated_contributions == 1 && state.generation == 2);
}

void test_multi_candidate_solution_and_snapshot() {
    std::uint64_t bytes[2]{};
    solution_candidate candidates[2]{};
    candidates[0] = {{1, 31}, {3, 4}, {}, {&bytes[0], sizeof(bytes[0])}, false, false, {}};
    candidates[1] = {{2, 31}, {5, 6}, {}, {&bytes[1], sizeof(bytes[1])}, true, true, {}};
    require(static_cast<bool>(validate_multi_candidate_solution(
        {optimizer_stage::portable_semantic_geometry, {}, candidates, 2})));
    optimizer_snapshot snapshot{};
    snapshot.strategy_identity = {3, 4};
    snapshot.problem_identity = {7, 8};
    snapshot.work_window_identity = {9, 10};
    snapshot.iteration = (std::uint64_t{1} << 32) + 1;
    snapshot.state = {bytes, sizeof(bytes)};
    require(static_cast<bool>(validate_optimizer_snapshot(snapshot)));
}

void test_target_cover_keeps_semantics_separate() {
    workload_component component{};
    component.identity = {41, 42};
    workload_profile workload{workload_profile_schema_version,
        sizeof(workload_profile), &component, 1};
    const std::uint64_t edge_count = (std::uint64_t{1} << 32) + 5;
    cellerator::compute::architecture::target_cover::semantic_component semantic{
        7, 0, edge_count};
    cellerator::compute::architecture::target_cover::strategy_problem problem{};
    problem.semantic_geometry_identity = {43, 44};
    problem.provider_identity = {45, 46};
    problem.semantic_components = &semantic;
    problem.semantic_component_count = 1;
    problem.logical_edge_count = edge_count;
    problem.workload = workload;
    using namespace cellerator::compute::architecture::target_cover;
    require(static_cast<bool>(validate_problem(problem)));
    target_region region{1, 7, {}, region_role::pure_sparse, {}, edge_count};
    ownership_range ownership{0, edge_count, 0};
    cover_candidate candidate{};
    candidate.identity = {47, 48};
    candidate.regions = &region;
    candidate.region_count = 1;
    candidate.ownership = &ownership;
    candidate.ownership_range_count = 1;
    strategy_solution solution{};
    solution.semantic_geometry_identity = problem.semantic_geometry_identity;
    solution.provider_identity = problem.provider_identity;
    solution.candidates = &candidate;
    solution.candidate_count = 1;
    solution.logical_edge_count = edge_count;
    require(static_cast<bool>(validate_solution(problem, solution)));
    candidate.kind = cover_kind::conservative_hybrid;
    require(validate_solution(problem, solution).code == workload_status_code::invalid_requirements);
}
}

int main() { test_workload_profile(); test_original_groups_and_incremental_window();
    test_separate_strategy_registries(); test_exact_evaluator_and_incremental_state();
    test_multi_candidate_solution_and_snapshot();
    test_target_cover_keeps_semantics_separate();
    return 0; }
