#include <Cellerator/geometry/admissibility.hh>
#include <Cellerator/geometry/compiler/strategy_registry.hh>

#include <cassert>

namespace cellerator::geometry::compiler {

const geometry_strategy_descriptor_v1 &identity_geometry_strategy() noexcept;

geometry_strategy_status compile_geometry(
    geometry_strategy_registry_v1,
    const geometry_problem_v1 &,
    const geometry_search_policy_v1 &,
    geometry_strategy_workspace_v1,
    geometry_solution_buffers_v1,
    relation_cover_validation_workspace,
    geometry_solution_v1 *) noexcept;

} // namespace cellerator::geometry::compiler

namespace {

namespace comp = cellerator::geometry::compiler;
namespace geo = cellerator::geometry;
namespace ex = cellerator::execution;

constexpr ex::axis_identity make_axis(std::uint32_t seed) noexcept {
    return {
        {seed + 1u, 1u},
        {seed + 2u, 1u},
        {seed + 3u, 1u},
        {seed + 4u, 1u}
    };
}

comp::geometry_problem_v1 make_problem(
    const std::uint32_t *members,
    const geo::admissibility_view_v1 *admissibility) noexcept {
    comp::geometry_problem_v1 problem{};
    problem.primary_relation.source_axis = make_axis(10u);
    problem.primary_relation.destination_axis = make_axis(20u);
    problem.primary_relation.structure = {31u, 1u};
    problem.primary_relation.epoch = {9u};
    problem.primary_relation.logical_edge_count = 5u;
    problem.primary_relation.location = {ex::residency_kind::host, {}, -1, 0u};
    problem.work_window.identity = {0x101u, 0x202u};
    problem.work_window.axis = problem.primary_relation.destination_axis;
    problem.work_window.axis_extent = 4u;
    problem.work_window.member_count = 3u;
    problem.work_window.members = members;
    problem.admissibility = admissibility;
    problem.workload.relation_value_type = ex::numeric_type::f32;
    problem.workload.dense_input_type = ex::numeric_type::f32;
    problem.workload.accumulation_type = ex::numeric_type::f32;
    problem.workload.output_type = ex::numeric_type::f32;
    problem.workload.dense_width = 1u;
    return problem;
}

struct output_storage {
    std::uint32_t forward[3]{};
    std::uint32_t inverse[3]{};
    geo::semantic_component_v1 component{};
    std::uint64_t edges[5]{};
    std::uint8_t marks[5]{};
};

comp::geometry_solution_buffers_v1 buffers(output_storage *storage) noexcept {
    comp::geometry_solution_buffers_v1 result{};
    result.execution_to_window = storage->forward;
    result.window_to_execution = storage->inverse;
    result.work_item_capacity = 3u;
    result.components = &storage->component;
    result.component_capacity = 1u;
    result.logical_edge_ids = storage->edges;
    result.logical_edge_capacity = 5u;
    return result;
}

void test_identity_strategy_builds_full_relation_cover() {
    const std::uint32_t members[] = {3u, 0u, 2u};
    const geo::admissibility_view_v1 admissibility{};
    const comp::geometry_problem_v1 problem =
        make_problem(members, &admissibility);
    const comp::geometry_strategy_descriptor_v1 &strategy =
        comp::identity_geometry_strategy();
    const comp::geometry_strategy_registry_v1 registry{&strategy, 1u};
    comp::geometry_search_policy_v1 policy{};
    policy.strategy_id = strategy.strategy_id;
    output_storage storage{};
    comp::geometry_solution_v1 solution{};

    assert(comp::compile_geometry(registry, problem, policy, {},
        buffers(&storage), {storage.marks, 5u}, &solution)
        == comp::geometry_strategy_status::ok);
    assert(solution.work_layout.work_count == 3u);
    for (std::uint32_t index = 0u; index < 3u; ++index) {
        assert(storage.forward[index] == index);
        assert(storage.inverse[index] == index);
    }
    assert(solution.relation_cover.component_count == 1u);
    assert(storage.component.kind == geo::semantic_component_kind::unstructured);
    assert(storage.component.logical_edge_count == 5u);
    for (std::uint64_t edge = 0u; edge < 5u; ++edge)
        assert(storage.edges[edge] == edge);
}

comp::geometry_strategy_status query_malformed(
    const comp::geometry_problem_v1 &,
    const comp::geometry_search_policy_v1 &,
    comp::geometry_strategy_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr)
        return comp::geometry_strategy_status::invalid_argument;
    requirements->work_item_capacity = 3u;
    requirements->component_capacity = 1u;
    requirements->logical_edge_capacity = 5u;
    return comp::geometry_strategy_status::ok;
}

comp::geometry_strategy_status execute_malformed(
    const comp::geometry_problem_v1 &problem,
    const comp::geometry_search_policy_v1 &,
    comp::geometry_strategy_workspace_v1,
    comp::geometry_solution_buffers_v1 output,
    comp::geometry_solution_v1 *solution) noexcept {
    output.execution_to_window[0] = 0u;
    output.execution_to_window[1] = 0u;
    output.execution_to_window[2] = 2u;
    output.window_to_execution[0] = 0u;
    output.window_to_execution[1] = 1u;
    output.window_to_execution[2] = 2u;
    output.components[0] =
        {1u, geo::semantic_component_kind::unstructured, {}, 0u, 5u};
    for (std::uint64_t edge = 0u; edge < 5u; ++edge)
        output.logical_edge_ids[edge] = edge;

    comp::geometry_solution_v1 result{};
    result.work_layout.work_window = problem.work_window.identity;
    result.work_layout.axis = problem.work_window.axis;
    result.work_layout.work_count = 3u;
    result.work_layout.execution_to_window = output.execution_to_window;
    result.work_layout.window_to_execution = output.window_to_execution;
    result.relation_cover.structure = problem.primary_relation.structure;
    result.relation_cover.structure_epoch = problem.primary_relation.epoch;
    result.relation_cover.source_axis = problem.primary_relation.source_axis;
    result.relation_cover.destination_axis =
        problem.primary_relation.destination_axis;
    result.relation_cover.logical_edge_count = 5u;
    result.relation_cover.component_count = 1u;
    result.relation_cover.components = output.components;
    result.relation_cover.logical_edge_ids = output.logical_edge_ids;
    *solution = result;
    return comp::geometry_strategy_status::ok;
}

void test_strategy_cannot_certify_malformed_output() {
    const std::uint32_t members[] = {3u, 0u, 2u};
    const comp::geometry_problem_v1 problem = make_problem(members, nullptr);
    const comp::geometry_strategy_descriptor_v1 strategy = {
        comp::geometry_strategy_schema_version,
        comp::geometry_search_tier_instant,
        99u,
        query_malformed,
        execute_malformed
    };
    const comp::geometry_strategy_registry_v1 registry{&strategy, 1u};
    comp::geometry_search_policy_v1 policy{};
    policy.strategy_id = 99u;
    output_storage storage{};
    comp::geometry_solution_v1 solution{};
    assert(comp::compile_geometry(registry, problem, policy, {},
        buffers(&storage), {storage.marks, 5u}, &solution)
        == comp::geometry_strategy_status::strategy_failed);
}

void test_identity_rejects_nonpermissive_constraints() {
    const std::uint32_t members[] = {3u, 0u, 2u};
    geo::admissibility_record_v1 record{};
    record.kind = geo::admissibility_constraint_kind::fixed_position;
    record.axis = make_axis(20u);
    record.subject = 3u;
    record.related = 0u;
    geo::admissibility_view_v1 admissibility{};
    admissibility.record_count = 1u;
    admissibility.records = &record;
    const comp::geometry_problem_v1 problem =
        make_problem(members, &admissibility);
    const comp::geometry_strategy_descriptor_v1 &strategy =
        comp::identity_geometry_strategy();
    const comp::geometry_strategy_registry_v1 registry{&strategy, 1u};
    comp::geometry_search_policy_v1 policy{};
    policy.strategy_id = strategy.strategy_id;
    output_storage storage{};
    comp::geometry_solution_v1 solution{};
    assert(comp::compile_geometry(registry, problem, policy, {},
        buffers(&storage), {storage.marks, 5u}, &solution)
        == comp::geometry_strategy_status::requirements_failed);
}

} // namespace

int main() {
    test_identity_strategy_builds_full_relation_cover();
    test_strategy_cannot_certify_malformed_output();
    test_identity_rejects_nonpermissive_constraints();
    return 0;
}
