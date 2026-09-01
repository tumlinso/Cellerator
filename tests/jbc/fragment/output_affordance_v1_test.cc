#include <Cellerator/execution/atom_fragment/output_affordance_v1.hh>

#include <cassert>

namespace compute = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;
namespace fragment = execution::atom_fragment;
namespace program = execution::program;

template<typename Tag>
execution::persistent_identity<Tag> id(std::uint64_t value) {
    return {value, value + 100u};
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        id<execution::domain_tag>(seed), id<execution::order_tag>(seed + 1u),
        id<execution::geometry_tag>(seed + 2u),
        id<execution::partition_tag>(seed + 3u)};
}

program::program_status launch(const void *,
    const program::launch_binding_v2 &, void *) noexcept {
    return program::program_status::success;
}

int main() {
    compute::typed_relation relation{};
    relation.structure = id<execution::structure_tag>(1u);
    relation.epoch = {1u};
    relation.source_axis = axis(10u);
    relation.destination_axis = axis(20u);
    relation.logical_edge_order = id<execution::order_tag>(30u);
    relation.logical_edge_count = 8u;
    compute::operation_problem operation{};
    operation.persistent_problem_identity = {1u, 1u};
    operation.operation_identity = {2u, 1u};
    operation.relations = {&relation, 1u};
    operation.values_axis = axis(40u);
    operation.result_axis = axis(50u);
    operation.logical_edge_order = relation.logical_edge_order;
    operation.expected_value_generation = {1u};
    operation.logical_work_items = 8u;
    operation.dense_width = 1u;
    operation.numeric.relation_storage = execution::numeric_type::f32;
    operation.numeric.state_storage = execution::numeric_type::f32;
    operation.numeric.multiply = execution::numeric_type::f32;
    operation.numeric.accumulation = execution::numeric_type::f64;
    operation.numeric.output_storage = execution::numeric_type::f32;
    operation.numeric.scalar = execution::numeric_type::f32;
    operation.output.produced_axis = operation.result_axis;
    operation.output.canonical_axis = operation.result_axis;

    program::prepared_stage_v2 stage{};
    stage.stable_stage_id = 1u;
    stage.candidate_id = 7u;
    stage.launch = launch;
    program::prepared_program_v2 source{};
    source.stages = &stage;
    source.stage_count = 1u;
    fragment::atom_bound_candidate_v1 bound{};
    bound.candidate_id = 7u;
    bound.atom_identity = {3u, 1u};
    bound.requirement_identity = {4u, 1u};
    bound.affordance_identity = {5u, 1u};
    fragment::prepared_atom_fragment_v1 prepared{};
    assert(fragment::prepare_atom_fragment_v1(bound, source,
        operation.values_axis.order, operation.result_axis.order, &prepared));

    fragment::output_affordance_recipe_v1 recipe{};
    recipe.output_atom_identity = {10u, 1u};
    recipe.output_affordance_identity = {11u, 1u};
    recipe.output_plane_identity = {12u, 1u};
    recipe.exact_output_coverage = {13u, 1u};
    recipe.output_generation = {9u};
    recipe.produces_partial = true;
    recipe.partial_affordance_identity = {14u, 1u};
    recipe.partial_plane_identity = {15u, 1u};
    recipe.partial_algebra = {16u, 1u};
    fragment::fragment_output_description_v1 description{};
    assert(fragment::describe_fragment_output_affordances_v1(
        prepared, operation, recipe, &description));
    assert(description.has_partial);
    assert(description.output.storage == execution::numeric_type::f32);
    assert(description.partial.storage == execution::numeric_type::f64);
    assert(description.partial.partial_algebra.local_identity == 1u);

    recipe.produces_partial = false;
    const auto inconsistent = fragment::describe_fragment_output_affordances_v1(
        prepared, operation, recipe, &description);
    assert(inconsistent.code == fragment::output_affordance_status_code_v1::
        invalid_partial_recipe);
    assert(!description.has_partial);

    recipe.partial_affordance_identity = {};
    recipe.partial_plane_identity = {};
    recipe.partial_algebra = {};
    prepared.output_order = {99u, 1u};
    const auto order = fragment::describe_fragment_output_affordances_v1(
        prepared, operation, recipe, &description);
    assert(order.code == fragment::output_affordance_status_code_v1::
        inconsistent_output_order);
}
