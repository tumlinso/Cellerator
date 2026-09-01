#include <Cellerator/execution/atom_fragment/canonical_fallback_v1.hh>
#include <Cellerator/execution/atom_fragment/compiler_registry_v1.hh>
#include <Cellerator/execution/atom_fragment/external_plane_binding_v1.hh>
#include <Cellerator/execution/atom_fragment/output_affordance_v1.hh>

#include <cassert>

namespace compute = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;
namespace fragment = execution::atom_fragment;
namespace joint = execution::joint_compiler;
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

struct relation_state {
    std::uint32_t source[3]{0u, 1u, 1u};
    std::uint32_t destination[3]{0u, 0u, 1u};
};

program::program_status relation_apply(const void *state_value,
    const program::launch_binding_v2 &binding, void *) noexcept {
    const auto &state = *static_cast<const relation_state *>(state_value);
    const auto *input = static_cast<const float *>(binding.input);
    const auto *values = static_cast<const float *>(binding.values);
    auto *output = static_cast<float *>(binding.output);
    if (input == nullptr || values == nullptr || output == nullptr)
        return program::program_status::invalid_argument;
    output[0] = 0.0f;
    output[1] = 0.0f;
    for (std::uint64_t edge = 0u; edge < 3u; ++edge)
        output[state.destination[edge]] += values[edge] * input[state.source[edge]];
    return program::program_status::success;
}

fragment::fragment_compile_status_code_v1 compile_fragment(
    const void *source_context, const joint::atom_fragment_request_v1 &,
    std::uint64_t candidate_id, program::prepared_program_v2 *output) noexcept {
    if (source_context == nullptr || candidate_id != 7u || output == nullptr)
        return fragment::fragment_compile_status_code_v1::invalid_request;
    *output = *static_cast<const program::prepared_program_v2 *>(source_context);
    return fragment::fragment_compile_status_code_v1::success;
}

int main() {
    compute::typed_relation relation{};
    relation.structure = id<execution::structure_tag>(1u);
    relation.epoch = {1u};
    relation.source_axis = axis(10u);
    relation.destination_axis = axis(20u);
    relation.logical_edge_order = id<execution::order_tag>(30u);
    relation.logical_edge_count = 3u;
    compute::operation_problem operation{};
    operation.persistent_problem_identity = {1u, 1u};
    operation.operation_identity = {2u, 1u};
    operation.relations = {&relation, 1u};
    operation.values_axis = axis(40u);
    operation.result_axis = relation.destination_axis;
    operation.logical_edge_order = relation.logical_edge_order;
    operation.expected_value_generation = {1u};
    operation.logical_work_items = 3u;
    operation.dense_width = 1u;
    operation.numeric.relation_storage = execution::numeric_type::f32;
    operation.numeric.state_storage = execution::numeric_type::f32;
    operation.numeric.multiply = execution::numeric_type::f32;
    operation.numeric.accumulation = execution::numeric_type::f32;
    operation.numeric.output_storage = execution::numeric_type::f32;
    operation.numeric.scalar = execution::numeric_type::f32;
    operation.output.produced_axis = operation.result_axis;
    operation.output.canonical_axis = operation.result_axis;
    assert(compute::validate_operation_problem(operation));

    relation_state state{};
    program::prepared_stage_v2 stage{};
    stage.stable_stage_id = 1u;
    stage.candidate_id = 7u;
    stage.prepared_state = &state;
    stage.launch = relation_apply;
    program::prepared_program_v2 compiled_source{};
    compiled_source.stages = &stage;
    compiled_source.stage_count = 1u;
    fragment::fragment_compiler_entry_v1 compiler{{100u, 1u}, 7u,
        {101u, 1u}, &compiled_source, compile_fragment};
    const fragment::fragment_compiler_registry_v1 registry{&compiler, 1u};
    const auto *resolved = fragment::find_fragment_compiler_v1(
        registry, compiler.source_identity, 7u);
    assert(resolved != nullptr);
    program::prepared_program_v2 compiled{};
    joint::atom_fragment_request_v1 compile_request{};
    assert(resolved->compile(resolved->source_context, compile_request, 7u,
        &compiled) == fragment::fragment_compile_status_code_v1::success);

    fragment::atom_bound_candidate_v1 bound{};
    bound.candidate_id = 7u;
    bound.atom_identity = {110u, 1u};
    bound.requirement_identity = {111u, 1u};
    bound.affordance_identity = {112u, 1u};
    fragment::prepared_atom_fragment_v1 prepared{};
    assert(fragment::prepare_atom_fragment_v1(bound, compiled,
        operation.values_axis.order, operation.result_axis.order, &prepared));

    const joint::persistent_identity_v1 species[] = {{113u, 1u}};
    const joint::persistent_identity_v1 plane[] = {{114u, 1u}};
    joint::atom_requirement_v1 requirement{};
    requirement.requirement_identity = bound.requirement_identity;
    requirement.exact_coverage_identity = {115u, 1u};
    requirement.accepted_atom_species = species;
    requirement.accepted_atom_species_count = 1u;
    requirement.required_planes = plane;
    requirement.required_plane_count = 1u;
    requirement.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    requirement.required_order = operation.logical_edge_order;
    requirement.generation_policy = joint::generation_requirement_v1::exact;
    requirement.required_generation = operation.expected_value_generation;
    joint::atom_plane_affordance_v1 plane_affordance{};
    plane_affordance.plane_identity = plane[0];
    plane_affordance.order = requirement.required_order;
    plane_affordance.storage = execution::numeric_type::f32;
    plane_affordance.logical = execution::numeric_type::f32;
    plane_affordance.generation = requirement.required_generation;
    joint::atom_affordance_v1 affordance{};
    affordance.affordance_identity = bound.affordance_identity;
    affordance.atom_species = species[0];
    affordance.exact_coverage_identity = requirement.exact_coverage_identity;
    affordance.physical_encoding = {116u, 1u};
    affordance.local_projection_abi = {117u, 1u};
    affordance.planes = &plane_affordance;
    affordance.plane_count = 1u;

    alignas(16) float values[] = {4.0f, 5.0f, 6.0f};
    joint::external_extent_v1 extent{};
    extent.address = values;
    extent.location = {execution::residency_kind::host, {0u, 0u, 0u}, -1, 1u};
    extent.bytes = sizeof(values);
    extent.alignment = alignof(decltype(values));
    extent.order = requirement.required_order;
    extent.generation = requirement.required_generation;
    extent.readiness = {1u, 1u};
    extent.lease = {2u, 1u};
    joint::external_binding_v1 external{};
    external.binding_identity = {118u, 1u};
    external.atom_identity = bound.atom_identity;
    external.plane_identity = plane[0];
    external.extents = &extent;
    external.extent_count = 1u;
    external.total_bytes = sizeof(values);
    fragment::bound_atom_extent_v1 bound_extent{};
    std::uint64_t written = 0u;
    assert(fragment::bind_external_atom_planes_v1(prepared, requirement,
        affordance, &external, 1u, &bound_extent, 1u, &written));
    assert(written == 1u && bound_extent.lease.slot == 2u);

    const float input[] = {2.0f, 3.0f};
    float output[] = {-1.0f, -1.0f};
    program::launch_binding_v2 launch_binding{};
    launch_binding.input = input;
    launch_binding.values = bound_extent.address;
    launch_binding.output = output;
    assert(program::execute_prepared_program_v2(
        *prepared.program, &launch_binding, 1u, nullptr)
        == program::program_status::success);
    assert(output[0] == 23.0f && output[1] == 18.0f);

    fragment::output_affordance_recipe_v1 output_recipe{};
    output_recipe.output_atom_identity = {120u, 1u};
    output_recipe.output_affordance_identity = {121u, 1u};
    output_recipe.output_plane_identity = {122u, 1u};
    output_recipe.exact_output_coverage = {123u, 1u};
    output_recipe.output_generation = {2u};
    fragment::fragment_output_description_v1 description{};
    assert(fragment::describe_fragment_output_affordances_v1(
        prepared, operation, output_recipe, &description));
    assert(!description.has_partial);

    fragment::canonical_fallback_request_v1 fallback_request{};
    fallback_request.candidate_id = 7u;
    fallback_request.reason = fragment::canonical_fallback_reason_v1::
        forced_by_caller;
    fragment::canonical_fallback_v1 fallback{};
    fragment::canonical_fallback_diagnostic_v1 diagnostic{};
    assert(fragment::make_canonical_fallback_v1(operation, &bound, 1u,
        fallback_request, &fallback, &diagnostic));
    assert(execution::same_identity(
        fallback.output_order, operation.result_axis.order));
}
