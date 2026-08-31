#include <Cellerator/compute/operation/operation_core_v2.hh>

#include <cstdlib>

using namespace cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

template <class Tag>
execution::persistent_identity<Tag> persistent(std::uint64_t low) {
    execution::persistent_identity<Tag> result{};
    result.low = low;
    result.high = low + 1;
    return result;
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    execution::persistent_axis_identity result{};
    result.header.schema_version = execution::biological_abi_version;
    result.header.kind = execution::serialized_record_kind::persistent_axis_identity;
    result.header.byte_count = sizeof(result);
    result.domain = persistent<execution::domain_tag>(seed);
    result.order = persistent<execution::order_tag>(seed + 2);
    result.geometry = persistent<execution::geometry_tag>(seed + 4);
    result.partition = persistent<execution::partition_tag>(seed + 6);
    return result;
}

void set_common_policies(operation_problem *problem) {
    problem->numeric.relation_storage = execution::numeric_type::f32;
    problem->numeric.state_storage = execution::numeric_type::f32;
    problem->numeric.multiply = execution::numeric_type::f32;
    problem->numeric.accumulation = execution::numeric_type::f32;
    problem->numeric.output_storage = execution::numeric_type::f32;
    problem->numeric.scalar = execution::numeric_type::f32;
    problem->values_axis = axis(200);
    problem->result_axis = axis(220);
    problem->output.produced_axis = problem->result_axis;
    problem->output.canonical_axis = problem->result_axis;
}

void test_schema_v2_problem() {
    typed_relation relation{};
    relation.structure = persistent<execution::structure_tag>(10);
    relation.epoch.value = 3;
    relation.source_axis = axis(20);
    relation.destination_axis = axis(40);
    relation.logical_edge_order = persistent<execution::order_tag>(60);
    relation.logical_edge_count = (std::uint64_t{1} << 32) + 17;

    operation_problem problem{};
    problem.persistent_problem_identity = {1, 2};
    problem.operation_identity = {3, 4};
    problem.relations = {&relation, 1};
    problem.expected_value_generation.value = 7;
    problem.logical_work_items = relation.logical_edge_count;
    problem.dense_width = 32;
    set_common_policies(&problem);
    require(static_cast<bool>(validate_operation_problem(problem)));

    problem.expected_value_generation.value = 0;
    require(validate_operation_problem(problem).code == schema_status_code::invalid_generation);
}

operation_problem relation_problem(
    operation_kind kind,
    relation_orientation orientation,
    const typed_relation *relation) {
    operation_problem problem{};
    problem.kind = kind;
    problem.orientation = orientation;
    problem.persistent_problem_identity = {11, 12};
    problem.operation_identity = {13, 14};
    problem.relations = {relation, 1};
    problem.logical_edge_order = relation->logical_edge_order;
    problem.expected_value_generation.value = 9;
    problem.logical_work_items = relation->logical_edge_count;
    problem.dense_width = 16;
    set_common_policies(&problem);
    return problem;
}

void test_complete_relation_algebra_bindings() {
    typed_relation relation{};
    relation.structure = persistent<execution::structure_tag>(100);
    relation.epoch.value = 5;
    relation.source_axis = axis(120);
    relation.destination_axis = axis(140);
    relation.logical_edge_order = persistent<execution::order_tag>(160);
    relation.logical_edge_count = 19;

    relation_value_binding_contract values{};
    values.structure = relation.structure;
    values.epoch = relation.epoch;
    values.generation.value = 9;
    relation_binding_contract binding{};
    binding.source_state_operand = 0;
    binding.destination_state_operand = 1;
    binding.relation_values = 0;

    relation_algebra_problem problem{};
    problem.core = relation_problem(
        operation_kind::relation_apply, relation_orientation::forward, &relation);
    problem.bindings = {&binding, 1};
    problem.value_bindings = &values;
    problem.value_binding_count = 1;
    problem.semantic_flags = alpha_applied_once | beta_applied_once;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));

    problem.core.output.update = destination_update::affine_accumulate;
    problem.core.output.alpha_binding = 0;
    problem.core.output.beta_binding = 1;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));

    problem.core.numeric.rounding = rounding_policy::stochastic;
    require(validate_relation_algebra_problem(problem).code
        == schema_status_code::invalid_determinism_contract);
    problem.core.determinism.deterministic_required = false;
    problem.core.determinism.deterministic_seed_binding = 2;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));

    problem.core.kind = operation_kind::relation_apply_transpose;
    problem.core.orientation = relation_orientation::transpose;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));

    problem.core.kind = operation_kind::edge_map_or_gate;
    problem.edge = edge_operation::multiplicative_gate;
    problem.gate = gate_indexing::per_source;
    problem.semantic_flags = projection_aware_edge_values;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));

    values.ownership = value_ownership_mode::projection_primary;
    values.layout = execution::value_layout_kind::projection_local_order;
    values.required_components = mma_physical_value_plane
        | residual_physical_value_plane | physical_to_logical_map;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));
}

void test_output_order_and_determinism_contracts() {
    operation_problem problem{};
    problem.persistent_problem_identity = {21, 22};
    problem.operation_identity = {23, 24};
    problem.kind = operation_kind::segment_reduce;
    problem.logical_work_items = 8;
    set_common_policies(&problem);
    problem.output.order = output_order_requirement::canonical_required;
    problem.output.canonical_axis.order = persistent<execution::order_tag>(500);
    require(validate_operation_problem(problem).code
        == schema_status_code::invalid_output_contract);
    problem.output.explicit_order_transform = true;
    require(static_cast<bool>(validate_operation_problem(problem)));

    problem.numeric.rounding = rounding_policy::stochastic;
    require(validate_operation_problem(problem).code
        == schema_status_code::invalid_determinism_contract);
    problem.determinism.deterministic_required = false;
    problem.determinism.deterministic_seed_binding = 3;
    require(static_cast<bool>(validate_operation_problem(problem)));
}

void test_sparse_update_and_composition_descriptors() {
    sparse_axis_update_descriptor update{};
    update.target_axis = axis(600);
    update.index_type = sparse_index_type::u64;
    update.value_type = execution::numeric_type::f32;
    for (std::uint8_t value = 1; value <= 5; ++value) {
        update.update = static_cast<sparse_update_operation>(value);
        require(static_cast<bool>(validate_sparse_axis_update(update)));
    }
    update.preserve_canonical_identity = false;
    require(validate_sparse_axis_update(update).code == schema_status_code::invalid_axis);

    const composition_stage stages[] = {
        {{31, 32}, operation_kind::contract_on_support, 0},
        {{33, 34}, operation_kind::segment_reduce, 1}
    };
    composition_dependency dependency{0, 1};
    composition_descriptor composition{};
    composition.identity = {35, 36};
    composition.kind = composition_kind::contraction_to_segment;
    composition.stages = stages;
    composition.stage_count = 2;
    composition.dependencies = &dependency;
    composition.dependency_count = 1;
    require(static_cast<bool>(validate_composition(composition)));
    dependency = {1, 0};
    require(validate_composition(composition).code == schema_status_code::invalid_argument);
}

void test_v1_adapter_uses_persistent_identity() {
    cellerator::compute::operation::relation_algebra_problem_v1 source{};
    source.kind = cellerator::compute::operation::relation_algebra_kind_v1::relation_apply;
    source.operation_identity = {71, 72};
    source.relation.structure = persistent<execution::structure_tag>(700);
    source.relation.epoch.value = 2;
    source.relation.source_axis = axis(720);
    source.relation.destination_axis = axis(740);
    source.relation.logical_edge_count = 23;
    source.logical_edge_order = persistent<execution::order_tag>(760);
    source.dense_width = 8;
    source.numeric.relation_storage = execution::numeric_type::f32;
    source.numeric.state_storage = execution::numeric_type::f32;
    source.numeric.multiply = execution::numeric_type::f32;
    source.numeric.accumulation = execution::numeric_type::f32;
    source.numeric.output_storage = execution::numeric_type::f32;
    source.numeric.scalar = execution::numeric_type::f32;
    source.semantic_flags = cellerator::compute::operation::alpha_applied_once
        | cellerator::compute::operation::beta_applied_once;

    typed_relation relations[1]{};
    relation_binding_contract bindings[1]{};
    relation_value_binding_contract values[1]{};
    v1_adapter_request request{};
    request.persistent_problem_identity = {81, 82};
    request.value_generation.value = 4;
    request.storage = {relations, bindings, values, 1};
    v1_adapter_result result{};
    require(static_cast<bool>(adapt_relation_algebra_v1(source, request, &result)));
    require(same_stable_id(result.problem.core.persistent_problem_identity, {81, 82}));
    require(result.problem.core.relations.relations == relations);

    typed_relation moved_relations[1]{};
    relation_binding_contract moved_bindings[1]{};
    relation_value_binding_contract moved_values[1]{};
    request.storage = {moved_relations, moved_bindings, moved_values, 1};
    require(static_cast<bool>(adapt_relation_algebra_v1(source, request, &result)));
    require(same_stable_id(result.problem.core.persistent_problem_identity, {81, 82}));
    request.persistent_problem_identity = {};
    require(adapt_relation_algebra_v1(source, request, &result).code
        == schema_status_code::invalid_argument);
}

}  // namespace

int main() {
    test_schema_v2_problem();
    test_complete_relation_algebra_bindings();
    test_output_order_and_determinism_contracts();
    test_sparse_update_and_composition_descriptors();
    test_v1_adapter_uses_persistent_identity();
    return 0;
}
