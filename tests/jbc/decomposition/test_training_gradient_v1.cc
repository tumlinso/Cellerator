#include <Cellerator/compute/decomposition/training_gradient_v1.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::compute::decomposition;
namespace operation = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

template<typename Identity>
Identity identity(std::uint64_t value) { return {value, value + 1u}; }

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        identity<execution::domain_id>(seed),
        identity<execution::order_id>(seed + 2u),
        identity<execution::geometry_id>(seed + 4u),
        identity<execution::partition_id>(seed + 6u)};
}

operation::operation_problem problem(operation::typed_relation &relation) {
    const auto source = axis(10u);
    const auto destination = axis(20u);
    relation = {identity<execution::structure_id>(30u), {1u}, source,
        destination, identity<execution::order_id>(40u), 8u};
    operation::operation_problem result{};
    result.persistent_problem_identity = {50u, 51u};
    result.operation_identity = {52u, 53u};
    result.relations = {&relation, 1u};
    result.values_axis = source;
    result.result_axis = destination;
    result.logical_edge_order = relation.logical_edge_order;
    result.expected_value_generation = {1u};
    result.numeric.relation_storage = execution::numeric_type::f32;
    result.numeric.state_storage = execution::numeric_type::f32;
    result.numeric.multiply = execution::numeric_type::f32;
    result.numeric.accumulation = execution::numeric_type::f32;
    result.numeric.output_storage = execution::numeric_type::f32;
    result.numeric.scalar = execution::numeric_type::f32;
    result.output.produced_axis = destination;
    result.output.canonical_axis = destination;
    result.logical_work_items = 8u;
    result.dense_width = 4u;
    result.requirement_flags = operation::require_backward
        | operation::require_source_gradient
        | operation::require_value_gradient;
    return result;
}

}  // namespace

int main() {
    operation::typed_relation relation{};
    auto operation_problem = problem(relation);
    decomposition::gradient_decomposition_v1 gradients[2]{};
    gradients[0].target = decomposition::gradient_target_v1::source_state;
    gradients[0].biological_axis = relation.source_axis;
    gradients[0].split_axis = decomposition::split_axis_kind_v1::destination;
    gradients[0].accumulation =
        decomposition::gradient_accumulation_v1::associative_partial;
    gradients[0].requires_transpose_projection = true;
    gradients[0].requires_partial_algebra = true;
    gradients[1].target = decomposition::gradient_target_v1::relation_values;
    gradients[1].has_biological_axis = false;
    gradients[1].logical_edge_order = relation.logical_edge_order;
    gradients[1].split_axis = decomposition::split_axis_kind_v1::dense_channel;
    gradients[1].accumulation =
        decomposition::gradient_accumulation_v1::associative_partial;
    gradients[1].requires_partial_algebra = true;
    gradients[1].preserves_logical_edge_identity = true;

    decomposition::training_gradient_contract_v1 contract{};
    contract.identity = {60u, 61u};
    contract.forward_decomposition_identity = {62u, 63u};
    contract.problem = &operation_problem;
    contract.gradients = gradients;
    contract.gradient_count = 2u;
    assert(decomposition::validate_training_gradient_contract_v1(contract));

    auto invalid = contract;
    invalid.gradient_count = 1u;
    auto status = decomposition::validate_training_gradient_contract_v1(
        invalid);
    assert(status.code == decomposition::
        training_gradient_validation_code_v1::missing_requested_gradient);

    gradients[1].preserves_logical_edge_identity = false;
    status = decomposition::validate_training_gradient_contract_v1(contract);
    assert(status.code == decomposition::
        training_gradient_validation_code_v1::edge_identity_mismatch);
    gradients[1].preserves_logical_edge_identity = true;

    gradients[0].requires_transpose_projection = false;
    status = decomposition::validate_training_gradient_contract_v1(contract);
    assert(status.code == decomposition::
        training_gradient_validation_code_v1::invalid_transpose_requirement);
    return 0;
}
