#include <Cellerator/compute/decomposition/support_contraction_v1.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::compute::decomposition;
namespace operation = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

template<typename Identity>
Identity identity(std::uint64_t value) {
    return {value, value + 1u};
}

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
    relation = {identity<execution::structure_id>(30u),
        {1u}, source, destination, identity<execution::order_id>(40u), 12u};
    operation::operation_problem result{};
    result.kind = operation::operation_kind::contract_on_support;
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
    result.logical_work_items = 12u;
    result.dense_width = 1u;
    return result;
}

}  // namespace

int main() {
    operation::typed_relation relation{};
    auto operation_problem = problem(relation);
    assert(operation::validate_operation_problem(operation_problem));
    const decomposition::destination_interval_v1 intervals[] = {
        {0u, 4u}, {4u, 6u}};
    decomposition::support_contraction_decomposition_v1 value{};
    value.decomposition_identity = {60u, 61u};
    value.problem = &operation_problem;
    value.split_extent = 10u;
    value.intervals = intervals;
    value.interval_count = 2u;
    assert(decomposition::validate_support_contraction_decomposition_v1(
        value));

    value.split = decomposition::support_contraction_split_v1::source_partial;
    value.split_axis = decomposition::split_axis_kind_v1::source;
    value.produces_partial_results = true;
    value.requires_partial_algebra = true;
    assert(decomposition::validate_support_contraction_decomposition_v1(
        value));

    auto invalid = value;
    invalid.requires_partial_algebra = false;
    auto status = decomposition::
        validate_support_contraction_decomposition_v1(invalid);
    assert(status.code == decomposition::
        support_contraction_validation_code_v1::
            invalid_partial_result_contract);

    invalid = value;
    invalid.split_axis = decomposition::split_axis_kind_v1::destination;
    status = decomposition::validate_support_contraction_decomposition_v1(
        invalid);
    assert(status.code == decomposition::
        support_contraction_validation_code_v1::invalid_vocabulary);

    operation_problem.kind = operation::operation_kind::relation_apply;
    invalid = value;
    status = decomposition::validate_support_contraction_decomposition_v1(
        invalid);
    assert(status.code == decomposition::
        support_contraction_validation_code_v1::unsupported_operation);
    return 0;
}
