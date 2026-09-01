#include <Cellerator/compute/decomposition/split_segment_reduce_v1.hh>

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

operation::relation_algebra_problem problem(operation::typed_relation &relation) {
    const auto source = axis(10u);
    const auto destination = axis(20u);
    relation = {identity<execution::structure_id>(30u), {1u}, source,
        destination, identity<execution::order_id>(40u), 8u};
    operation::relation_algebra_problem result{};
    result.core.kind = operation::operation_kind::segment_reduce;
    result.core.persistent_problem_identity = {50u, 51u};
    result.core.operation_identity = {52u, 53u};
    result.core.relations = {&relation, 1u};
    result.core.values_axis = source;
    result.core.result_axis = destination;
    result.core.logical_edge_order = relation.logical_edge_order;
    result.core.expected_value_generation = {1u};
    result.core.numeric.relation_storage = execution::numeric_type::f32;
    result.core.numeric.state_storage = execution::numeric_type::f32;
    result.core.numeric.multiply = execution::numeric_type::f32;
    result.core.numeric.accumulation = execution::numeric_type::f32;
    result.core.numeric.output_storage = execution::numeric_type::f32;
    result.core.numeric.scalar = execution::numeric_type::f32;
    result.core.output.produced_axis = destination;
    result.core.output.canonical_axis = destination;
    result.core.logical_work_items = 8u;
    result.core.dense_width = 1u;
    result.segment = operation::segment_operation::sum;
    result.semantic_flags = operation::empty_sum_is_zero;
    return result;
}

}  // namespace

int main() {
    operation::typed_relation relation{};
    auto algebra = problem(relation);
    const std::uint64_t extents[] = {5u, 0u, 3u};
    const decomposition::split_segment_fragment_v1 fragments[] = {
        {0u, 0u, 2u}, {0u, 2u, 3u}, {2u, 0u, 3u}};
    decomposition::split_segment_reduce_decomposition_v1 value{};
    value.decomposition_identity = {60u, 61u};
    value.problem = &algebra;
    value.segment_member_counts = extents;
    value.segment_count = 3u;
    value.fragments = fragments;
    value.fragment_count = 3u;
    assert(decomposition::validate_split_segment_reduce_v1(value));

    auto invalid = value;
    const decomposition::split_segment_fragment_v1 gap[] = {
        {0u, 0u, 2u}, {0u, 3u, 2u}, {2u, 0u, 3u}};
    invalid.fragments = gap;
    auto status = decomposition::validate_split_segment_reduce_v1(invalid);
    assert(status.code == decomposition::
        split_segment_reduce_validation_code_v1::member_offset_mismatch);

    invalid = value;
    invalid.fragment_count = 2u;
    status = decomposition::validate_split_segment_reduce_v1(invalid);
    assert(status.code == decomposition::
        split_segment_reduce_validation_code_v1::incomplete_segment);

    invalid = value;
    invalid.requires_partial_algebra = false;
    status = decomposition::validate_split_segment_reduce_v1(invalid);
    assert(status.code == decomposition::
        split_segment_reduce_validation_code_v1::
            invalid_partial_result_contract);
    return 0;
}
