#include <Cellerator/compute/decomposition/relation_bundle_v1.hh>

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

execution::persistent_axis_identity axis(std::uint64_t seed,
    execution::domain_id domain) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        domain, identity<execution::order_id>(seed + 2u),
        identity<execution::geometry_id>(seed + 4u),
        identity<execution::partition_id>(seed + 6u)};
}

operation::operation_problem problem(operation::typed_relation (&relations)[3]) {
    const auto source_a_domain = identity<execution::domain_id>(10u);
    const auto source_b_domain = identity<execution::domain_id>(12u);
    const auto destination_domain = identity<execution::domain_id>(20u);
    const auto source_a = axis(30u, source_a_domain);
    const auto source_b = axis(40u, source_b_domain);
    const auto destination = axis(50u, destination_domain);
    relations[0] = {identity<execution::structure_id>(60u), {1u}, source_a,
        destination, identity<execution::order_id>(70u), 4u};
    relations[1] = {identity<execution::structure_id>(62u), {1u}, source_a,
        destination, identity<execution::order_id>(72u), 4u};
    relations[2] = {identity<execution::structure_id>(64u), {1u}, source_b,
        destination, identity<execution::order_id>(74u), 4u};

    operation::operation_problem result{};
    result.kind = operation::operation_kind::relation_bundle_apply;
    result.persistent_problem_identity = {80u, 81u};
    result.operation_identity = {82u, 83u};
    result.relations = {relations, 3u};
    result.values_axis = source_a;
    result.result_axis = destination;
    result.logical_edge_order = relations[0].logical_edge_order;
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
    operation::typed_relation relations[3]{};
    auto operation_problem = problem(relations);
    assert(operation::validate_operation_problem(operation_problem));
    const decomposition::relation_bundle_type_fragment_v1 fragments[] = {
        {0u, 2u, relations[0].source_axis.domain,
            relations[0].destination_axis.domain},
        {2u, 1u, relations[2].source_axis.domain,
            relations[2].destination_axis.domain}};
    decomposition::relation_bundle_type_decomposition_v1 value{};
    value.decomposition_identity = {90u, 91u};
    value.problem = &operation_problem;
    value.fragments = fragments;
    value.fragment_count = 2u;
    assert(decomposition::validate_relation_bundle_type_decomposition_v1(
        value));

    auto invalid = value;
    decomposition::relation_bundle_type_fragment_v1 bad_type[] = {
        fragments[0], fragments[1]};
    bad_type[0].source_domain = relations[2].source_axis.domain;
    invalid.fragments = bad_type;
    auto status = decomposition::
        validate_relation_bundle_type_decomposition_v1(invalid);
    assert(status.code == decomposition::
        relation_bundle_validation_code_v1::relation_type_mismatch);
    assert(status.relation_index == 0u);

    decomposition::relation_bundle_type_fragment_v1 incomplete[] = {
        {0u, 2u, relations[0].source_axis.domain,
            relations[0].destination_axis.domain}};
    invalid.fragments = incomplete;
    invalid.fragment_count = 1u;
    status = decomposition::validate_relation_bundle_type_decomposition_v1(
        invalid);
    assert(status.code == decomposition::
        relation_bundle_validation_code_v1::incomplete_relation_partition);

    operation_problem.kind = operation::operation_kind::relation_apply;
    invalid = value;
    status = decomposition::validate_relation_bundle_type_decomposition_v1(
        invalid);
    assert(status.code == decomposition::
        relation_bundle_validation_code_v1::unsupported_operation);
    return 0;
}
