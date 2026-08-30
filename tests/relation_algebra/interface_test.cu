#include <Cellerator/compute/operation/relation_algebra.hh>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace operation = cellerator::compute::operation;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        {seed + 1u, seed + 2u}, {seed + 3u, seed + 4u},
        {seed + 5u, seed + 6u}, {seed + 7u, seed + 8u}};
}

operation::typed_relation_v1 relation(std::uint64_t seed,
    execution::persistent_axis_identity source = axis(10u),
    execution::persistent_axis_identity destination = axis(20u)) {
    return {{seed + 1u, seed + 2u}, {1u}, source, destination, 17u};
}

operation::relation_numeric_semantics_v1 numeric() {
    return {execution::numeric_type::f16, execution::numeric_type::f16,
        execution::numeric_type::f16, execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32,
        cellerator::compute::math::core::rounding_policy::nearest_even,
        cellerator::compute::math::core::saturation_policy::none,
        operation::nan_policy_v1::propagate, {}};
}

operation::relation_algebra_problem_v1 base_problem() {
    operation::relation_algebra_problem_v1 result{};
    result.operation_identity = {1u, 2u};
    result.relation = relation(30u);
    result.numeric = numeric();
    result.dense_width = 64u;
    result.semantic_flags = operation::alpha_applied_once
        | operation::beta_applied_once;
    return result;
}

} // namespace

int main() {
    static_assert(std::is_trivially_copyable<
        operation::relation_algebra_problem_v1>::value,
        "public relation problem must remain caller-owned POD");
    static_assert(std::is_trivially_copyable<operation::typed_relation_v1>::value,
        "public typed relation must not own model or framework state");

    operation::relation_algebra_problem_v1 problem = base_problem();
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::ok);
    problem.semantic_flags = operation::alpha_applied_once;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::invalid_operation_semantics);

    problem = base_problem();
    problem.kind = operation::relation_algebra_kind_v1::contract_on_support;
    problem.result_axis = axis(40u);
    problem.logical_edge_order = {41u, 42u};
    problem.semantic_flags = operation::stable_logical_edge_output;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::ok);
    problem.logical_edge_order = {};
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::invalid_operation_semantics);

    problem = base_problem();
    problem.kind = operation::relation_algebra_kind_v1::segment_reduce;
    problem.segment = operation::segment_operation_v1::sum;
    problem.values_axis = axis(50u);
    problem.result_axis = axis(60u);
    problem.semantic_flags = operation::empty_sum_is_zero;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::ok);
    problem.segment = operation::segment_operation_v1::maximum;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::invalid_operation_semantics);
    problem.semantic_flags = operation::empty_max_is_negative_infinity;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::ok);

    problem.kind = operation::relation_algebra_kind_v1::segment_normalize;
    problem.segment = operation::segment_operation_v1::softmax;
    problem.semantic_flags = operation::empty_normalization_has_no_output
        | operation::singleton_normalization_is_one;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::ok);
    problem.semantic_flags = operation::singleton_normalization_is_one;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::invalid_operation_semantics);

    problem = base_problem();
    problem.kind = operation::relation_algebra_kind_v1::edge_map_or_gate;
    problem.edge = operation::edge_operation_v1::predicate_gate;
    problem.logical_edge_order = {71u, 72u};
    problem.semantic_flags = operation::projection_aware_edge_values;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::ok);
    problem.edge = operation::edge_operation_v1::none;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::invalid_operation_semantics);

    const auto destination = axis(80u);
    operation::typed_relation_v1 relations[] = {
        relation(81u, axis(82u), destination),
        relation(83u, axis(84u), destination)};
    problem = {};
    problem.kind = operation::relation_algebra_kind_v1::relation_bundle_apply;
    problem.operation_identity = {85u, 86u};
    problem.bundle = {relations, 2u, 0u, destination};
    problem.numeric = numeric();
    problem.semantic_flags = operation::sequential_bundle_is_valid;
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::ok);
    relations[1].destination_axis = axis(90u);
    assert(operation::validate_relation_algebra_problem_v1(problem)
        == operation::relation_algebra_status_v1::invalid_bundle);

    for (std::uint16_t kind = 3u; kind <= 7u; ++kind)
        assert(operation::operation_core_transition(
            static_cast<operation::relation_algebra_kind_v1>(kind)).compatibility
            == operation::operation_core_compatibility_v1::requires_schema_v2);
    return 0;
}
