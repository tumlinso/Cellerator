#include <Cellerator/compute/operation/relation_algebra.hh>

#include <iostream>

namespace {

namespace operation = cellerator::compute::operation;
namespace execution = cellerator::execution;

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {
        {execution::biological_abi_version,
         execution::serialized_record_kind::persistent_axis_identity,
         sizeof(execution::persistent_axis_identity)},
        {seed + 1u, seed + 2u},
        {seed + 3u, seed + 4u},
        {seed + 5u, seed + 6u},
        {seed + 7u, seed + 8u}};
}

operation::typed_relation_v1 relation(std::uint64_t seed) {
    return {{seed + 1u, seed + 2u}, {seed + 3u}, axis(10u), axis(20u), 17u};
}

operation::relation_numeric_semantics_v1 numeric() {
    return {execution::numeric_type::f16,
        execution::numeric_type::f16,
        execution::numeric_type::f16,
        execution::numeric_type::f32,
        execution::numeric_type::f32,
        execution::numeric_type::f32,
        cellerator::compute::math::core::rounding_policy::nearest_even,
        cellerator::compute::math::core::saturation_policy::none,
        operation::nan_policy_v1::propagate,
        {}};
}

int require(bool condition, const char *message) {
    if (condition) return 0;
    std::cerr << "relation algebra contract test failed: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    operation::relation_algebra_problem_v1 problem{};
    problem.operation_identity = {1u, 2u};
    problem.relation = relation(30u);
    problem.numeric = numeric();
    problem.dense_width = 64u;
    problem.semantic_flags = operation::alpha_applied_once
        | operation::beta_applied_once;
    if (require(operation::validate_relation_algebra_problem_v1(problem)
                    == operation::relation_algebra_status_v1::ok,
                "typed relation apply")
        || require(operation::operation_core_transition(problem.kind).compatibility
                       == operation::operation_core_compatibility_v1::direct_schema_v1,
                   "reviewed v1 compatibility")) return 1;

    problem.kind = operation::relation_algebra_kind_v1::contract_on_support;
    problem.result_axis = axis(40u);
    problem.logical_edge_order = {41u, 42u};
    problem.semantic_flags = operation::stable_logical_edge_output;
    if (require(operation::validate_relation_algebra_problem_v1(problem)
                    == operation::relation_algebra_status_v1::ok,
                "support contraction edge identity")
        || require(operation::operation_core_transition(problem.kind).compatibility
                       == operation::operation_core_compatibility_v1::requires_schema_v2,
                   "reviewed schema-v2 transition")) return 1;

    problem.kind = operation::relation_algebra_kind_v1::segment_normalize;
    problem.segment = operation::segment_operation_v1::softmax;
    problem.values_axis = axis(70u);
    problem.result_axis = axis(80u);
    problem.semantic_flags = operation::empty_normalization_has_no_output
        | operation::singleton_normalization_is_one;
    if (require(operation::validate_relation_algebra_problem_v1(problem)
                    == operation::relation_algebra_status_v1::ok,
                "FP32 segment normalization semantics")) return 1;
    problem.numeric.accumulation = execution::numeric_type::f16;
    if (require(operation::validate_relation_algebra_problem_v1(problem)
                    == operation::relation_algebra_status_v1::invalid_operation_semantics,
                "non-FP32 normalization rejection")) return 1;

    operation::typed_relation_v1 relations[2]{relation(50u), relation(60u)};
    problem = {};
    problem.kind = operation::relation_algebra_kind_v1::relation_bundle_apply;
    problem.operation_identity = {3u, 4u};
    problem.numeric = numeric();
    problem.bundle = {relations, 2u, 0u, relations[0].destination_axis};
    problem.semantic_flags = operation::sequential_bundle_is_valid;
    if (require(operation::validate_relation_algebra_problem_v1(problem)
                    == operation::relation_algebra_status_v1::ok,
                "typed relation bundle")) return 1;
    relations[1].destination_axis = axis(90u);
    if (require(operation::validate_relation_algebra_problem_v1(problem)
                    == operation::relation_algebra_status_v1::invalid_bundle,
                "bundle destination mismatch")) return 1;

    std::cout << "celleratorRelationAlgebraContractTest passed\n";
    return 0;
}
