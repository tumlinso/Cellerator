#include <Cellerator/compute/operation/relation_algebra.hh>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace operation = cellerator::compute::operation;
namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;

// CE-GEO-81 intentionally keeps this mapping source-private. CE-GEO-86 owns
// the reviewed public catalog declaration.
namespace cellerator::compute::operation::compatibility_detail {
relation_algebra_status_v1 map_relation_apply_to_operation_core_v1(
    const relation_algebra_problem_v1 &typed,
    execution::structure_handle runtime_structure,
    const core::projection_key &projection,
    core::operation_problem *problem,
    core::structure_set_key *structures,
    core::numeric_policy *numeric,
    execution::persistent_axis_identity *input_axis,
    execution::persistent_axis_identity *output_axis) noexcept;
}

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "relation_apply_compatibility_test: " << message << '\n';
        std::exit(1);
    }
}

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

operation::relation_numeric_semantics_v1 numeric() {
    return {execution::numeric_type::f16,
        execution::numeric_type::f16,
        execution::numeric_type::f16,
        execution::numeric_type::f32,
        execution::numeric_type::f32,
        execution::numeric_type::f32,
        core::rounding_policy::nearest_even,
        core::saturation_policy::none,
        operation::nan_policy_v1::propagate,
        {}};
}

operation::relation_algebra_problem_v1 typed_problem(
    operation::relation_algebra_kind_v1 kind,
    std::uint32_t dense_width = 16u) {
    operation::relation_algebra_problem_v1 result{};
    result.kind = kind;
    result.operation_identity = {0x111u, 0x222u};
    result.relation = {{0x333u, 0x444u}, {7u}, axis(10u), axis(20u), 13u};
    result.numeric = numeric();
    result.semantic_flags = operation::alpha_applied_once
        | operation::beta_applied_once;
    result.dense_width = dense_width;
    return result;
}

core::projection_key projection(core::projection_kind kind) {
    return {{0x555u, 0x666u}, {9u, 1u}, kind, 1u, 0u};
}

struct mapped_result {
    core::operation_problem problem{};
    core::structure_set_key structures{};
    core::numeric_policy numeric{};
    execution::persistent_axis_identity input_axis{};
    execution::persistent_axis_identity output_axis{};
};

operation::relation_algebra_status_v1 map(
    const operation::relation_algebra_problem_v1 &typed,
    const core::projection_key &physical,
    mapped_result *out,
    execution::structure_handle runtime = {8u, 1u}) {
    return operation::compatibility_detail::
        map_relation_apply_to_operation_core_v1(typed, runtime, physical,
            &out->problem, &out->structures, &out->numeric,
            &out->input_axis, &out->output_axis);
}

void forward_preserves_frozen_v1_semantics() {
    const auto typed = typed_problem(
        operation::relation_algebra_kind_v1::relation_apply);
    mapped_result mapped{};
    require(map(typed, projection(core::projection_kind::csr), &mapped)
            == operation::relation_algebra_status_v1::ok,
        "forward relation apply mapping failed");
    require(mapped.problem.schema_version == core::operation_core_schema_version
        && mapped.problem.kind == core::operation_kind::sparse_dense_multiply
        && core::same_stable_id(mapped.problem.operation,
            typed.operation_identity)
        && mapped.problem.input_count == 1u
        && mapped.problem.output_count == 1u
        && mapped.problem.logical_work_items == 13u * 16u,
        "forward mapping changed frozen operation semantics");
    require(core::validate_operation_problem(
            mapped.problem, mapped.structures)
        && core::validate_numeric_policy(mapped.numeric),
        "forward mapping did not produce valid operation-core inputs");
    require(mapped.structures.count == 1u
        && execution::same_identity(
            mapped.structures.structures[0].persistent,
            typed.relation.structure)
        && execution::same_handle(
            mapped.structures.structures[0].runtime, {8u, 1u})
        && mapped.structures.structures[0].epoch.value
            == typed.relation.epoch.value,
        "forward mapping changed structure identity or epoch");
    require(operation::same_persistent_axis(
            mapped.input_axis, typed.relation.source_axis)
        && operation::same_persistent_axis(
            mapped.output_axis, typed.relation.destination_axis),
        "forward mapping changed source or destination axis");
    require(mapped.numeric.sparse_storage == typed.numeric.relation_storage
        && mapped.numeric.dense_storage == typed.numeric.state_storage
        && mapped.numeric.output_storage == typed.numeric.output_storage
        && mapped.numeric.multiply == typed.numeric.multiply
        && mapped.numeric.accumulation == typed.numeric.accumulation
        && mapped.numeric.scalar == typed.numeric.scalar
        && mapped.numeric.quantization == core::quantization_granularity::none,
        "forward mapping changed numeric semantics");
}

void transpose_reverses_axes_without_reinterpreting_enum() {
    const auto typed = typed_problem(
        operation::relation_algebra_kind_v1::relation_apply_transpose, 1u);
    mapped_result mapped{};
    require(map(typed,
            projection(core::projection_kind::transpose_or_backward), &mapped)
            == operation::relation_algebra_status_v1::ok,
        "transpose relation apply mapping failed");
    require(mapped.problem.kind == core::operation_kind::sparse_dense_multiply
        && core::same_stable_id(mapped.problem.operation,
            typed.operation_identity)
        && mapped.problem.logical_work_items == typed.relation.logical_edge_count,
        "transpose reinterpreted frozen v1 operation kind or identity");
    require(operation::same_persistent_axis(
            mapped.input_axis, typed.relation.destination_axis)
        && operation::same_persistent_axis(
            mapped.output_axis, typed.relation.source_axis),
        "transpose did not reverse typed relation axes");
}

void reject_ambiguous_or_unrepresentable_mappings() {
    mapped_result mapped{};
    const auto forward = typed_problem(
        operation::relation_algebra_kind_v1::relation_apply);
    require(map(forward,
            projection(core::projection_kind::transpose_or_backward), &mapped)
            == operation::relation_algebra_status_v1::invalid_operation_semantics,
        "forward accepted transpose-only projection");
    const auto transpose = typed_problem(
        operation::relation_algebra_kind_v1::relation_apply_transpose);
    require(map(transpose, projection(core::projection_kind::csr), &mapped)
            == operation::relation_algebra_status_v1::invalid_operation_semantics,
        "transpose accepted forward projection");
    require(map(forward, projection(core::projection_kind::csr), &mapped,
            {}) == operation::relation_algebra_status_v1::invalid_identity,
        "mapping invented a runtime structure handle");

    auto overflow = forward;
    overflow.relation.logical_edge_count =
        std::numeric_limits<std::uint64_t>::max();
    require(map(overflow, projection(core::projection_kind::csr), &mapped)
            == operation::relation_algebra_status_v1::invalid_operation_semantics,
        "mapping accepted overflowing complete work count");
    auto empty = forward;
    empty.relation.logical_edge_count = 0u;
    require(map(empty, projection(core::projection_kind::csr), &mapped)
            == operation::relation_algebra_status_v1::invalid_operation_semantics,
        "mapping accepted an empty v1 operation-core work count");
    auto stale = forward;
    stale.relation.epoch = {};
    require(map(stale, projection(core::projection_kind::csr), &mapped)
            == operation::relation_algebra_status_v1::invalid_relation,
        "mapping accepted stale typed relation epoch");
}

} // namespace

int main() {
    static_assert(core::operation_core_schema_version == 1u,
        "CE-GEO-81 may not silently revise operation-core schema");
    static_assert(static_cast<std::uint16_t>(
            core::operation_kind::sparse_dense_multiply) == 1u,
        "frozen sparse-dense operation enum changed");
    static_assert(static_cast<std::uint16_t>(
            core::projection_kind::transpose_or_backward) == 10u,
        "frozen transpose projection enum changed");
    forward_preserves_frozen_v1_semantics();
    transpose_reverses_axes_without_reinterpreting_enum();
    reject_ambiguous_or_unrepresentable_mappings();
    std::cout << "relation_apply_compatibility_test passed forward=1 "
                 "transpose=1 frozen_enum=1 identity=1\n";
    return 0;
}
