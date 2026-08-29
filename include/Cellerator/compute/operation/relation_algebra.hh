#pragma once

#include <Cellerator/compute/operation/operation_core.hh>
#include <Cellerator/execution/biological_abi.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation {

namespace core = math::core;

inline constexpr std::uint32_t relation_algebra_schema_version_v1 = 1u;
inline constexpr std::uint32_t relation_algebra_operation_core_schema_v2 = 2u;

enum class relation_algebra_kind_v1 : std::uint16_t {
    relation_apply = 1u,
    relation_apply_transpose = 2u,
    contract_on_support = 3u,
    segment_reduce = 4u,
    segment_normalize = 5u,
    edge_map_or_gate = 6u,
    relation_bundle_apply = 7u
};

enum class segment_operation_v1 : std::uint8_t {
    none = 0u,
    sum = 1u,
    maximum = 2u,
    log_sum_exp = 3u,
    softmax = 4u
};

enum class edge_operation_v1 : std::uint8_t {
    none = 0u,
    map = 1u,
    multiplicative_gate = 2u,
    predicate_gate = 3u
};

enum class nan_policy_v1 : std::uint8_t {
    propagate = 1u,
    reject = 2u
};

enum class relation_algebra_status_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_operation = 2u,
    invalid_identity = 3u,
    invalid_relation = 4u,
    invalid_bundle = 5u,
    invalid_numeric_policy = 6u,
    invalid_operation_semantics = 7u
};

enum relation_algebra_semantic_flag_v1 : std::uint32_t {
    alpha_applied_once = 1u << 0u,
    beta_applied_once = 1u << 1u,
    stable_logical_edge_output = 1u << 2u,
    empty_sum_is_zero = 1u << 3u,
    empty_max_is_negative_infinity = 1u << 4u,
    empty_normalization_has_no_output = 1u << 5u,
    singleton_normalization_is_one = 1u << 6u,
    projection_aware_edge_values = 1u << 7u,
    sequential_bundle_is_valid = 1u << 8u
};

// Persistent identity, axes, and epoch describe one immutable typed relation.
// Values and their generations remain launch bindings.
struct typed_relation_v1 {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::persistent_axis_identity source_axis{};
    execution::persistent_axis_identity destination_axis{};
    std::uint64_t logical_edge_count = 0u;
};

struct relation_numeric_semantics_v1 {
    execution::numeric_type relation_storage = execution::numeric_type::invalid;
    execution::numeric_type state_storage = execution::numeric_type::invalid;
    execution::numeric_type multiply = execution::numeric_type::invalid;
    execution::numeric_type accumulation = execution::numeric_type::invalid;
    execution::numeric_type output_storage = execution::numeric_type::invalid;
    execution::numeric_type scalar = execution::numeric_type::invalid;
    core::rounding_policy rounding = core::rounding_policy::nearest_even;
    core::saturation_policy saturation = core::saturation_policy::none;
    nan_policy_v1 nan = nan_policy_v1::propagate;
    std::uint8_t reserved[3]{};
};

// A bundle is a caller-owned cold view. Constituents must share one exact
// destination axis; fusion is optional and sequential accumulation is valid.
struct relation_bundle_view_v1 {
    const typed_relation_v1 *relations = nullptr;
    std::uint32_t relation_count = 0u;
    std::uint32_t reserved = 0u;
    execution::persistent_axis_identity destination_axis{};
};

struct relation_algebra_problem_v1 {
    std::uint32_t schema_version = relation_algebra_schema_version_v1;
    relation_algebra_kind_v1 kind = relation_algebra_kind_v1::relation_apply;
    segment_operation_v1 segment = segment_operation_v1::none;
    edge_operation_v1 edge = edge_operation_v1::none;
    core::stable_id operation_identity{};
    typed_relation_v1 relation{};
    relation_bundle_view_v1 bundle{};
    // Segment operations reduce values_axis into segment_axis. For support
    // contraction this is the stable logical-edge result axis.
    execution::persistent_axis_identity values_axis{};
    execution::persistent_axis_identity result_axis{};
    execution::order_id logical_edge_order{};
    relation_numeric_semantics_v1 numeric{};
    std::uint32_t semantic_flags = 0u;
    std::uint32_t dense_width = 0u;
};

enum class operation_core_compatibility_v1 : std::uint8_t {
    direct_schema_v1 = 1u,
    requires_schema_v2 = 2u
};

struct operation_core_transition_v1 {
    std::uint32_t relation_schema = relation_algebra_schema_version_v1;
    std::uint32_t current_operation_core_schema = core::operation_core_schema_version;
    std::uint32_t required_operation_core_schema =
        relation_algebra_operation_core_schema_v2;
    operation_core_compatibility_v1 compatibility =
        operation_core_compatibility_v1::requires_schema_v2;
    core::operation_kind compatibility_kind =
        core::operation_kind::sparse_dense_multiply;
    std::uint8_t reserved[3]{};
};

constexpr bool same_persistent_axis(
    const execution::persistent_axis_identity &left,
    const execution::persistent_axis_identity &right) noexcept {
    return execution::same_identity(left.domain, right.domain)
        && execution::same_identity(left.order, right.order)
        && execution::same_identity(left.geometry, right.geometry)
        && execution::same_identity(left.partition, right.partition);
}

constexpr bool valid_typed_relation_v1(
    const typed_relation_v1 &relation) noexcept {
    return execution::valid_identity(relation.structure)
        && relation.epoch.value != 0u
        && execution::validate_persistent_axis_identity(relation.source_axis)
            == execution::biological_validation_code::ok
        && execution::validate_persistent_axis_identity(relation.destination_axis)
            == execution::biological_validation_code::ok;
}

constexpr bool valid_relation_numeric_semantics_v1(
    const relation_numeric_semantics_v1 &numeric) noexcept {
    return numeric.relation_storage != execution::numeric_type::invalid
        && numeric.state_storage != execution::numeric_type::invalid
        && numeric.multiply != execution::numeric_type::invalid
        && numeric.accumulation != execution::numeric_type::invalid
        && numeric.output_storage != execution::numeric_type::invalid
        && numeric.scalar != execution::numeric_type::invalid
        && numeric.saturation == core::saturation_policy::none
        && (numeric.nan == nan_policy_v1::propagate
            || numeric.nan == nan_policy_v1::reject);
}

constexpr operation_core_transition_v1 operation_core_transition(
    relation_algebra_kind_v1 kind) noexcept {
    operation_core_transition_v1 transition{};
    if (kind == relation_algebra_kind_v1::relation_apply
        || kind == relation_algebra_kind_v1::relation_apply_transpose) {
        transition.compatibility = operation_core_compatibility_v1::direct_schema_v1;
    }
    return transition;
}

constexpr relation_algebra_status_v1 validate_relation_algebra_problem_v1(
    const relation_algebra_problem_v1 &problem) noexcept {
    if (problem.schema_version != relation_algebra_schema_version_v1)
        return relation_algebra_status_v1::unsupported_schema;
    if (core::same_stable_id(problem.operation_identity, {}))
        return relation_algebra_status_v1::invalid_identity;
    if (!valid_relation_numeric_semantics_v1(problem.numeric))
        return relation_algebra_status_v1::invalid_numeric_policy;

    if (problem.kind == relation_algebra_kind_v1::relation_bundle_apply) {
        if (problem.bundle.relations == nullptr || problem.bundle.relation_count == 0u
            || execution::validate_persistent_axis_identity(
                   problem.bundle.destination_axis)
                != execution::biological_validation_code::ok)
            return relation_algebra_status_v1::invalid_bundle;
        for (std::uint32_t index = 0u; index < problem.bundle.relation_count; ++index) {
            if (!valid_typed_relation_v1(problem.bundle.relations[index])
                || !same_persistent_axis(
                    problem.bundle.relations[index].destination_axis,
                    problem.bundle.destination_axis))
                return relation_algebra_status_v1::invalid_bundle;
        }
        return (problem.semantic_flags & sequential_bundle_is_valid) != 0u
            ? relation_algebra_status_v1::ok
            : relation_algebra_status_v1::invalid_operation_semantics;
    }

    if (!valid_typed_relation_v1(problem.relation))
        return relation_algebra_status_v1::invalid_relation;
    if (problem.kind == relation_algebra_kind_v1::relation_apply
        || problem.kind == relation_algebra_kind_v1::relation_apply_transpose) {
        return problem.dense_width != 0u
                && (problem.semantic_flags & alpha_applied_once) != 0u
                && (problem.semantic_flags & beta_applied_once) != 0u
            ? relation_algebra_status_v1::ok
            : relation_algebra_status_v1::invalid_operation_semantics;
    }
    if (problem.kind == relation_algebra_kind_v1::contract_on_support) {
        return problem.dense_width != 0u
                && execution::validate_persistent_axis_identity(problem.result_axis)
                    == execution::biological_validation_code::ok
                && execution::valid_identity(problem.logical_edge_order)
                && (problem.semantic_flags & stable_logical_edge_output) != 0u
            ? relation_algebra_status_v1::ok
            : relation_algebra_status_v1::invalid_operation_semantics;
    }
    if (problem.kind == relation_algebra_kind_v1::segment_reduce) {
        const bool valid_segment = problem.segment == segment_operation_v1::sum
            || problem.segment == segment_operation_v1::maximum;
        const bool valid_axes = execution::validate_persistent_axis_identity(
                problem.values_axis) == execution::biological_validation_code::ok
            && execution::validate_persistent_axis_identity(problem.result_axis)
                == execution::biological_validation_code::ok;
        const std::uint32_t required = problem.segment == segment_operation_v1::sum
            ? empty_sum_is_zero : empty_max_is_negative_infinity;
        return valid_segment && valid_axes
                && (problem.semantic_flags & required) != 0u
            ? relation_algebra_status_v1::ok
            : relation_algebra_status_v1::invalid_operation_semantics;
    }
    if (problem.kind == relation_algebra_kind_v1::segment_normalize) {
        const bool valid_segment = problem.segment == segment_operation_v1::log_sum_exp
            || problem.segment == segment_operation_v1::softmax;
        const bool valid_axes = execution::validate_persistent_axis_identity(
                problem.values_axis) == execution::biological_validation_code::ok
            && execution::validate_persistent_axis_identity(problem.result_axis)
                == execution::biological_validation_code::ok;
        const std::uint32_t required = empty_normalization_has_no_output
            | singleton_normalization_is_one;
        return valid_segment && valid_axes
                && problem.numeric.accumulation == execution::numeric_type::f32
                && (problem.semantic_flags & required) == required
            ? relation_algebra_status_v1::ok
            : relation_algebra_status_v1::invalid_operation_semantics;
    }
    if (problem.kind == relation_algebra_kind_v1::edge_map_or_gate) {
        return problem.edge != edge_operation_v1::none
                && execution::valid_identity(problem.logical_edge_order)
                && (problem.semantic_flags & projection_aware_edge_values) != 0u
            ? relation_algebra_status_v1::ok
            : relation_algebra_status_v1::invalid_operation_semantics;
    }
    return relation_algebra_status_v1::invalid_operation;
}

static_assert(std::is_trivially_copyable<typed_relation_v1>::value,
    "typed relation contract must remain pointer-free");
static_assert(std::is_trivially_copyable<relation_algebra_problem_v1>::value,
    "relation algebra problem must remain a caller-owned POD view");
static_assert(std::is_trivially_copyable<operation_core_transition_v1>::value,
    "operation-core transition must remain a POD review record");

} // namespace cellerator::compute::operation
