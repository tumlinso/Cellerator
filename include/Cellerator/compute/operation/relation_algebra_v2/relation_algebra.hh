#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::compute::operation::v2 {

inline constexpr std::uint64_t invalid_binding_index =
    std::numeric_limits<std::uint64_t>::max();

enum class segment_operation : std::uint8_t {
    none = 0,
    sum,
    maximum,
    log_sum_exp,
    softmax,
    log_softmax,
    l1_normalize,
    l2_normalize,
    rms_normalize,
    softmax_backward,
    log_softmax_backward,
    l1_backward,
    l2_backward,
    rms_backward
};

enum class edge_operation : std::uint8_t {
    none = 0,
    arbitrary_map,
    multiplicative_gate,
    predicate_gate,
    active_support_mask
};

enum class gate_indexing : std::uint8_t {
    none = 0,
    per_edge,
    per_source,
    per_destination,
    per_component,
    factorized_source_destination,
    predicate
};

enum relation_value_component_flag : std::uint32_t {
    logical_value_plane = 1u << 0u,
    mma_physical_value_plane = 1u << 1u,
    residual_physical_value_plane = 1u << 2u,
    logical_to_physical_map = 1u << 3u,
    physical_to_logical_map = 1u << 4u,
    value_gradient_plane = 1u << 5u
};

enum relation_semantic_flag : std::uint32_t {
    alpha_applied_once = 1u << 0u,
    beta_applied_once = 1u << 1u,
    stable_logical_edge_output = 1u << 2u,
    empty_sum_is_zero = 1u << 3u,
    empty_max_is_negative_infinity = 1u << 4u,
    empty_normalization_has_no_output = 1u << 5u,
    singleton_normalization_is_exact = 1u << 6u,
    projection_aware_edge_values = 1u << 7u,
    support_superset_preserved = 1u << 8u
};

struct relation_value_binding_contract {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::value_generation generation{};
    execution::value_layout_kind layout = execution::value_layout_kind::logical_edge_order;
    value_ownership_mode ownership = value_ownership_mode::logical_primary;
    std::uint8_t reserved[6]{};
    std::uint32_t required_components = logical_value_plane;
    std::uint32_t reserved_flags = 0;
};

struct relation_binding_contract {
    std::uint64_t relation_index = 0;
    std::uint64_t source_state_operand = invalid_binding_index;
    std::uint64_t destination_state_operand = invalid_binding_index;
    std::uint64_t relation_values = invalid_binding_index;
    std::uint64_t segment_offsets = invalid_binding_index;
    std::uint64_t gate_values = invalid_binding_index;
    std::uint64_t source_gradient = invalid_binding_index;
    std::uint64_t destination_gradient = invalid_binding_index;
    std::uint64_t value_gradient = invalid_binding_index;
};

struct relation_binding_view {
    const relation_binding_contract *bindings = nullptr;
    std::uint64_t binding_count = 0;
};

struct relation_algebra_problem {
    operation_problem core{};
    relation_binding_view bindings{};
    const relation_value_binding_contract *value_bindings = nullptr;
    std::uint64_t value_binding_count = 0;
    segment_operation segment = segment_operation::none;
    edge_operation edge = edge_operation::none;
    gate_indexing gate = gate_indexing::none;
    std::uint8_t reserved[5]{};
    std::uint32_t semantic_flags = 0;
    std::uint32_t reserved_flags = 0;
};

schema_status validate_relation_value_binding(
    const relation_value_binding_contract &binding) noexcept;
schema_status validate_relation_algebra_problem(
    const relation_algebra_problem &problem) noexcept;

constexpr bool is_segment_reduction(segment_operation operation) noexcept {
    return operation == segment_operation::sum
        || operation == segment_operation::maximum;
}

constexpr bool is_segment_normalization(segment_operation operation) noexcept {
    return operation >= segment_operation::log_sum_exp
        && operation <= segment_operation::rms_backward;
}

static_assert(std::is_trivially_copyable_v<relation_value_binding_contract>);
static_assert(std::is_trivially_copyable_v<relation_binding_contract>);
static_assert(std::is_trivially_copyable_v<relation_binding_view>);
static_assert(std::is_trivially_copyable_v<relation_algebra_problem>);

}  // namespace cellerator::compute::operation::v2
