#pragma once

#include <Cellerator/execution/biological_abi.hh>
#include <Cellerator/execution/lifetimes.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation::v2 {

inline constexpr std::uint32_t operation_core_schema_version = 2u;

struct stable_id {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
};

enum class operation_kind : std::uint16_t {
    relation_apply = 1,
    relation_apply_transpose = 2,
    contract_on_support = 3,
    segment_reduce = 4,
    segment_normalize = 5,
    edge_map_or_gate = 6,
    relation_bundle_apply = 7,
    sparse_axis_update = 8
};

enum class relation_orientation : std::uint8_t {
    forward = 1,
    transpose = 2
};

enum class value_ownership_mode : std::uint8_t {
    logical_primary = 1,
    projection_primary = 2
};

enum operation_requirement_flag : std::uint32_t {
    require_forward = 1u << 0u,
    require_backward = 1u << 1u,
    require_value_gradient = 1u << 2u,
    require_source_gradient = 1u << 3u,
    require_destination_gradient = 1u << 4u,
    dynamic_values = 1u << 5u,
    dynamic_support_mask = 1u << 6u
};

enum class rounding_policy : std::uint8_t {
    nearest_even = 1,
    toward_zero = 2,
    stochastic = 3
};

enum class saturation_policy : std::uint8_t {
    none = 1,
    saturate = 2
};

enum class nan_policy : std::uint8_t {
    propagate = 1,
    reject = 2
};

enum class infinity_policy : std::uint8_t {
    propagate = 1,
    reject = 2,
    saturate = 3
};

enum class destination_update : std::uint8_t {
    overwrite = 1,
    accumulate = 2,
    affine_accumulate = 3,
    partial_write = 4
};

enum class output_order_requirement : std::uint8_t {
    preserve_persistent = 1,
    packed_allowed = 2,
    canonical_required = 3
};

inline constexpr std::uint32_t invalid_scalar_binding = 0xffffffffu;

struct numerical_policy {
    execution::numeric_type relation_storage = execution::numeric_type::invalid;
    execution::numeric_type state_storage = execution::numeric_type::invalid;
    execution::numeric_type multiply = execution::numeric_type::invalid;
    execution::numeric_type accumulation = execution::numeric_type::invalid;
    execution::numeric_type output_storage = execution::numeric_type::invalid;
    execution::numeric_type scalar = execution::numeric_type::invalid;
    rounding_policy rounding = rounding_policy::nearest_even;
    saturation_policy saturation = saturation_policy::none;
    nan_policy nan = nan_policy::propagate;
    infinity_policy infinity = infinity_policy::propagate;
};

struct output_contract {
    execution::persistent_axis_identity produced_axis{};
    execution::persistent_axis_identity canonical_axis{};
    destination_update update = destination_update::overwrite;
    output_order_requirement order = output_order_requirement::preserve_persistent;
    bool explicit_order_transform = false;
    bool input_output_aliasing_legal = false;
    std::uint32_t alpha_binding = invalid_scalar_binding;
    std::uint32_t beta_binding = invalid_scalar_binding;
};

struct determinism_contract {
    bool deterministic_required = true;
    bool stable_work_order = true;
    bool fixed_reduction_tree = true;
    bool nondeterministic_atomics_permitted = false;
    std::uint32_t deterministic_seed_binding = invalid_scalar_binding;
    std::uint32_t reserved = 0;
};

struct typed_relation {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::persistent_axis_identity source_axis{};
    execution::persistent_axis_identity destination_axis{};
    execution::order_id logical_edge_order{};
    std::uint64_t logical_edge_count = 0;
};

struct typed_relation_view {
    const typed_relation *relations = nullptr;
    std::uint64_t relation_count = 0;
};

struct value_generation_contract {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::value_generation generation{};
    value_ownership_mode ownership = value_ownership_mode::logical_primary;
    std::uint8_t reserved[7]{};
};

struct operation_problem {
    std::uint32_t schema_version = operation_core_schema_version;
    operation_kind kind = operation_kind::relation_apply;
    relation_orientation orientation = relation_orientation::forward;
    value_ownership_mode value_ownership = value_ownership_mode::logical_primary;
    stable_id persistent_problem_identity{};
    stable_id operation_identity{};
    typed_relation_view relations{};
    execution::persistent_axis_identity values_axis{};
    execution::persistent_axis_identity result_axis{};
    execution::order_id logical_edge_order{};
    execution::value_generation expected_value_generation{};
    numerical_policy numeric{};
    output_contract output{};
    determinism_contract determinism{};
    std::uint64_t logical_work_items = 0;
    std::uint32_t dense_width = 0;
    std::uint32_t requirement_flags = 0;
};

enum class schema_status_code : std::uint8_t {
    ok = 0,
    unsupported_schema,
    invalid_operation,
    invalid_identity,
    invalid_relation,
    invalid_axis,
    invalid_orientation,
    invalid_value_ownership,
    invalid_numeric_policy,
    invalid_output_contract,
    invalid_determinism_contract,
    invalid_generation,
    invalid_shape,
    invalid_argument
};

struct schema_status {
    schema_status_code code = schema_status_code::ok;
    std::uint64_t index = 0;

    constexpr explicit operator bool() const noexcept {
        return code == schema_status_code::ok;
    }
};

constexpr bool same_stable_id(stable_id left, stable_id right) noexcept {
    return left.low == right.low && left.high == right.high;
}

constexpr bool valid_stable_id(stable_id identity) noexcept {
    return !same_stable_id(identity, {});
}

constexpr bool valid_operation_kind(operation_kind kind) noexcept {
    return kind >= operation_kind::relation_apply
        && kind <= operation_kind::sparse_axis_update;
}

schema_status validate_typed_relation(const typed_relation &relation) noexcept;
schema_status validate_numerical_policy(const numerical_policy &policy) noexcept;
schema_status validate_output_contract(const output_contract &contract) noexcept;
schema_status validate_determinism_contract(
    const determinism_contract &contract,
    const numerical_policy &numeric) noexcept;
schema_status validate_operation_problem(const operation_problem &problem) noexcept;

static_assert(std::is_trivially_copyable_v<stable_id>);
static_assert(std::is_trivially_copyable_v<typed_relation>);
static_assert(std::is_trivially_copyable_v<typed_relation_view>);
static_assert(std::is_trivially_copyable_v<value_generation_contract>);
static_assert(std::is_trivially_copyable_v<numerical_policy>);
static_assert(std::is_trivially_copyable_v<output_contract>);
static_assert(std::is_trivially_copyable_v<determinism_contract>);
static_assert(std::is_trivially_copyable_v<operation_problem>);

}  // namespace cellerator::compute::operation::v2
