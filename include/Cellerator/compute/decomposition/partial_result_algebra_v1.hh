#pragma once

#include <Cellerator/execution/joint_compiler/persistent_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t partial_result_algebra_schema_version_v1 = 1u;

enum partial_result_algebra_flag_v1 : std::uint32_t {
    associative_v1 = 1u << 0u,
    commutative_v1 = 1u << 1u,
    idempotent_v1 = 1u << 2u,
    ordered_only_v1 = 1u << 3u,
    deterministic_tree_required_v1 = 1u << 4u
};

inline constexpr std::uint32_t known_partial_result_algebra_flags_v1 =
    associative_v1 | commutative_v1 | idempotent_v1 | ordered_only_v1
    | deterministic_tree_required_v1;

struct partial_result_algebra_v1 {
    std::uint32_t schema_version = partial_result_algebra_schema_version_v1;
    std::uint32_t record_bytes = sizeof(partial_result_algebra_v1);
    execution::joint_compiler::persistent_identity_v1 algebra_identity{};
    execution::joint_compiler::persistent_identity_v1 state_layout_identity{};
    execution::joint_compiler::persistent_identity_v1 neutral_element_identity{};
    execution::joint_compiler::persistent_identity_v1 merge_operation_identity{};
    execution::joint_compiler::persistent_identity_v1 finalize_operation_identity{};
    std::uint64_t state_bytes = 0u;
    std::uint64_t state_alignment = 1u;
    std::uint32_t flags = 0u;
    std::uint32_t reserved = 0u;
    execution::order_id required_merge_order{};
    execution::joint_compiler::persistent_identity_v1
        deterministic_tree_identity{};
    operation::v2::numerical_policy numerical{};
};

enum class partial_result_algebra_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    nonzero_reserved = 3u,
    invalid_algebra_identity = 4u,
    invalid_state_layout = 5u,
    invalid_neutral_element = 6u,
    invalid_merge_operation = 7u,
    invalid_finalize_operation = 8u,
    invalid_state_size = 9u,
    invalid_state_alignment = 10u,
    missing_reconstruction_rule = 11u,
    unknown_flag = 12u,
    invalid_order_constraint = 13u,
    unexpected_order_constraint = 14u,
    missing_deterministic_tree = 15u,
    unexpected_deterministic_tree = 16u,
    invalid_numerical_policy = 17u
};

struct partial_result_algebra_validation_result_v1 {
    partial_result_algebra_validation_code_v1 code =
        partial_result_algebra_validation_code_v1::ok;

    constexpr explicit operator bool() const noexcept {
        return code == partial_result_algebra_validation_code_v1::ok;
    }
};

partial_result_algebra_validation_result_v1
validate_partial_result_algebra_v1(
    const partial_result_algebra_v1 &algebra) noexcept;

static_assert(std::is_standard_layout_v<partial_result_algebra_v1>);
static_assert(std::is_trivially_copyable_v<partial_result_algebra_v1>);

}  // namespace cellerator::compute::decomposition
