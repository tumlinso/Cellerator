#pragma once

#include <Cellerator/compute/projection_family/support_family_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::projection_family {

inline constexpr std::uint32_t value_gradient_identity_schema_version_v1 = 1;

enum value_gradient_identity_flag_v1 : std::uint32_t {
    value_identity_trainable_v1 = 1u << 0u,
    gradient_identity_present_v1 = 1u << 1u,
};

inline constexpr std::uint32_t known_value_gradient_identity_flags_v1 =
    value_identity_trainable_v1 | gradient_identity_present_v1;

// Pointer-free logical lineage shared by every physical view of one support.
// A generation change replaces mutable state only; it never changes family,
// structure epoch, logical edge ownership, or any physical projection.
struct value_gradient_identity_v1 {
    std::uint32_t schema_version = value_gradient_identity_schema_version_v1;
    std::uint32_t record_bytes = sizeof(value_gradient_identity_v1);
    support_family_identity_v1 family{};
    operation::v2::stable_id value_identity{};
    execution::value_generation value_generation{};
    operation::v2::stable_id gradient_identity{};
    execution::value_generation gradient_generation{};
    execution::order_id logical_edge_order{};
    std::uint64_t logical_edge_count = 0;
    std::uint32_t flags = 0;
    std::uint32_t reserved = 0;
};

enum class value_gradient_identity_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_family,
    invalid_value_identity,
    missing_value_generation,
    invalid_logical_edge_order,
    logical_edge_order_mismatch,
    logical_edge_count_mismatch,
    unknown_flags,
    gradient_without_trainable_value,
    missing_gradient_identity,
    missing_gradient_generation,
    unexpected_gradient_identity,
    unexpected_gradient_generation,
    nonzero_reserved,
};

struct value_gradient_identity_status_v1 {
    value_gradient_identity_code_v1 code =
        value_gradient_identity_code_v1::valid;
    std::uint32_t nested_code = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == value_gradient_identity_code_v1::valid;
    }
};

[[nodiscard]] constexpr value_gradient_identity_status_v1
validate_value_gradient_identity_v1(
    const value_gradient_identity_v1 &identity) noexcept {
    if (identity.schema_version != value_gradient_identity_schema_version_v1) {
        return {value_gradient_identity_code_v1::unsupported_schema};
    }
    if (identity.record_bytes != sizeof(value_gradient_identity_v1)) {
        return {value_gradient_identity_code_v1::invalid_record_bytes};
    }
    const auto family_status = validate_support_family_identity_v1(identity.family);
    if (!family_status.valid()) {
        return {value_gradient_identity_code_v1::invalid_family,
                static_cast<std::uint32_t>(family_status.code)};
    }
    if (!operation::v2::valid_stable_id(identity.value_identity)) {
        return {value_gradient_identity_code_v1::invalid_value_identity};
    }
    if (identity.value_generation.value == 0) {
        return {value_gradient_identity_code_v1::missing_value_generation};
    }
    if (!execution::valid_identity(identity.logical_edge_order)) {
        return {value_gradient_identity_code_v1::invalid_logical_edge_order};
    }
    if (!execution::same_identity(
            identity.logical_edge_order, identity.family.logical_edge_order)) {
        return {value_gradient_identity_code_v1::
                    logical_edge_order_mismatch};
    }
    if (identity.logical_edge_count != identity.family.logical_edge_count) {
        return {value_gradient_identity_code_v1::
                    logical_edge_count_mismatch};
    }
    if ((identity.flags & ~known_value_gradient_identity_flags_v1) != 0) {
        return {value_gradient_identity_code_v1::unknown_flags};
    }
    const bool trainable =
        (identity.flags & value_identity_trainable_v1) != 0;
    const bool has_gradient =
        (identity.flags & gradient_identity_present_v1) != 0;
    if (has_gradient && !trainable) {
        return {value_gradient_identity_code_v1::
                    gradient_without_trainable_value};
    }
    if (has_gradient) {
        if (!operation::v2::valid_stable_id(identity.gradient_identity)) {
            return {value_gradient_identity_code_v1::missing_gradient_identity};
        }
        if (identity.gradient_generation.value == 0) {
            return {value_gradient_identity_code_v1::
                        missing_gradient_generation};
        }
    } else {
        if (operation::v2::valid_stable_id(identity.gradient_identity)) {
            return {value_gradient_identity_code_v1::
                        unexpected_gradient_identity};
        }
        if (identity.gradient_generation.value != 0) {
            return {value_gradient_identity_code_v1::
                        unexpected_gradient_generation};
        }
    }
    if (identity.reserved != 0) {
        return {value_gradient_identity_code_v1::nonzero_reserved};
    }
    return {};
}

[[nodiscard]] constexpr bool same_value_lineage_v1(
    const value_gradient_identity_v1 &lhs,
    const value_gradient_identity_v1 &rhs) noexcept {
    return same_support_family_identity_v1(lhs.family, rhs.family)
        && operation::v2::same_stable_id(
            lhs.value_identity, rhs.value_identity)
        && execution::same_identity(
            lhs.logical_edge_order, rhs.logical_edge_order)
        && lhs.logical_edge_count == rhs.logical_edge_count;
}

[[nodiscard]] constexpr bool same_value_generation_v1(
    const value_gradient_identity_v1 &lhs,
    const value_gradient_identity_v1 &rhs) noexcept {
    return same_value_lineage_v1(lhs, rhs)
        && lhs.value_generation.value == rhs.value_generation.value;
}

[[nodiscard]] constexpr bool same_gradient_generation_v1(
    const value_gradient_identity_v1 &lhs,
    const value_gradient_identity_v1 &rhs) noexcept {
    return same_value_generation_v1(lhs, rhs)
        && (lhs.flags & gradient_identity_present_v1) != 0
        && (rhs.flags & gradient_identity_present_v1) != 0
        && operation::v2::same_stable_id(
            lhs.gradient_identity, rhs.gradient_identity)
        && lhs.gradient_generation.value == rhs.gradient_generation.value;
}

static_assert(std::is_standard_layout_v<value_gradient_identity_v1>);
static_assert(std::is_trivially_copyable_v<value_gradient_identity_v1>);

} // namespace cellerator::compute::projection_family
