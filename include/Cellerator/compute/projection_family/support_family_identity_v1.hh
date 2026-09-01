#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::projection_family {

inline constexpr std::uint32_t support_family_identity_schema_version_v1 = 1;

enum support_family_operation_flag_v1 : std::uint32_t {
    support_relation_apply_v1 = 1u << 0u,
    support_relation_apply_transpose_v1 = 1u << 1u,
    support_contract_on_support_v1 = 1u << 2u,
    support_segment_reduce_v1 = 1u << 3u,
    support_segment_normalize_v1 = 1u << 4u,
    support_edge_map_or_gate_v1 = 1u << 5u,
};

inline constexpr std::uint32_t known_support_family_operations_v1 =
    support_relation_apply_v1 | support_relation_apply_transpose_v1
    | support_contract_on_support_v1 | support_segment_reduce_v1
    | support_segment_normalize_v1 | support_edge_map_or_gate_v1;

// Operation-polymorphic identity for one exact immutable logical support.
// Physical projections, device classes, value generations, pointers, and
// operation choices are deliberately absent. The canonical source/destination
// axes retain direction; a transpose view remains in the same family.
struct support_family_identity_v1 {
    std::uint32_t schema_version =
        support_family_identity_schema_version_v1;
    std::uint32_t record_bytes = sizeof(support_family_identity_v1);
    operation::v2::stable_id family_identity{};
    operation::v2::stable_id exact_support_identity{};
    execution::structure_id structure_identity{};
    execution::structure_epoch structure_epoch{};
    execution::persistent_axis_identity source_axis{};
    execution::persistent_axis_identity destination_axis{};
    execution::order_id logical_edge_order{};
    std::uint64_t logical_edge_count = 0;
};

// Capabilities are separate from identity so adding an operation-specific view
// cannot silently create or mutate the underlying support family.
struct support_family_descriptor_v1 {
    support_family_identity_v1 identity{};
    std::uint32_t supported_operations = 0;
    std::uint32_t reserved = 0;
};

enum class support_family_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_family_identity,
    invalid_exact_support_identity,
    invalid_structure_identity,
    missing_structure_epoch,
    invalid_source_axis,
    invalid_destination_axis,
    invalid_logical_edge_order,
    empty_logical_support,
    empty_operation_set,
    unknown_operation,
    nonzero_reserved,
};

struct support_family_validation_v1 {
    support_family_validation_code_v1 code =
        support_family_validation_code_v1::valid;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == support_family_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr support_family_validation_v1
validate_support_family_identity_v1(
    const support_family_identity_v1 &identity) noexcept {
    if (identity.schema_version
        != support_family_identity_schema_version_v1) {
        return {support_family_validation_code_v1::unsupported_schema};
    }
    if (identity.record_bytes != sizeof(support_family_identity_v1)) {
        return {support_family_validation_code_v1::invalid_record_bytes};
    }
    if (!operation::v2::valid_stable_id(identity.family_identity)) {
        return {support_family_validation_code_v1::invalid_family_identity};
    }
    if (!operation::v2::valid_stable_id(identity.exact_support_identity)) {
        return {support_family_validation_code_v1::
                    invalid_exact_support_identity};
    }
    if (!execution::valid_identity(identity.structure_identity)) {
        return {support_family_validation_code_v1::
                    invalid_structure_identity};
    }
    if (identity.structure_epoch.value == 0) {
        return {support_family_validation_code_v1::missing_structure_epoch};
    }
    const auto source_status =
        execution::validate_persistent_axis_identity(identity.source_axis);
    if (source_status != execution::biological_validation_code::ok) {
        return {support_family_validation_code_v1::invalid_source_axis,
                static_cast<std::uint32_t>(source_status)};
    }
    const auto destination_status =
        execution::validate_persistent_axis_identity(identity.destination_axis);
    if (destination_status != execution::biological_validation_code::ok) {
        return {support_family_validation_code_v1::invalid_destination_axis,
                static_cast<std::uint32_t>(destination_status)};
    }
    if (!execution::valid_identity(identity.logical_edge_order)) {
        return {support_family_validation_code_v1::
                    invalid_logical_edge_order};
    }
    if (identity.logical_edge_count == 0) {
        return {support_family_validation_code_v1::empty_logical_support};
    }
    return {};
}

[[nodiscard]] constexpr support_family_validation_v1
validate_support_family_descriptor_v1(
    const support_family_descriptor_v1 &descriptor) noexcept {
    const auto identity_status =
        validate_support_family_identity_v1(descriptor.identity);
    if (!identity_status.valid()) return identity_status;
    if (descriptor.supported_operations == 0) {
        return {support_family_validation_code_v1::empty_operation_set};
    }
    if ((descriptor.supported_operations
         & ~known_support_family_operations_v1) != 0) {
        return {support_family_validation_code_v1::unknown_operation};
    }
    if (descriptor.reserved != 0) {
        return {support_family_validation_code_v1::nonzero_reserved};
    }
    return {};
}

[[nodiscard]] constexpr bool same_support_family_identity_v1(
    const support_family_identity_v1 &lhs,
    const support_family_identity_v1 &rhs) noexcept {
    return operation::v2::same_stable_id(
               lhs.family_identity, rhs.family_identity)
        && operation::v2::same_stable_id(
            lhs.exact_support_identity, rhs.exact_support_identity)
        && execution::same_identity(
            lhs.structure_identity, rhs.structure_identity)
        && lhs.structure_epoch.value == rhs.structure_epoch.value
        && lhs.source_axis.header.schema_version
               == rhs.source_axis.header.schema_version
        && lhs.source_axis.header.kind == rhs.source_axis.header.kind
        && lhs.source_axis.header.byte_count
               == rhs.source_axis.header.byte_count
        && execution::same_identity(lhs.source_axis.domain, rhs.source_axis.domain)
        && execution::same_identity(lhs.source_axis.order, rhs.source_axis.order)
        && execution::same_identity(
            lhs.source_axis.geometry, rhs.source_axis.geometry)
        && execution::same_identity(
            lhs.source_axis.partition, rhs.source_axis.partition)
        && lhs.destination_axis.header.schema_version
               == rhs.destination_axis.header.schema_version
        && lhs.destination_axis.header.kind == rhs.destination_axis.header.kind
        && lhs.destination_axis.header.byte_count
               == rhs.destination_axis.header.byte_count
        && execution::same_identity(
            lhs.destination_axis.domain, rhs.destination_axis.domain)
        && execution::same_identity(
            lhs.destination_axis.order, rhs.destination_axis.order)
        && execution::same_identity(
            lhs.destination_axis.geometry, rhs.destination_axis.geometry)
        && execution::same_identity(
            lhs.destination_axis.partition, rhs.destination_axis.partition)
        && execution::same_identity(
            lhs.logical_edge_order, rhs.logical_edge_order)
        && lhs.logical_edge_count == rhs.logical_edge_count;
}

[[nodiscard]] constexpr bool support_family_supports_v1(
    const support_family_descriptor_v1 &descriptor,
    support_family_operation_flag_v1 operation) noexcept {
    const auto bit = static_cast<std::uint32_t>(operation);
    return bit != 0 && (bit & (bit - 1)) == 0
        && (bit & known_support_family_operations_v1) != 0
        && (descriptor.supported_operations & bit) != 0;
}

static_assert(std::is_standard_layout_v<support_family_identity_v1>);
static_assert(std::is_trivially_copyable_v<support_family_identity_v1>);
static_assert(std::is_standard_layout_v<support_family_descriptor_v1>);
static_assert(std::is_trivially_copyable_v<support_family_descriptor_v1>);

} // namespace cellerator::compute::projection_family
