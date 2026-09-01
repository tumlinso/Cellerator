#pragma once

#include <Cellerator/compute/projection_family/support_family_identity_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::compute::projection_family {

struct logical_edge_segment_v1 {
    std::uint64_t segment_index = 0;
    std::uint64_t logical_edge_id = 0;
};

struct segment_physical_storage_v1 {
    std::uint64_t *segment_offsets = nullptr;
    std::uint64_t segment_offset_capacity = 0;
    std::uint64_t *logical_edge_ids = nullptr;
    std::uint64_t logical_edge_capacity = 0;
    std::uint8_t *logical_edge_marks = nullptr;
    std::uint64_t logical_edge_mark_capacity = 0;
};

struct segment_physical_view_v1 {
    support_family_identity_v1 family{};
    execution::projection_id projection_identity{};
    execution::order_id physical_edge_order{};
    const std::uint64_t *segment_offsets = nullptr;
    const std::uint64_t *logical_edge_ids = nullptr;
    std::uint64_t segment_count = 0;
    std::uint64_t logical_edge_count = 0;
    support_family_operation_flag_v1 operation = support_segment_reduce_v1;
    std::uint32_t reserved = 0;
};

struct gate_physical_view_v1 {
    support_family_identity_v1 family{};
    execution::projection_id projection_identity{};
    execution::order_id physical_edge_order{};
    const std::uint64_t *logical_edge_ids = nullptr;
    const std::uint64_t *gate_indices = nullptr;
    std::uint64_t gate_count = 0;
    std::uint64_t logical_edge_count = 0;
};

enum class segment_physical_code_v1 : std::uint32_t {
    built = 0,
    invalid_family,
    invalid_operation,
    operation_not_supported,
    invalid_projection_identity,
    invalid_physical_order,
    empty_segment_set,
    edge_count_mismatch,
    missing_assignments,
    segment_offset_overflow,
    missing_storage,
    insufficient_offset_capacity,
    insufficient_edge_capacity,
    insufficient_mark_capacity,
    segment_out_of_range,
    logical_edge_out_of_range,
    unordered_assignment,
    duplicate_logical_edge,
    missing_logical_edge,
    missing_output,
};

enum class gate_physical_code_v1 : std::uint32_t {
    built = 0,
    invalid_family,
    operation_not_supported,
    invalid_projection_identity,
    invalid_physical_order,
    empty_gate_set,
    edge_count_mismatch,
    missing_arrays,
    missing_workspace,
    insufficient_mark_capacity,
    logical_edge_out_of_range,
    duplicate_logical_edge,
    gate_index_out_of_range,
    missing_logical_edge,
    missing_output,
};

struct segment_physical_result_v1 {
    segment_physical_code_v1 code = segment_physical_code_v1::built;
    std::uint64_t item_index = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == segment_physical_code_v1::built;
    }
};

struct gate_physical_result_v1 {
    gate_physical_code_v1 code = gate_physical_code_v1::built;
    std::uint64_t item_index = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == gate_physical_code_v1::built;
    }
};

[[nodiscard]] inline segment_physical_result_v1
build_segment_physical_view_v1(
    const support_family_descriptor_v1 &family,
    support_family_operation_flag_v1 operation,
    execution::projection_id projection_identity,
    execution::order_id physical_edge_order,
    std::uint64_t segment_count,
    const logical_edge_segment_v1 *assignments,
    std::uint64_t assignment_count,
    segment_physical_storage_v1 storage,
    segment_physical_view_v1 *output) noexcept {
    const auto status = validate_support_family_descriptor_v1(family);
    if (!status.valid()) {
        return {segment_physical_code_v1::invalid_family,
                static_cast<std::uint64_t>(status.code)};
    }
    if (operation != support_segment_reduce_v1
        && operation != support_segment_normalize_v1) {
        return {segment_physical_code_v1::invalid_operation};
    }
    if (!support_family_supports_v1(family, operation)) {
        return {segment_physical_code_v1::operation_not_supported};
    }
    if (!execution::valid_identity(projection_identity)) {
        return {segment_physical_code_v1::invalid_projection_identity};
    }
    if (!execution::valid_identity(physical_edge_order)) {
        return {segment_physical_code_v1::invalid_physical_order};
    }
    if (segment_count == 0) {
        return {segment_physical_code_v1::empty_segment_set};
    }
    if (assignment_count != family.identity.logical_edge_count) {
        return {segment_physical_code_v1::edge_count_mismatch};
    }
    if (assignments == nullptr) {
        return {segment_physical_code_v1::missing_assignments};
    }
    if (segment_count == std::numeric_limits<std::uint64_t>::max()) {
        return {segment_physical_code_v1::segment_offset_overflow};
    }
    if (storage.segment_offsets == nullptr || storage.logical_edge_ids == nullptr
        || storage.logical_edge_marks == nullptr) {
        return {segment_physical_code_v1::missing_storage};
    }
    if (storage.segment_offset_capacity < segment_count + 1) {
        return {segment_physical_code_v1::insufficient_offset_capacity};
    }
    if (storage.logical_edge_capacity < assignment_count) {
        return {segment_physical_code_v1::insufficient_edge_capacity};
    }
    if (storage.logical_edge_mark_capacity < assignment_count) {
        return {segment_physical_code_v1::insufficient_mark_capacity};
    }
    if (output == nullptr) {
        return {segment_physical_code_v1::missing_output};
    }
    *output = {};
    for (std::uint64_t edge = 0; edge < assignment_count; ++edge) {
        storage.logical_edge_marks[edge] = 0;
    }
    std::uint64_t current_segment = 0;
    storage.segment_offsets[0] = 0;
    for (std::uint64_t index = 0; index < assignment_count; ++index) {
        const auto &assignment = assignments[index];
        if (assignment.segment_index >= segment_count) {
            return {segment_physical_code_v1::segment_out_of_range, index};
        }
        if (assignment.logical_edge_id >= assignment_count) {
            return {segment_physical_code_v1::logical_edge_out_of_range,
                    index};
        }
        if (index != 0) {
            const auto &previous = assignments[index - 1];
            if (previous.segment_index > assignment.segment_index
                || (previous.segment_index == assignment.segment_index
                    && previous.logical_edge_id >= assignment.logical_edge_id)) {
                return {segment_physical_code_v1::unordered_assignment,
                        index};
            }
        }
        if (storage.logical_edge_marks[assignment.logical_edge_id] != 0) {
            return {segment_physical_code_v1::duplicate_logical_edge, index};
        }
        storage.logical_edge_marks[assignment.logical_edge_id] = 1;
        while (current_segment < assignment.segment_index) {
            ++current_segment;
            storage.segment_offsets[current_segment] = index;
        }
        storage.logical_edge_ids[index] = assignment.logical_edge_id;
    }
    while (current_segment < segment_count) {
        ++current_segment;
        storage.segment_offsets[current_segment] = assignment_count;
    }
    for (std::uint64_t edge = 0; edge < assignment_count; ++edge) {
        if (storage.logical_edge_marks[edge] == 0) {
            return {segment_physical_code_v1::missing_logical_edge, edge};
        }
    }
    *output = {family.identity, projection_identity, physical_edge_order,
               storage.segment_offsets, storage.logical_edge_ids,
               segment_count, assignment_count, operation, 0};
    return {segment_physical_code_v1::built, assignment_count};
}

[[nodiscard]] inline gate_physical_result_v1 build_gate_physical_view_v1(
    const support_family_descriptor_v1 &family,
    execution::projection_id projection_identity,
    execution::order_id physical_edge_order,
    std::uint64_t gate_count,
    const std::uint64_t *logical_edge_ids,
    const std::uint64_t *gate_indices,
    std::uint64_t edge_count,
    std::uint8_t *logical_edge_marks,
    std::uint64_t logical_edge_mark_capacity,
    gate_physical_view_v1 *output) noexcept {
    const auto status = validate_support_family_descriptor_v1(family);
    if (!status.valid()) {
        return {gate_physical_code_v1::invalid_family,
                static_cast<std::uint64_t>(status.code)};
    }
    if (!support_family_supports_v1(family, support_edge_map_or_gate_v1)) {
        return {gate_physical_code_v1::operation_not_supported};
    }
    if (!execution::valid_identity(projection_identity)) {
        return {gate_physical_code_v1::invalid_projection_identity};
    }
    if (!execution::valid_identity(physical_edge_order)) {
        return {gate_physical_code_v1::invalid_physical_order};
    }
    if (gate_count == 0) return {gate_physical_code_v1::empty_gate_set};
    if (edge_count != family.identity.logical_edge_count) {
        return {gate_physical_code_v1::edge_count_mismatch};
    }
    if (logical_edge_ids == nullptr || gate_indices == nullptr) {
        return {gate_physical_code_v1::missing_arrays};
    }
    if (logical_edge_marks == nullptr) {
        return {gate_physical_code_v1::missing_workspace};
    }
    if (logical_edge_mark_capacity < edge_count) {
        return {gate_physical_code_v1::insufficient_mark_capacity};
    }
    if (output == nullptr) return {gate_physical_code_v1::missing_output};
    *output = {};
    for (std::uint64_t edge = 0; edge < edge_count; ++edge) {
        logical_edge_marks[edge] = 0;
    }
    for (std::uint64_t index = 0; index < edge_count; ++index) {
        const auto edge = logical_edge_ids[index];
        if (edge >= edge_count) {
            return {gate_physical_code_v1::logical_edge_out_of_range, index};
        }
        if (logical_edge_marks[edge] != 0) {
            return {gate_physical_code_v1::duplicate_logical_edge, index};
        }
        if (gate_indices[index] >= gate_count) {
            return {gate_physical_code_v1::gate_index_out_of_range, index};
        }
        logical_edge_marks[edge] = 1;
    }
    for (std::uint64_t edge = 0; edge < edge_count; ++edge) {
        if (logical_edge_marks[edge] == 0) {
            return {gate_physical_code_v1::missing_logical_edge, edge};
        }
    }
    *output = {family.identity, projection_identity, physical_edge_order,
               logical_edge_ids, gate_indices, gate_count, edge_count};
    return {gate_physical_code_v1::built, edge_count};
}

static_assert(std::is_trivially_copyable_v<logical_edge_segment_v1>);
static_assert(std::is_trivially_copyable_v<segment_physical_view_v1>);
static_assert(std::is_trivially_copyable_v<gate_physical_view_v1>);

} // namespace cellerator::compute::projection_family
