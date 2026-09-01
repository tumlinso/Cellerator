#pragma once

#include <Cellerator/compute/projection_family/support_family_identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::compute::projection_family {

struct logical_relation_edge_v1 {
    std::uint64_t source_index = 0;
    std::uint64_t destination_index = 0;
    std::uint64_t logical_edge_id = 0;
};

struct forward_relation_apply_storage_v1 {
    std::uint64_t *destination_offsets = nullptr;
    std::uint64_t destination_offset_capacity = 0;
    std::uint64_t *source_indices = nullptr;
    std::uint64_t source_index_capacity = 0;
    std::uint64_t *logical_edge_ids = nullptr;
    std::uint64_t logical_edge_capacity = 0;
    std::uint8_t *logical_edge_marks = nullptr;
    std::uint64_t logical_edge_mark_capacity = 0;
};

// Destination-major physical view for forward relation apply. Immutable
// structure arrays are caller-owned and contain no mutable value pointers.
struct forward_relation_apply_view_v1 {
    support_family_identity_v1 family{};
    execution::projection_id projection_identity{};
    execution::order_id physical_edge_order{};
    const std::uint64_t *destination_offsets = nullptr;
    const std::uint64_t *source_indices = nullptr;
    const std::uint64_t *logical_edge_ids = nullptr;
    std::uint64_t source_count = 0;
    std::uint64_t destination_count = 0;
    std::uint64_t logical_edge_count = 0;
};

enum class forward_relation_apply_code_v1 : std::uint32_t {
    built = 0,
    invalid_family,
    operation_not_supported,
    invalid_projection_identity,
    invalid_physical_order,
    empty_source_axis,
    empty_destination_axis,
    edge_count_mismatch,
    missing_edges,
    offset_count_overflow,
    missing_storage,
    insufficient_offset_capacity,
    insufficient_source_capacity,
    insufficient_edge_capacity,
    insufficient_mark_capacity,
    source_index_out_of_range,
    destination_index_out_of_range,
    logical_edge_out_of_range,
    unordered_edge,
    duplicate_logical_edge,
    missing_logical_edge,
    missing_output,
};

struct forward_relation_apply_result_v1 {
    forward_relation_apply_code_v1 code =
        forward_relation_apply_code_v1::built;
    std::uint64_t edge_index = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == forward_relation_apply_code_v1::built;
    }
};

[[nodiscard]] inline forward_relation_apply_result_v1
build_forward_relation_apply_view_v1(
    const support_family_descriptor_v1 &family,
    execution::projection_id projection_identity,
    execution::order_id physical_edge_order,
    std::uint64_t source_count,
    std::uint64_t destination_count,
    const logical_relation_edge_v1 *edges,
    std::uint64_t edge_count,
    forward_relation_apply_storage_v1 storage,
    forward_relation_apply_view_v1 *output) noexcept {
    const auto family_status = validate_support_family_descriptor_v1(family);
    if (!family_status.valid()) {
        return {forward_relation_apply_code_v1::invalid_family,
                static_cast<std::uint64_t>(family_status.code)};
    }
    if (!support_family_supports_v1(
            family, support_relation_apply_v1)) {
        return {forward_relation_apply_code_v1::operation_not_supported};
    }
    if (!execution::valid_identity(projection_identity)) {
        return {forward_relation_apply_code_v1::invalid_projection_identity};
    }
    if (!execution::valid_identity(physical_edge_order)) {
        return {forward_relation_apply_code_v1::invalid_physical_order};
    }
    if (source_count == 0) {
        return {forward_relation_apply_code_v1::empty_source_axis};
    }
    if (destination_count == 0) {
        return {forward_relation_apply_code_v1::empty_destination_axis};
    }
    if (edge_count != family.identity.logical_edge_count) {
        return {forward_relation_apply_code_v1::edge_count_mismatch};
    }
    if (edges == nullptr) {
        return {forward_relation_apply_code_v1::missing_edges};
    }
    if (destination_count == std::numeric_limits<std::uint64_t>::max()) {
        return {forward_relation_apply_code_v1::offset_count_overflow};
    }
    if (storage.destination_offsets == nullptr
        || storage.source_indices == nullptr
        || storage.logical_edge_ids == nullptr
        || storage.logical_edge_marks == nullptr) {
        return {forward_relation_apply_code_v1::missing_storage};
    }
    if (storage.destination_offset_capacity < destination_count + 1) {
        return {forward_relation_apply_code_v1::
                    insufficient_offset_capacity};
    }
    if (storage.source_index_capacity < edge_count) {
        return {forward_relation_apply_code_v1::
                    insufficient_source_capacity};
    }
    if (storage.logical_edge_capacity < edge_count) {
        return {forward_relation_apply_code_v1::
                    insufficient_edge_capacity};
    }
    if (storage.logical_edge_mark_capacity < edge_count) {
        return {forward_relation_apply_code_v1::
                    insufficient_mark_capacity};
    }
    if (output == nullptr) {
        return {forward_relation_apply_code_v1::missing_output};
    }
    *output = {};

    for (std::uint64_t index = 0; index < edge_count; ++index) {
        storage.logical_edge_marks[index] = 0;
    }
    std::uint64_t current_destination = 0;
    storage.destination_offsets[0] = 0;
    for (std::uint64_t index = 0; index < edge_count; ++index) {
        const auto &edge = edges[index];
        if (edge.source_index >= source_count) {
            return {forward_relation_apply_code_v1::
                        source_index_out_of_range,
                    index};
        }
        if (edge.destination_index >= destination_count) {
            return {forward_relation_apply_code_v1::
                        destination_index_out_of_range,
                    index};
        }
        if (edge.logical_edge_id >= edge_count) {
            return {forward_relation_apply_code_v1::
                        logical_edge_out_of_range,
                    index};
        }
        if (index != 0) {
            const auto &previous = edges[index - 1];
            if (previous.destination_index > edge.destination_index
                || (previous.destination_index == edge.destination_index
                    && previous.source_index >= edge.source_index)) {
                return {forward_relation_apply_code_v1::unordered_edge, index};
            }
        }
        if (storage.logical_edge_marks[edge.logical_edge_id] != 0) {
            return {forward_relation_apply_code_v1::duplicate_logical_edge,
                    index};
        }
        storage.logical_edge_marks[edge.logical_edge_id] = 1;
        while (current_destination < edge.destination_index) {
            ++current_destination;
            storage.destination_offsets[current_destination] = index;
        }
        storage.source_indices[index] = edge.source_index;
        storage.logical_edge_ids[index] = edge.logical_edge_id;
    }
    while (current_destination < destination_count) {
        ++current_destination;
        storage.destination_offsets[current_destination] = edge_count;
    }
    for (std::uint64_t logical_edge = 0;
         logical_edge < edge_count;
         ++logical_edge) {
        if (storage.logical_edge_marks[logical_edge] == 0) {
            return {forward_relation_apply_code_v1::missing_logical_edge,
                    logical_edge};
        }
    }
    *output = {family.identity,
               projection_identity,
               physical_edge_order,
               storage.destination_offsets,
               storage.source_indices,
               storage.logical_edge_ids,
               source_count,
               destination_count,
               edge_count};
    return {forward_relation_apply_code_v1::built, edge_count};
}

static_assert(std::is_standard_layout_v<logical_relation_edge_v1>);
static_assert(std::is_trivially_copyable_v<logical_relation_edge_v1>);
static_assert(std::is_standard_layout_v<forward_relation_apply_view_v1>);
static_assert(std::is_trivially_copyable_v<forward_relation_apply_view_v1>);

} // namespace cellerator::compute::projection_family
