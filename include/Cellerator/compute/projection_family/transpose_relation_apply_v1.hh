#pragma once

#include <Cellerator/compute/projection_family/forward_relation_apply_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::compute::projection_family {

struct transpose_relation_apply_storage_v1 {
    std::uint64_t *source_offsets = nullptr;
    std::uint64_t source_offset_capacity = 0;
    std::uint64_t *destination_indices = nullptr;
    std::uint64_t destination_index_capacity = 0;
    std::uint64_t *logical_edge_ids = nullptr;
    std::uint64_t logical_edge_capacity = 0;
    std::uint64_t *source_cursors = nullptr;
    std::uint64_t source_cursor_capacity = 0;
    std::uint8_t *logical_edge_marks = nullptr;
    std::uint64_t logical_edge_mark_capacity = 0;
};

struct transpose_relation_apply_view_v1 {
    support_family_identity_v1 family{};
    execution::projection_id projection_identity{};
    execution::order_id physical_edge_order{};
    const std::uint64_t *source_offsets = nullptr;
    const std::uint64_t *destination_indices = nullptr;
    const std::uint64_t *logical_edge_ids = nullptr;
    std::uint64_t source_count = 0;
    std::uint64_t destination_count = 0;
    std::uint64_t logical_edge_count = 0;
};

enum class transpose_relation_apply_code_v1 : std::uint32_t {
    built = 0,
    invalid_family,
    operation_not_supported,
    family_mismatch,
    invalid_projection_identity,
    invalid_physical_order,
    invalid_forward_view,
    offset_count_overflow,
    missing_storage,
    insufficient_offset_capacity,
    insufficient_destination_capacity,
    insufficient_edge_capacity,
    insufficient_cursor_capacity,
    insufficient_mark_capacity,
    invalid_forward_offset,
    source_index_out_of_range,
    logical_edge_out_of_range,
    unordered_forward_row,
    duplicate_logical_edge,
    missing_logical_edge,
    missing_output,
};

struct transpose_relation_apply_result_v1 {
    transpose_relation_apply_code_v1 code =
        transpose_relation_apply_code_v1::built;
    std::uint64_t subject = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == transpose_relation_apply_code_v1::built;
    }
};

[[nodiscard]] inline transpose_relation_apply_result_v1
build_transpose_relation_apply_view_v1(
    const support_family_descriptor_v1 &family,
    const forward_relation_apply_view_v1 &forward,
    execution::projection_id projection_identity,
    execution::order_id physical_edge_order,
    transpose_relation_apply_storage_v1 storage,
    transpose_relation_apply_view_v1 *output) noexcept {
    const auto family_status = validate_support_family_descriptor_v1(family);
    if (!family_status.valid()) {
        return {transpose_relation_apply_code_v1::invalid_family,
                static_cast<std::uint64_t>(family_status.code)};
    }
    if (!support_family_supports_v1(
            family, support_relation_apply_transpose_v1)) {
        return {transpose_relation_apply_code_v1::operation_not_supported};
    }
    if (!same_support_family_identity_v1(family.identity, forward.family)) {
        return {transpose_relation_apply_code_v1::family_mismatch};
    }
    if (!execution::valid_identity(projection_identity)) {
        return {transpose_relation_apply_code_v1::invalid_projection_identity};
    }
    if (!execution::valid_identity(physical_edge_order)) {
        return {transpose_relation_apply_code_v1::invalid_physical_order};
    }
    if (forward.source_count == 0 || forward.destination_count == 0
        || forward.logical_edge_count != family.identity.logical_edge_count
        || forward.destination_offsets == nullptr
        || forward.source_indices == nullptr
        || forward.logical_edge_ids == nullptr) {
        return {transpose_relation_apply_code_v1::invalid_forward_view};
    }
    if (forward.source_count == std::numeric_limits<std::uint64_t>::max()) {
        return {transpose_relation_apply_code_v1::offset_count_overflow};
    }
    if (storage.source_offsets == nullptr
        || storage.destination_indices == nullptr
        || storage.logical_edge_ids == nullptr
        || storage.source_cursors == nullptr
        || storage.logical_edge_marks == nullptr) {
        return {transpose_relation_apply_code_v1::missing_storage};
    }
    if (storage.source_offset_capacity < forward.source_count + 1) {
        return {transpose_relation_apply_code_v1::
                    insufficient_offset_capacity};
    }
    if (storage.destination_index_capacity < forward.logical_edge_count) {
        return {transpose_relation_apply_code_v1::
                    insufficient_destination_capacity};
    }
    if (storage.logical_edge_capacity < forward.logical_edge_count) {
        return {transpose_relation_apply_code_v1::insufficient_edge_capacity};
    }
    if (storage.source_cursor_capacity < forward.source_count) {
        return {transpose_relation_apply_code_v1::
                    insufficient_cursor_capacity};
    }
    if (storage.logical_edge_mark_capacity < forward.logical_edge_count) {
        return {transpose_relation_apply_code_v1::
                    insufficient_mark_capacity};
    }
    if (output == nullptr) {
        return {transpose_relation_apply_code_v1::missing_output};
    }
    *output = {};

    for (std::uint64_t source = 0; source <= forward.source_count; ++source) {
        storage.source_offsets[source] = 0;
    }
    for (std::uint64_t edge = 0; edge < forward.logical_edge_count; ++edge) {
        storage.logical_edge_marks[edge] = 0;
    }
    if (forward.destination_offsets[0] != 0
        || forward.destination_offsets[forward.destination_count]
               != forward.logical_edge_count) {
        return {transpose_relation_apply_code_v1::invalid_forward_offset};
    }
    for (std::uint64_t destination = 0;
         destination < forward.destination_count;
         ++destination) {
        const auto begin = forward.destination_offsets[destination];
        const auto end = forward.destination_offsets[destination + 1];
        if (end < begin || end > forward.logical_edge_count) {
            return {transpose_relation_apply_code_v1::invalid_forward_offset,
                    destination};
        }
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const auto source = forward.source_indices[edge];
            const auto logical_edge = forward.logical_edge_ids[edge];
            if (source >= forward.source_count) {
                return {transpose_relation_apply_code_v1::
                            source_index_out_of_range,
                        edge};
            }
            if (edge != begin && forward.source_indices[edge - 1] >= source) {
                return {transpose_relation_apply_code_v1::
                            unordered_forward_row,
                        edge};
            }
            if (logical_edge >= forward.logical_edge_count) {
                return {transpose_relation_apply_code_v1::
                            logical_edge_out_of_range,
                        edge};
            }
            if (storage.logical_edge_marks[logical_edge] != 0) {
                return {transpose_relation_apply_code_v1::
                            duplicate_logical_edge,
                        edge};
            }
            storage.logical_edge_marks[logical_edge] = 1;
            ++storage.source_offsets[source + 1];
        }
    }
    for (std::uint64_t edge = 0;
         edge < forward.logical_edge_count;
         ++edge) {
        if (storage.logical_edge_marks[edge] == 0) {
            return {transpose_relation_apply_code_v1::missing_logical_edge,
                    edge};
        }
    }
    for (std::uint64_t source = 0;
         source < forward.source_count;
         ++source) {
        storage.source_offsets[source + 1] += storage.source_offsets[source];
        storage.source_cursors[source] = storage.source_offsets[source];
    }
    for (std::uint64_t destination = 0;
         destination < forward.destination_count;
         ++destination) {
        const auto begin = forward.destination_offsets[destination];
        const auto end = forward.destination_offsets[destination + 1];
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const auto source = forward.source_indices[edge];
            const auto position = storage.source_cursors[source]++;
            storage.destination_indices[position] = destination;
            storage.logical_edge_ids[position] = forward.logical_edge_ids[edge];
        }
    }
    *output = {family.identity,
               projection_identity,
               physical_edge_order,
               storage.source_offsets,
               storage.destination_indices,
               storage.logical_edge_ids,
               forward.source_count,
               forward.destination_count,
               forward.logical_edge_count};
    return {transpose_relation_apply_code_v1::built,
            forward.logical_edge_count};
}

static_assert(std::is_standard_layout_v<transpose_relation_apply_view_v1>);
static_assert(std::is_trivially_copyable_v<transpose_relation_apply_view_v1>);

} // namespace cellerator::compute::projection_family
