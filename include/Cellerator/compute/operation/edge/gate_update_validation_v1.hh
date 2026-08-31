#pragma once

#include <Cellerator/compute/operation/edge/dynamic_support_mask_v1.cuh>
#include <Cellerator/compute/operation/edge/indexed_gates_v1.cuh>
#include <Cellerator/compute/operation/sparse_axis_update/sparse_axis_update_v1.cuh>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::operation::edge {

enum class registered_operation_v1 : std::uint8_t {
    general_map = 0u,
    per_edge_multiplicative,
    per_edge_predicate,
    per_source_gate,
    per_destination_gate,
    per_component_gate,
    factorized_source_destination_gate,
    dynamic_support_byte_mask,
    dynamic_support_bit_mask,
    sparse_assign,
    sparse_add,
    sparse_subtract,
    sparse_multiply,
    sparse_maximum
};

struct registry_entry_v1 {
    std::uint64_t stable_id = 0u;
    const char *unique_name = nullptr;
    registered_operation_v1 operation = registered_operation_v1::general_map;
    bool exact = true;
    bool profiler_visible = true;
    bool requires_measurement = true;
    bool promoted = false;
};

struct validation_result_v1 {
    std::uint64_t checked_item_count = 0u;
    std::uint64_t first_invalid_global_item = 0u;
    bool valid = false;
};

const registry_entry_v1 *registry_v1(std::size_t *count) noexcept;

status_v1 validate_edge_coordinates_v1(const edge_coordinate_v1 *coordinates,
    local_edge_slice_v1 edges, std::uint32_t source_count,
    std::uint32_t destination_count, std::uint32_t component_count,
    validation_result_v1 *result) noexcept;

status_v1 reference_indexed_gate_v1(const edge_coordinate_v1 *coordinates,
    std::uint32_t edge_count, const float *input, const float *primary_gate,
    const float *secondary_gate, indexed_gate_kind_v1 kind,
    float *output) noexcept;

sparse_axis_update::status_v1 reference_sparse_axis_update_v1(
    float *target, std::uint32_t local_axis_count,
    std::uint32_t component_count, const std::uint64_t *global_indices,
    std::uint64_t global_axis_begin, const float *updates,
    std::uint32_t update_count, sparse_axis_update::operation_v1 operation,
    validation_result_v1 *result) noexcept;

} // namespace cellerator::compute::operation::edge
