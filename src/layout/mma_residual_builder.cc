#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cstdint>
#include <limits>

namespace cellerator::compute::projection {

// Build one row-owned residual without changing the caller's pinned physical
// edge order. Input edges must already be ordered by physical destination row;
// columns retain their input order within each row.
bool build_row_owned_mma_residual_v1(
    std::uint32_t region_id,
    std::uint32_t semantic_component_id,
    std::uint32_t destination_group_index,
    std::uint32_t row_count,
    const std::uint32_t *edge_rows,
    const std::uint32_t *edge_columns,
    const width_tagged_logical_edge_id_v1 *logical_edge_ids,
    std::uint32_t edge_count,
    std::uint32_t value_map_offset,
    std::uint32_t *row_offsets,
    std::uint32_t row_offset_capacity,
    std::uint32_t *columns,
    std::uint32_t column_capacity,
    projection_value_map_v1 *value_maps,
    std::uint32_t value_map_capacity,
    residual_region_v1 *out) noexcept {
    if (semantic_component_id == 0u || row_count == 0u
        || row_count == std::numeric_limits<std::uint32_t>::max()
        || edge_rows == nullptr || edge_columns == nullptr
        || logical_edge_ids == nullptr || edge_count == 0u
        || row_offsets == nullptr || row_offset_capacity < row_count + 1u
        || columns == nullptr || column_capacity < edge_count
        || value_maps == nullptr || value_map_capacity < edge_count
        || out == nullptr
        || value_map_offset > std::numeric_limits<std::uint32_t>::max()
            - edge_count)
        return false;

    logical_edge_id_width_v1 width = logical_edge_ids[0].width;
    if (!valid_logical_edge_id_width_v1(width)) return false;
    std::uint32_t previous_row = 0u;
    for (std::uint32_t edge = 0u; edge < edge_count; ++edge) {
        const width_tagged_logical_edge_id_v1 &identity = logical_edge_ids[edge];
        if (edge_rows[edge] >= row_count
            || (edge != 0u && edge_rows[edge] < previous_row)
            || identity.width != width
            || (width == logical_edge_id_width_v1::u32
                && identity.value > std::numeric_limits<std::uint32_t>::max()))
            return false;
        for (std::uint8_t value : identity.reserved)
            if (value != 0u) return false;
        for (std::uint32_t prior = 0u; prior < edge; ++prior)
            if (logical_edge_ids[prior].value == identity.value)
                return false;
        previous_row = edge_rows[edge];
    }

    std::uint32_t cursor = 0u;
    for (std::uint32_t row = 0u; row < row_count; ++row) {
        row_offsets[row] = cursor;
        while (cursor < edge_count && edge_rows[cursor] == row) {
            columns[cursor] = edge_columns[cursor];
            projection_value_map_v1 map{};
            map.logical_edge_id = logical_edge_ids[cursor];
            map.region_kind = physical_region_kind_v1::residual;
            map.region_index = region_id;
            map.projection_slot = cursor;
            value_maps[cursor] = map;
            ++cursor;
        }
    }
    if (cursor != edge_count) return false;
    row_offsets[row_count] = edge_count;

    residual_region_v1 result{};
    result.region_id = region_id;
    result.semantic_component_id = semantic_component_id;
    result.encoding = residual_encoding_v1::row_owned_csr;
    result.destination_group_index = destination_group_index;
    result.row_offset_index = 0u;
    result.row_count = row_count;
    result.column_index_offset = 0u;
    result.edge_count = edge_count;
    result.value_map_offset = value_map_offset;
    *out = result;
    return true;
}

} // namespace cellerator::compute::projection
