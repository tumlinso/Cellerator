#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace projection = cellerator::compute::projection;

int main() {
    static_assert(std::is_standard_layout<
        projection::physical_mma_hybrid_header_v1>::value);
    static_assert(std::is_standard_layout<projection::mma_tile_v1>::value);
    static_assert(std::is_standard_layout<
        projection::projection_value_map_v1>::value);

    projection::physical_mma_hybrid_header_v1 header{};
    header.image_bytes = sizeof(header);
    header.logical_edge_count = 2u;
    header.dense_width = 64u;
    header.source_group_count = 1u;
    header.destination_group_count = 1u;
    header.tile_count = 1u;
    header.compact_slot_count = 2u;
    header.value_map_count = 2u;
    assert(header.header_bytes == sizeof(header));
    assert(header.logical_edge_id_width ==
        projection::logical_edge_id_width_v1::u32);

    projection::physical_group_v1 group{};
    group.group_id = 0u;
    group.member_count = 2u;
    group.padded_count = 16u;
    assert(group.member_count <= projection::mma_group_extent_limit_v1);

    projection::mma_tile_v1 tile{};
    tile.tile_id = 0u;
    tile.source_group_index = 0u;
    tile.destination_group_index = 0u;
    tile.occupancy_mask[0] = (std::uint64_t{1u} << 0u)
        | (std::uint64_t{1u} << 17u);
    tile.compact_slot_count = 2u;
    assert(projection::mma_occupancy_bit_v1(tile, 0u, 0u));
    assert(projection::mma_occupancy_bit_v1(tile, 1u, 1u));
    assert(!projection::mma_occupancy_bit_v1(tile, 0u, 1u));
    assert(!projection::mma_occupancy_bit_v1(tile, 16u, 0u));

    projection::mma_compact_slot_v1 slots[2]{};
    slots[0] = {0u, 0u, 0u, 0u};
    slots[1] = {1u, 1u, 17u, 1u};
    assert(slots[1].dense_slot == 17u);

    projection::projection_value_map_v1 value_map{};
    value_map.logical_edge_id.value = 1u;
    value_map.logical_edge_id.width = projection::logical_edge_id_width_v1::u32;
    value_map.region_kind = projection::physical_region_kind_v1::mma;
    value_map.region_index = 0u;
    value_map.projection_slot = 17u;
    assert(projection::valid_logical_edge_id_width_v1(
        value_map.logical_edge_id.width));
    assert(projection::valid_physical_region_kind_v1(value_map.region_kind));

    projection::residual_region_v1 residual{};
    residual.region_id = 1u;
    residual.row_count = 2u;
    residual.edge_count = 0u;
    assert(residual.encoding == projection::residual_encoding_v1::row_owned_csr);

    projection::projection_schedule_entry_v1 schedule{};
    schedule.kind = projection::schedule_work_kind_v1::mma_tile;
    schedule.work_index = tile.tile_id;
    schedule.dense_column_count = header.dense_width;
    assert(schedule.work_index == 0u);

    return 0;
}
