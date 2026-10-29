#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cassert>
#include <cstdint>

namespace projection = cellerator::compute::projection;

namespace cellerator::compute::projection {
bool build_row_owned_mma_residual_v1(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    const std::uint32_t *, const std::uint32_t *,
    const width_tagged_logical_edge_id_v1 *, std::uint32_t, std::uint32_t,
    std::uint32_t *, std::uint32_t, std::uint32_t *, std::uint32_t,
    projection_value_map_v1 *, std::uint32_t, residual_region_v1 *) noexcept;
}

int main() {
    const std::uint32_t rows[] = {0u, 0u, 2u, 2u};
    const std::uint32_t input_columns[] = {4u, 1u, 3u, 2u};
    projection::width_tagged_logical_edge_id_v1 edge_ids[4]{};
    edge_ids[0].value = 9u;
    edge_ids[1].value = 2u;
    edge_ids[2].value = 7u;
    edge_ids[3].value = 1u;
    std::uint32_t row_offsets[4]{};
    std::uint32_t columns[4]{};
    projection::projection_value_map_v1 value_maps[4]{};
    projection::residual_region_v1 residual{};

    assert(projection::build_row_owned_mma_residual_v1(
        3u, 8u, 1u, 3u, rows, input_columns, edge_ids, 4u, 11u,
        row_offsets, 4u, columns, 4u, value_maps, 4u, &residual));
    const std::uint32_t expected_offsets[] = {0u, 2u, 2u, 4u};
    for (std::uint32_t i = 0u; i < 4u; ++i) {
        assert(row_offsets[i] == expected_offsets[i]);
        assert(columns[i] == input_columns[i]);
        assert(value_maps[i].logical_edge_id.value == edge_ids[i].value);
        assert(value_maps[i].region_kind ==
            projection::physical_region_kind_v1::residual);
        assert(value_maps[i].region_index == 3u);
        assert(value_maps[i].projection_slot == i);
    }
    assert(residual.region_id == 3u);
    assert(residual.semantic_component_id == 8u);
    assert(residual.row_count == 3u);
    assert(residual.edge_count == 4u);
    assert(residual.value_map_offset == 11u);

    // Stable logical identities are independent of physical column order.
    assert(value_maps[0].logical_edge_id.value == 9u);
    assert(columns[0] == 4u);

    const std::uint32_t unsorted_rows[] = {0u, 2u, 1u, 2u};
    assert(!projection::build_row_owned_mma_residual_v1(
        3u, 8u, 1u, 3u, unsorted_rows, input_columns, edge_ids, 4u, 11u,
        row_offsets, 4u, columns, 4u, value_maps, 4u, &residual));
    edge_ids[3].value = edge_ids[0].value;
    assert(!projection::build_row_owned_mma_residual_v1(
        3u, 8u, 1u, 3u, rows, input_columns, edge_ids, 4u, 11u,
        row_offsets, 4u, columns, 4u, value_maps, 4u, &residual));
    return 0;
}
