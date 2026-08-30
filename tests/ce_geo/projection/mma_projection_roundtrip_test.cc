#include <Cellerator/compute/projection/physical_mma_hybrid.hh>
#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace projection = cellerator::compute::projection;
namespace persistence = cellpack::persistence;
namespace execution = cellerator::execution;

namespace cellerator::compute::projection {
bool validate_physical_mma_hybrid_image_v1(
    const void *, std::size_t) noexcept;
bool make_physical_mma_hybrid_cpe2_source_v1(
    const void *, std::size_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t, std::uint32_t, persistence::execution_section_source *,
    persistence::execution_projection_source *) noexcept;
}

namespace {

std::size_t align64(std::size_t value) {
    return (value + 63u) & ~std::size_t{63u};
}

template<typename T>
std::size_t append_records(std::vector<unsigned char> *image,
    const T *records, std::size_t count) {
    const std::size_t offset = align64(image->size());
    image->resize(offset + sizeof(T) * count);
    std::memcpy(image->data() + offset, records, sizeof(T) * count);
    return offset;
}

std::vector<unsigned char> physical_image() {
    std::vector<unsigned char> image(
        sizeof(projection::physical_mma_hybrid_header_v1));

    projection::physical_group_v1 source{};
    source.group_id = 0u;
    source.semantic_component_id = 1u;
    source.member_count = 1u;
    source.padded_count = 16u;
    projection::physical_group_v1 destination = source;

    projection::mma_tile_v1 tile{};
    tile.tile_id = 0u;
    tile.source_group_index = 0u;
    tile.destination_group_index = 0u;
    tile.semantic_component_id = 1u;
    tile.occupancy_mask[0] = 1u;
    tile.compact_slot_count = 1u;

    projection::mma_compact_slot_v1 slot{};
    slot.logical_edge_index = 0u;

    projection::residual_region_v1 residual{};
    residual.region_id = 0u;
    residual.semantic_component_id = 1u;
    residual.destination_group_index = 0u;
    residual.row_count = 1u;
    residual.edge_count = 1u;
    residual.value_map_offset = 1u;
    const std::uint32_t row_offsets[] = {0u, 1u};
    const std::uint32_t columns[] = {0u};

    projection::projection_schedule_entry_v1 schedules[2]{};
    schedules[0].kind = projection::schedule_work_kind_v1::mma_tile;
    schedules[0].work_index = 0u;
    schedules[0].destination_group_index = 0u;
    schedules[0].dense_column_count = 64u;
    schedules[1].kind = projection::schedule_work_kind_v1::residual_region;
    schedules[1].work_index = 0u;
    schedules[1].destination_group_index = 0u;
    schedules[1].dense_column_count = 64u;

    projection::projection_value_map_v1 maps[2]{};
    maps[0].logical_edge_id.value = 0u;
    maps[0].region_kind = projection::physical_region_kind_v1::mma;
    maps[0].region_index = 0u;
    maps[0].projection_slot = 0u;
    maps[1].logical_edge_id.value = 1u;
    maps[1].region_kind = projection::physical_region_kind_v1::residual;
    maps[1].region_index = 0u;
    maps[1].projection_slot = 0u;

    projection::physical_mma_hybrid_header_v1 header{};
    header.structure_identity.identity_version = 1u;
    header.structure_identity.value = 0x101u;
    header.source_order.feature_count = 1u;
    header.source_order.feature_axis_identity_version = 1u;
    header.source_order.feature_axis_identity = 0x201u;
    header.destination_order.feature_count = 1u;
    header.destination_order.feature_axis_identity_version = 1u;
    header.destination_order.feature_axis_identity = 0x301u;
    header.provider_identity_low = 0x401u;
    header.capability_identity_low = 0x501u;
    header.logical_edge_count = 2u;
    header.dense_width = 64u;
    header.source_group_count = 1u;
    header.destination_group_count = 1u;
    header.tile_count = 1u;
    header.compact_slot_count = 1u;
    header.residual_region_count = 1u;
    header.schedule_entry_count = 2u;
    header.value_map_count = 2u;
    header.source_group_offset = append_records(&image, &source, 1u);
    header.destination_group_offset = append_records(&image, &destination, 1u);
    header.tile_offset = append_records(&image, &tile, 1u);
    header.compact_slot_offset = append_records(&image, &slot, 1u);
    header.residual_region_offset = append_records(&image, &residual, 1u);
    header.residual_row_offset_offset = append_records(&image, row_offsets, 2u);
    header.residual_column_index_offset = append_records(&image, columns, 1u);
    header.schedule_entry_offset = append_records(&image, schedules, 2u);
    header.value_map_offset = append_records(&image, maps, 2u);
    header.image_bytes = image.size();
    std::memcpy(image.data(), &header, sizeof(header));
    return image;
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    execution::persistent_axis_identity result{};
    result.header = {execution::biological_abi_version,
        execution::serialized_record_kind::persistent_axis_identity,
        sizeof(result)};
    result.domain = {seed + 1u, seed + 2u};
    result.order = {seed + 3u, seed + 4u};
    result.geometry = {seed + 5u, seed + 6u};
    result.partition = {seed + 7u, seed + 8u};
    return result;
}

} // namespace

int main() {
    std::vector<unsigned char> physical = physical_image();
    assert(projection::validate_physical_mma_hybrid_image_v1(
        physical.data(), physical.size()));

    persistence::execution_section_source physical_section{};
    persistence::execution_projection_source physical_projection{};
    assert(projection::make_physical_mma_hybrid_cpe2_source_v1(
        physical.data(), physical.size(), 0x51u, 0x52u, 0x61u, 0x62u, 4u,
        &physical_section, &physical_projection));
    assert(physical_projection.entry.kind ==
        persistence::execution_projection_kind::architecture_specific);

    const std::uint64_t foundational_data[] = {1u, 2u, 3u, 4u};
    persistence::execution_section_source sections[5]{};
    const persistence::execution_section_kind kinds[] = {
        persistence::execution_section_kind::domain_table,
        persistence::execution_section_kind::order_partition_table,
        persistence::execution_section_kind::relation_structure,
        persistence::execution_section_kind::semantic_geometry};
    for (std::uint32_t i = 0u; i < 4u; ++i) {
        sections[i].kind = kinds[i];
        sections[i].schema_version = 1u;
        sections[i].identity_low = 0x100u + i;
        sections[i].data = &foundational_data[i];
        sections[i].bytes = sizeof(foundational_data[i]);
        sections[i].element_count = 1u;
        sections[i].element_bytes = sizeof(foundational_data[i]);
    }
    sections[4] = physical_section;

    persistence::execution_image_v2_build_request request{};
    request.structure_identity = {0x101u, 0x102u};
    request.structure_epoch = 3u;
    request.semantic_geometry_identity = {0x201u, 0x202u};
    request.projection_catalog_identity = {0x301u, 0x302u};
    request.source_axis = axis(0x1000u);
    request.destination_axis = axis(0x2000u);
    request.sections = sections;
    request.section_count = 5u;
    request.projections = &physical_projection;
    request.projection_count = 1u;

    persistence::execution_image_v2_requirements required{};
    assert(persistence::query_execution_image_v2_requirements_host(
        request, &required));
    std::vector<unsigned char> cpe2(required.image_bytes);
    persistence::execution_image_v2_view built{};
    assert(persistence::build_execution_image_v2_host(request,
        {cpe2.data(), cpe2.size()}, &built));

    const persistence::execution_image_v2_expected expected{
        request.structure_identity, request.structure_epoch,
        request.semantic_geometry_identity, request.projection_catalog_identity, 0u};
    persistence::execution_image_v2_view loaded{};
    assert(persistence::validate_execution_image_v2_host(
        cpe2.data(), cpe2.size(), expected, &loaded));
    persistence::prebound_projection_view_v1 prebound{};
    assert(persistence::prebind_execution_projection_host(loaded, 0u, &prebound));
    assert(projection::validate_physical_mma_hybrid_image_v1(
        prebound.payload, prebound.payload_bytes));

    std::vector<unsigned char> corrupt_physical = physical;
    projection::physical_mma_hybrid_header_v1 bad_header{};
    std::memcpy(&bad_header, corrupt_physical.data(), sizeof(bad_header));
    bad_header.value_map_offset = bad_header.image_bytes;
    std::memcpy(corrupt_physical.data(), &bad_header, sizeof(bad_header));
    assert(!projection::validate_physical_mma_hybrid_image_v1(
        corrupt_physical.data(), corrupt_physical.size()));

    std::vector<unsigned char> corrupt_cpe2 = cpe2;
    const auto payload_offset = loaded.sections[4].offset;
    corrupt_cpe2[payload_offset + sizeof(
        projection::physical_mma_hybrid_header_v1)] ^= 1u;
    persistence::execution_image_v2_view rejected{};
    assert(!persistence::validate_execution_image_v2_host(
        corrupt_cpe2.data(), corrupt_cpe2.size(), expected, &rejected));
    return 0;
}
