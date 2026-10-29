#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace projection = cellerator::compute::projection;

namespace {

template<class T>
T read_record(const std::vector<std::uint8_t>& image,
              std::uint64_t offset, std::uint64_t index) {
    T result{};
    std::memcpy(&result, image.data() + offset + index * sizeof(T), sizeof(T));
    return result;
}

void builds_and_validates_dense_geometry_ordinals() {
    const std::uint32_t sources[]{0u, 1u, 0u, 17u, 2u};
    const std::uint32_t destinations[]{3u, 3u, 3u, 18u, 4u};
    projection::physical_mma_hybrid_build_request_v1 request{};
    request.structure_identity.identity_version = 1u;
    request.structure_identity.value = 0x842a91u;
    request.source_order.feature_count = 18u;
    request.source_order.feature_axis_identity_version = 1u;
    request.source_order.feature_axis_identity = 0x1001u;
    request.destination_order.feature_count = 20u;
    request.destination_order.feature_axis_identity_version = 1u;
    request.destination_order.feature_axis_identity = 0x1002u;
    request.provider_identity_low = 0x2001u;
    request.capability_identity_low = 0x3001u;
    request.source_indices = sources;
    request.destination_indices = destinations;
    request.edge_count = 5u;
    request.source_count = 18u;
    request.destination_count = 20u;

    std::vector<std::uint8_t> image;
    assert(projection::build_physical_mma_hybrid_image_v1(request, &image)
        == projection::physical_image_build_status_v1::success);
    assert(projection::validate_physical_mma_hybrid_image_v1(
        image.data(), image.size()));
    const auto header = read_record<projection::physical_mma_hybrid_header_v1>(
        image, 0u, 0u);
    assert(header.logical_edge_count == 5u);
    assert(header.dense_width == 64u);
    assert(header.source_group_count == 2u);
    assert(header.destination_group_count == 2u);
    assert(header.tile_count == 2u);
    assert(header.residual_region_count == 1u);
    assert(header.value_map_count == 5u);

    // Physical IDs describe canonical geometry ordinals, while the support
    // layer retains its independently stable IDs and their explicit mapping.
    for (std::uint64_t i = 0u; i < header.value_map_count; ++i) {
        const auto map = read_record<projection::projection_value_map_v1>(
            image, header.value_map_offset, i);
        assert(map.logical_edge_id.value < header.logical_edge_count);
    }
    const auto duplicate = read_record<projection::residual_region_v1>(
        image, header.residual_region_offset, 0u);
    assert(duplicate.edge_count == 1u);
    assert(duplicate.destination_group_index == 0u);
}

void rejects_bad_support_without_overwriting_output() {
    const std::uint32_t sources[]{0u};
    const std::uint32_t destinations[]{3u};
    projection::physical_mma_hybrid_build_request_v1 request{};
    request.structure_identity.identity_version = 1u;
    request.structure_identity.value = 1u;
    request.source_order.feature_count = 1u;
    request.source_order.feature_axis_identity_version = 1u;
    request.source_order.feature_axis_identity = 1u;
    request.destination_order.feature_count = 1u;
    request.destination_order.feature_axis_identity_version = 1u;
    request.destination_order.feature_axis_identity = 2u;
    request.provider_identity_low = 1u;
    request.capability_identity_low = 1u;
    request.source_indices = sources;
    request.destination_indices = destinations;
    request.edge_count = 1u;
    request.source_count = 1u;
    request.destination_count = 1u;
    const std::vector<std::uint8_t> sentinel{1u, 2u, 3u};
    auto image = sentinel;
    assert(projection::build_physical_mma_hybrid_image_v1(request, &image)
        == projection::physical_image_build_status_v1::invalid_argument);
    assert(image == sentinel);
}

} // namespace

int main() {
    builds_and_validates_dense_geometry_ordinals();
    rejects_bad_support_without_overwriting_output();
}
