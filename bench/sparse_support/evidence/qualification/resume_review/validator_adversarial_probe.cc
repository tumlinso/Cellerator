#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace projection = cellerator::compute::projection;
namespace math = cellerator::compute::math;

namespace cellerator::compute::projection {
bool validate_physical_mma_hybrid_image_v1_reference(
    const void*, std::size_t) noexcept;
}

namespace {
template<class T>
T load(const std::vector<unsigned char>& image, std::uint64_t offset,
       std::uint64_t index = 0u) {
    T result{};
    std::memcpy(&result, image.data() + offset + index * sizeof(T), sizeof(T));
    return result;
}

template<class T>
void store(std::vector<unsigned char>* image, std::uint64_t offset,
           const T& value, std::uint64_t index = 0u) {
    std::memcpy(image->data() + offset + index * sizeof(T), &value, sizeof(T));
}

std::vector<unsigned char> build(const std::vector<std::uint32_t>& sources,
                                 const std::vector<std::uint32_t>& destinations,
                                 std::uint32_t entities) {
    projection::physical_mma_hybrid_build_request_v1 request{};
    request.structure_identity.identity_version = 1u;
    request.structure_identity.value = 0x123456789u;
    request.source_order.kind = math::feature_order_kind::canonical;
    request.source_order.feature_count = entities;
    request.source_order.feature_axis_identity_version = 1u;
    request.source_order.feature_axis_identity = 0x101u;
    request.destination_order.kind = math::feature_order_kind::canonical;
    request.destination_order.feature_count = entities;
    request.destination_order.feature_axis_identity_version = 1u;
    request.destination_order.feature_axis_identity = 0x202u;
    request.provider_identity_low = 0x301u;
    request.capability_identity_low = 0x401u;
    request.source_indices = sources.data();
    request.destination_indices = destinations.data();
    request.edge_count = sources.size();
    request.source_count = entities;
    request.destination_count = entities;
    request.dense_width = 64u;
    std::vector<unsigned char> image;
    assert(projection::build_physical_mma_hybrid_image_v1(request, &image)
        == projection::physical_image_build_status_v1::success);
    return image;
}

void check_duplicate_physical_slot_rejection() {
    auto image = build({0u, 0u, 1u}, {0u, 1u, 0u}, 2u);
    const auto header = load<projection::physical_mma_hybrid_header_v1>(image, 0u);
    assert(header.tile_count == 1u && header.compact_slot_count >= 2u);
    auto first = load<projection::mma_compact_slot_v1>(
        image, header.compact_slot_offset, 0u);
    auto second = load<projection::mma_compact_slot_v1>(
        image, header.compact_slot_offset, 1u);
    second.row = first.row;
    second.column = first.column;
    second.dense_slot = first.dense_slot;
    store(&image, header.compact_slot_offset, second, 1u);
    auto map = load<projection::projection_value_map_v1>(
        image, header.value_map_offset, second.logical_edge_index);
    map.projection_slot = first.dense_slot;
    store(&image, header.value_map_offset, map, second.logical_edge_index);

    const bool old_result = projection::validate_physical_mma_hybrid_image_v1_reference(
        image.data(), image.size());
    const bool new_result = projection::validate_physical_mma_hybrid_image_v1(
        image.data(), image.size());
    assert(old_result && !new_result);
}
} // namespace

int main() {
    check_duplicate_physical_slot_rejection();
    std::cout << "validator adversarial probe passed: duplicate_slot\n";
}
