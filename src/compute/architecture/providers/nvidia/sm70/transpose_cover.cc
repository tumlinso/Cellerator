#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class transpose_cover_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    insufficient_capacity = 2u,
    duplicate_logical_edge = 3u
};

struct logical_relation_edge_v1 {
    std::uint64_t logical_edge_id = 0u;
    std::uint32_t source_index = 0u;
    std::uint32_t destination_index = 0u;
};

// One record is a target-local placement, not a second semantic edge. Forward
// and transpose arrays may choose different group coordinates while retaining
// the same logical_edge_id for value-map recovery and gradient accumulation.
struct target_edge_placement_v1 {
    projection::width_tagged_logical_edge_id_v1 logical_edge_id{};
    projection::physical_region_kind_v1 region_kind =
        projection::physical_region_kind_v1::residual;
    std::uint8_t row = 0u;
    std::uint8_t column = 0u;
    std::uint8_t reserved0 = 0u;
    std::uint32_t source_group = 0u;
    std::uint32_t destination_group = 0u;
};

struct transpose_cover_request_v1 {
    const logical_relation_edge_v1 *logical_edges = nullptr;
    std::uint64_t logical_edge_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    projection::logical_edge_id_width_v1 logical_edge_id_width =
        projection::logical_edge_id_width_v1::u32;
    target_edge_placement_v1 *forward = nullptr;
    std::uint64_t forward_capacity = 0u;
    target_edge_placement_v1 *transpose = nullptr;
    std::uint64_t transpose_capacity = 0u;
};

namespace {

constexpr std::uint32_t group_extent(
    std::uint32_t count, std::uint32_t group) noexcept {
    const std::uint32_t begin = group * projection::mma_group_extent_limit_v1;
    const std::uint32_t remaining = count - begin;
    return remaining < projection::mma_group_extent_limit_v1
        ? remaining : projection::mma_group_extent_limit_v1;
}

target_edge_placement_v1 place_edge(const logical_relation_edge_v1 &edge,
    std::uint32_t source_count, std::uint32_t destination_count,
    projection::logical_edge_id_width_v1 width, bool transpose) noexcept {
    const std::uint32_t source =
        transpose ? edge.destination_index : edge.source_index;
    const std::uint32_t destination =
        transpose ? edge.source_index : edge.destination_index;
    const std::uint32_t oriented_source_count =
        transpose ? destination_count : source_count;
    const std::uint32_t oriented_destination_count =
        transpose ? source_count : destination_count;
    target_edge_placement_v1 placement{};
    placement.logical_edge_id.value = edge.logical_edge_id;
    placement.logical_edge_id.width = width;
    placement.source_group =
        source / projection::mma_group_extent_limit_v1;
    placement.destination_group =
        destination / projection::mma_group_extent_limit_v1;
    placement.row = static_cast<std::uint8_t>(
        destination % projection::mma_group_extent_limit_v1);
    placement.column = static_cast<std::uint8_t>(
        source % projection::mma_group_extent_limit_v1);
    const bool full_source = group_extent(oriented_source_count,
        placement.source_group) == projection::mma_group_extent_limit_v1;
    const bool full_destination = group_extent(oriented_destination_count,
        placement.destination_group) == projection::mma_group_extent_limit_v1;
    placement.region_kind = full_source && full_destination
        ? projection::physical_region_kind_v1::mma
        : projection::physical_region_kind_v1::residual;
    return placement;
}

} // namespace

transpose_cover_status_v1 build_transpose_cover_v1(
    const transpose_cover_request_v1 &request) noexcept {
    if (request.logical_edges == nullptr || request.logical_edge_count == 0u
        || request.source_count == 0u || request.destination_count == 0u
        || !projection::valid_logical_edge_id_width_v1(
            request.logical_edge_id_width)
        || request.forward == nullptr || request.transpose == nullptr
        || request.forward == request.transpose)
        return transpose_cover_status_v1::invalid_argument;
    if (request.logical_edge_count > request.forward_capacity
        || request.logical_edge_count > request.transpose_capacity)
        return transpose_cover_status_v1::insufficient_capacity;

    for (std::uint64_t index = 0u; index < request.logical_edge_count;
        ++index) {
        const logical_relation_edge_v1 &edge = request.logical_edges[index];
        if (edge.logical_edge_id == 0u
            || (request.logical_edge_id_width
                    == projection::logical_edge_id_width_v1::u32
                && edge.logical_edge_id > 0xffffffffu)
            || edge.source_index >= request.source_count
            || edge.destination_index >= request.destination_count)
            return transpose_cover_status_v1::invalid_argument;
        for (std::uint64_t prior = 0u; prior < index; ++prior)
            if (request.logical_edges[prior].logical_edge_id
                == edge.logical_edge_id)
                return transpose_cover_status_v1::duplicate_logical_edge;
    }

    for (std::uint64_t index = 0u; index < request.logical_edge_count;
        ++index) {
        request.forward[index] = place_edge(request.logical_edges[index],
            request.source_count, request.destination_count,
            request.logical_edge_id_width, false);
        request.transpose[index] = place_edge(request.logical_edges[index],
            request.source_count, request.destination_count,
            request.logical_edge_id_width, true);
    }
    return transpose_cover_status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
