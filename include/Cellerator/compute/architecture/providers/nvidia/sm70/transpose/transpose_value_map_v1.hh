#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_cover_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {

struct local_destination_dictionary_v1 {
    // Strictly increasing global identities; array position is the compact
    // projection-local destination index.
    const std::uint64_t *global_destination_ids = nullptr;
    std::uint64_t count = 0u;
};

struct projection_gradient_position_v1 {
    std::uint64_t logical_edge_id = 0u;
    std::uint64_t projection_position = 0u;
};

struct direct_gradient_order_v1 {
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::uint64_t projection_order_id = 0u;
    const projection_gradient_position_v1 *projection_order = nullptr;
    const std::uint64_t *logical_order_to_projection = nullptr;
    std::uint64_t edge_count = 0u;
};

struct transpose_local_map_request_v1 {
    transpose_cover_view_v1 cover{};
    local_destination_dictionary_v1 destinations{};
    // Cover positions ordered by strictly increasing logical edge identity.
    const std::uint64_t *identity_order = nullptr;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::uint64_t projection_order_id = 0u;
};

struct transpose_local_map_storage_v1 {
    transpose_edge_placement_v1 *placements = nullptr;
    std::uint64_t placement_capacity = 0u;
    projection_gradient_position_v1 *projection_order = nullptr;
    std::uint64_t projection_order_capacity = 0u;
    std::uint64_t *logical_order_to_projection = nullptr;
    std::uint64_t logical_order_capacity = 0u;
};

transpose_status_v1 bind_transpose_local_maps_v1(
    const transpose_local_map_request_v1 &request,
    const transpose_local_map_storage_v1 &storage,
    transpose_cover_view_v1 *bound_cover,
    direct_gradient_order_v1 *gradient_order) noexcept;

transpose_status_v1 validate_direct_gradient_order_v1(
    const direct_gradient_order_v1 &order) noexcept;

static_assert(std::is_trivially_copyable<projection_gradient_position_v1>::value,
    "gradient positions must remain pointer-free");
static_assert(std::is_trivially_copyable<direct_gradient_order_v1>::value,
    "gradient orders must remain non-owning views");

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
