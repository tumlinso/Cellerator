#pragma once

#include <Cellerator/execution/projection_value_plane/value_plane_v1.hh>

#include <type_traits>

namespace cellerator::execution::projection_value_plane {

inline constexpr u32 invalid_local_value_slot_v1 = UINT32_MAX;

// Global logical identity is 64 bit. Component and slot coordinates are compact
// local indices; projections with more slots must be segmented into components.
struct logical_value_location_v1 {
    u32 component_index = invalid_local_value_slot_v1;
    u32 local_slot = invalid_local_value_slot_v1;
};

struct logical_value_index_v1 {
    logical_value_location_v1 *locations = nullptr;
    u64 capacity = 0u;
};

struct canonical_value_buffer_v1 {
    void *values = nullptr;
    u64 value_bytes = 0u;
    u64 element_count = 0u;
    u32 element_bytes = 0u;
    u32 reserved = 0u;
    order_id logical_order{};
    value_generation generation{};
    device_location location{};
};

struct dirty_logical_edges_v1 {
    const u64 *logical_edges = nullptr;
    u64 count = 0u;
    // Full imports use all_logical_edges. Otherwise the list must be strictly
    // increasing, making repeated or ambiguous dirty writes invalid.
    bool all_logical_edges = false;
    u8 reserved[7]{};
};

value_plane_status_v1 build_logical_value_index_v1(
    const projection_value_plane_v1 &plane,
    logical_value_index_v1 index) noexcept;

value_plane_status_v1 import_canonical_values_v1(
    const canonical_value_buffer_v1 &canonical,
    const logical_value_index_v1 &index,
    const dirty_logical_edges_v1 &dirty,
    projection_value_plane_v1 *plane) noexcept;

value_plane_status_v1 export_canonical_values_v1(
    const projection_value_plane_v1 &plane,
    const logical_value_index_v1 &index,
    canonical_value_buffer_v1 *canonical) noexcept;

static_assert(std::is_trivially_copyable<logical_value_location_v1>::value,
    "logical-to-projection locations must remain compact POD records");

}  // namespace cellerator::execution::projection_value_plane
