#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::execution {

// Width of an index into one independently bounded component.  The width is
// part of the prepared contract; launch code must not infer it from a global
// relation extent.
enum class local_index_width_v1 : std::uint8_t {
    u16 = 2u,
    u32 = 4u,
    u64 = 8u,
};

// Pointer-first description of one local index space.  local_to_global has
// local_extent uint64 entries and recovers the global coordinate for every
// local position.  global_identity_sidecar is optional; when present it has
// local_extent uint64 entries and carries an identity distinct from position.
// Neither pointer is owned by this view.
struct local_index_space_view_v1 {
    std::uint64_t global_extent = 0u;
    std::uint64_t partition_identity = 0u;
    std::uint64_t local_extent = 0u;
    const std::uint64_t *local_to_global = nullptr;
    const std::uint64_t *global_identity_sidecar = nullptr;
    local_index_width_v1 local_width = local_index_width_v1::u32;
    std::uint8_t reserved[7]{};
};

// One component of an aggregate relation.  aggregate_begin is a 64-bit
// logical offset; local indices remain compact and are interpreted only within
// this component.  component_identity is stable across relocation.
struct hierarchical_index_component_v1 {
    std::uint64_t component_identity = 0u;
    std::uint64_t aggregate_begin = 0u;
    local_index_space_view_v1 index_space{};
};

// A relation may exceed a single kernel-local index range by containing an
// arbitrary number of independently indexable components.  The component
// array and all maps are caller-owned.  aggregate_extent describes logical
// work and does not require a physically contiguous aggregate allocation.
struct hierarchical_index_space_view_v1 {
    std::uint64_t relation_identity = 0u;
    std::uint64_t aggregate_extent = 0u;
    const hierarchical_index_component_v1 *components = nullptr;
    std::uint64_t component_count = 0u;
};

static_assert(std::is_trivially_copyable_v<local_index_space_view_v1>);
static_assert(std::is_standard_layout_v<local_index_space_view_v1>);
static_assert(std::is_trivially_copyable_v<hierarchical_index_component_v1>);
static_assert(std::is_standard_layout_v<hierarchical_index_component_v1>);
static_assert(std::is_trivially_copyable_v<hierarchical_index_space_view_v1>);
static_assert(std::is_standard_layout_v<hierarchical_index_space_view_v1>);

}  // namespace cellerator::execution
