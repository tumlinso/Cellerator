#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Cellerator/layout/local_index_space.hh>

namespace cellerator::execution {

// One component of an aggregate relation.  aggregate_begin is a 64-bit
// logical offset; local indices remain compact and are interpreted only within
// this component.  component_identity is stable across relocation.
struct hierarchical_index_component_v1 {
    std::uint64_t component_identity = 0u;
    std::uint64_t aggregate_begin = 0u;
    layout::LocalIndexSpaceViewV1 index_space{};
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

static_assert(std::is_trivially_copyable_v<hierarchical_index_component_v1>);
static_assert(std::is_standard_layout_v<hierarchical_index_component_v1>);
static_assert(std::is_trivially_copyable_v<hierarchical_index_space_view_v1>);
static_assert(std::is_standard_layout_v<hierarchical_index_space_view_v1>);
static_assert(offsetof(hierarchical_index_component_v1, component_identity) == 0u);
static_assert(offsetof(hierarchical_index_component_v1, aggregate_begin) == 8u);
static_assert(offsetof(hierarchical_index_component_v1, index_space) == 16u);
static_assert(sizeof(hierarchical_index_component_v1)
              == ((16u + sizeof(layout::LocalIndexSpaceViewV1)
                   + alignof(hierarchical_index_component_v1) - 1u)
                  / alignof(hierarchical_index_component_v1))
                     * alignof(hierarchical_index_component_v1));
static_assert(offsetof(hierarchical_index_space_view_v1, relation_identity) == 0u);
static_assert(offsetof(hierarchical_index_space_view_v1, aggregate_extent) == 8u);
static_assert(offsetof(hierarchical_index_space_view_v1, components) == 16u);
static_assert(offsetof(hierarchical_index_space_view_v1, component_count) == 24u);
static_assert(sizeof(hierarchical_index_space_view_v1)
              == ((24u + sizeof(void *)
                   + alignof(hierarchical_index_space_view_v1) - 1u)
                  / alignof(hierarchical_index_space_view_v1))
                     * alignof(hierarchical_index_space_view_v1));

}  // namespace cellerator::execution
