#pragma once

#include <cstdint>
#include <type_traits>

#include <Cellerator/execution/index_space/hierarchical_index_space_v1.hh>

namespace cellerator::geometry {

using execution::hierarchical_index_space_view_v1;
using execution::local_index_space_view_v1;
using execution::local_index_width_v1;

// Non-owning array of component-local indices.  data points to count values
// whose scalar type is selected by width.
struct compact_index_array_view_v2 {
    const void *data = nullptr;
    std::uint64_t count = 0u;
    local_index_width_v1 width = local_index_width_v1::u32;
    std::uint8_t reserved[7]{};
};

// Exact recovery from a compact component-local position to an aggregate
// relation position.  global_identities is optional and distinguishes stable
// biological edge identity from aggregate position when required.
struct aggregate_index_map_view_v2 {
    std::uint64_t component_identity = 0u;
    std::uint64_t aggregate_begin = 0u;
    std::uint64_t local_count = 0u;
    const std::uint64_t *local_to_aggregate = nullptr;
    const std::uint64_t *global_identities = nullptr;
};

// Destination-owned sparse support for one independently bounded component.
// destination_offsets has destination_count + 1 entries in offset_width;
// source_indices has local_edge_count entries in source_width.  Both arrays
// are caller-owned and may use compact widths independently.
struct support_component_view_v2 {
    std::uint64_t component_identity = 0u;
    local_index_space_view_v1 source_space{};
    local_index_space_view_v1 destination_space{};
    const void *destination_offsets = nullptr;
    const void *source_indices = nullptr;
    std::uint64_t destination_count = 0u;
    std::uint64_t local_edge_count = 0u;
    local_index_width_v1 offset_width = local_index_width_v1::u32;
    local_index_width_v1 source_width = local_index_width_v1::u32;
    std::uint8_t reserved[6]{};
    aggregate_index_map_view_v2 edge_map{};
};

struct scalable_support_view_v2 {
    std::uint64_t relation_identity = 0u;
    std::uint64_t aggregate_edge_count = 0u;
    const support_component_view_v2 *components = nullptr;
    std::uint64_t component_count = 0u;
};

enum class cover_domain_v2 : std::uint8_t {
    semantic = 0u,
    physical = 1u,
};

// An exact cover lists component-local logical edges owned by one work item.
// A semantic cover and a physical mechanism cover use distinct arrays and
// identities; neither is inferred from the other.  No entry represents a
// deleted edge.
struct cover_work_item_view_v2 {
    std::uint64_t work_item_identity = 0u;
    std::uint64_t component_identity = 0u;
    compact_index_array_view_v2 local_edge_indices{};
};

struct scalable_cover_view_v2 {
    std::uint64_t cover_identity = 0u;
    std::uint64_t relation_identity = 0u;
    std::uint64_t aggregate_edge_count = 0u;
    const cover_work_item_view_v2 *work_items = nullptr;
    std::uint64_t work_item_count = 0u;
    cover_domain_v2 domain = cover_domain_v2::semantic;
    std::uint8_t reserved[7]{};
};

// Generic component-bounded work list.  payload is an array of item_count
// records with item_stride bytes per record.  Interpretation belongs to the
// operation/candidate contract, while aggregate_work_count remains 64-bit.
struct component_work_view_v2 {
    std::uint64_t component_identity = 0u;
    std::uint64_t aggregate_begin = 0u;
    const void *payload = nullptr;
    std::uint64_t item_count = 0u;
    std::uint32_t item_stride = 0u;
    std::uint32_t reserved = 0u;
};

struct scalable_work_view_v2 {
    std::uint64_t work_identity = 0u;
    std::uint64_t aggregate_work_count = 0u;
    const component_work_view_v2 *components = nullptr;
    std::uint64_t component_count = 0u;
};

static_assert(std::is_trivially_copyable_v<compact_index_array_view_v2>);
static_assert(std::is_trivially_copyable_v<aggregate_index_map_view_v2>);
static_assert(std::is_trivially_copyable_v<support_component_view_v2>);
static_assert(std::is_trivially_copyable_v<scalable_support_view_v2>);
static_assert(std::is_trivially_copyable_v<cover_work_item_view_v2>);
static_assert(std::is_trivially_copyable_v<scalable_cover_view_v2>);
static_assert(std::is_trivially_copyable_v<component_work_view_v2>);
static_assert(std::is_trivially_copyable_v<scalable_work_view_v2>);

}  // namespace cellerator::geometry
