#pragma once

#include <Cellerator/execution/biological_abi.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 relation_cover_schema_version = 1u;
inline constexpr u32 invalid_semantic_component_id = 0u;
inline constexpr u32 invalid_semantic_component_index = ~u32{0u};
inline constexpr u64 invalid_logical_edge_id = ~u64{0u};

// These kinds describe portable semantic organization only. They never imply a
// physical tile, instruction, residual format, padding rule, or device target.
enum class semantic_component_kind : u8 {
    unstructured = 1u,
    rectangular = 2u,
    hierarchical = 3u
};

// Each component owns one nonempty contiguous slice of the cover's logical
// edge-ID array. component_id is cover-local and stable within the artifact.
struct semantic_component_v1 {
    u32 component_id = invalid_semantic_component_id;
    semantic_component_kind kind = semantic_component_kind::unstructured;
    u8 reserved[3]{};
    u64 logical_edge_offset = 0u;
    u64 logical_edge_count = 0u;
};

// The logical-edge IDs form an exact disjoint permutation of [0, E). The
// component slices partition that array without gaps. This is a cold data view;
// all pointed-to storage remains caller-owned.
struct relation_cover_view_v1 {
    u32 schema_version = relation_cover_schema_version;
    u32 reserved = 0u;
    execution::structure_handle structure{};
    execution::structure_epoch structure_epoch{};
    execution::axis_identity source_axis{};
    execution::axis_identity destination_axis{};
    u64 logical_edge_count = 0u;
    u32 component_count = 0u;
    u32 reserved2 = 0u;
    const semantic_component_v1 *components = nullptr;
    const u64 *logical_edge_ids = nullptr;
};

// Exact ownership validation is linear in E and uses one caller-owned byte per
// logical edge. The validator clears the first logical_edge_count bytes.
struct relation_cover_validation_workspace {
    u8 *edge_marks = nullptr;
    u64 edge_mark_capacity = 0u;
};

enum class relation_cover_validation_code : u8 {
    ok = 0u,
    unsupported_version = 1u,
    nonzero_reserved = 2u,
    invalid_structure = 3u,
    invalid_source_axis = 4u,
    invalid_destination_axis = 5u,
    invalid_component_count = 6u,
    missing_components = 7u,
    missing_logical_edge_ids = 8u,
    missing_workspace = 9u,
    insufficient_workspace = 10u,
    invalid_component_id = 11u,
    duplicate_component_id = 12u,
    invalid_component_kind = 13u,
    nonzero_component_reserved = 14u,
    empty_component = 15u,
    component_offset_mismatch = 16u,
    component_edge_range_overflow = 17u,
    incomplete_component_partition = 18u,
    logical_edge_out_of_bounds = 19u,
    duplicate_logical_edge = 20u,
    missing_logical_edge = 21u
};

struct relation_cover_validation_result {
    relation_cover_validation_code code = relation_cover_validation_code::ok;
    u32 component_index = invalid_semantic_component_index;
    u64 logical_edge_id = invalid_logical_edge_id;

    constexpr explicit operator bool() const noexcept {
        return code == relation_cover_validation_code::ok;
    }
};

constexpr bool valid_semantic_component_kind(
    semantic_component_kind kind) noexcept {
    return kind == semantic_component_kind::unstructured
        || kind == semantic_component_kind::rectangular
        || kind == semantic_component_kind::hierarchical;
}

relation_cover_validation_result validate_relation_cover(
    const relation_cover_view_v1 &cover,
    relation_cover_validation_workspace workspace) noexcept;

static_assert(std::is_trivially_copyable<semantic_component_v1>::value,
    "semantic components must remain pointer-copyable");
static_assert(std::is_trivially_copyable<relation_cover_view_v1>::value,
    "relation-cover views must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<relation_cover_validation_workspace>::value,
    "relation-cover validation workspace must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<relation_cover_validation_result>::value,
    "relation-cover validation results must remain trivially copyable");

} // namespace cellerator::geometry
