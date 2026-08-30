#pragma once

#include <Cellerator/geometry/relation_cover.hh>
#include <Cellerator/geometry/work_layout.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::persistence {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 semantic_geometry_image_schema_version_v1 = 1u;
inline constexpr u32 semantic_geometry_image_kind_v1 = 0x43534731u; // CSG1
inline constexpr u32 semantic_geometry_image_alignment_v1 = 64u;
inline constexpr u32 semantic_geometry_image_header_bytes_v1 = 320u;
inline constexpr u32 semantic_geometry_section_entry_bytes_v1 = 64u;
inline constexpr u32 semantic_geometry_mandatory_section_count_v1 = 5u;
inline constexpr u32 semantic_geometry_first_optional_section_kind_v1 = 0x100u;

struct relation_identity_tag;
using relation_id = execution::persistent_identity<relation_identity_tag>;

enum class semantic_geometry_section_kind_v1 : u32 {
    work_window_members = 1u,
    execution_to_window = 2u,
    window_to_execution = 3u,
    semantic_components = 4u,
    logical_edge_ids = 5u
};

struct semantic_geometry_optional_section_v1 {
    u32 kind = 0u;
    u32 schema_version = 0u;
    u64 flags = 0u;
    u64 alignment = semantic_geometry_image_alignment_v1;
    const void *data = nullptr;
    u64 data_bytes = 0u;
};

struct semantic_geometry_image_build_request_v1 {
    relation_id relation{};
    execution::structure_id structure{};
    execution::structure_epoch structure_epoch{};
    execution::persistent_axis_identity source_axis{};
    execution::persistent_axis_identity destination_axis{};
    work_window_view_v1 work_window{};
    work_layout_view_v1 work_layout{};
    relation_cover_view_v1 relation_cover{};
    const semantic_geometry_optional_section_v1 *optional_sections = nullptr;
    u32 optional_section_count = 0u;
};

struct semantic_geometry_image_requirements_v1 {
    u64 image_bytes = 0u;
    u32 section_count = 0u;
    u32 reserved = 0u;
    u64 validation_workspace_bytes = 0u;
    u64 validation_workspace_alignment = 1u;
};

struct semantic_geometry_image_buffer_v1 {
    void *image = nullptr;
    u64 image_capacity = 0u;
};

struct semantic_geometry_image_validation_workspace_v1 {
    u8 *edge_marks = nullptr;
    u64 edge_mark_capacity = 0u;
};

// Relocated views contain no interior pointers. Every section is resolved from
// image_base plus a validated relative offset.
struct semantic_geometry_image_view_v1 {
    const void *image_base = nullptr;
    u64 image_bytes = 0u;
    execution::geometry_id geometry_identity{};
    relation_id relation{};
    execution::structure_id structure{};
    execution::structure_epoch structure_epoch{};
    execution::persistent_axis_identity source_axis{};
    execution::persistent_axis_identity destination_axis{};
    work_window_id work_window{};
    u64 logical_edge_count = 0u;
    u32 work_count = 0u;
    u32 component_count = 0u;
    u32 section_count = 0u;
    u32 reserved = 0u;
};

struct semantic_geometry_section_view_v1 {
    u32 kind = 0u;
    u32 schema_version = 0u;
    u64 flags = 0u;
    const void *data = nullptr;
    u64 data_bytes = 0u;
    u64 element_count = 0u;
    u32 element_bytes = 0u;
    u32 alignment = 0u;
    u64 checksum = 0u;
};

enum class semantic_geometry_image_status_v1 : u8 {
    ok = 0u,
    invalid_argument = 1u,
    invalid_work_window = 2u,
    invalid_work_layout = 3u,
    invalid_relation_cover = 4u,
    invalid_optional_section = 5u,
    arithmetic_overflow = 6u,
    insufficient_capacity = 7u,
    misaligned_image = 8u,
    invalid_format = 9u,
    invalid_section_directory = 10u,
    missing_mandatory_section = 11u,
    duplicate_section = 12u,
    section_out_of_bounds = 13u,
    section_checksum_mismatch = 14u,
    image_checksum_mismatch = 15u,
    geometry_identity_mismatch = 16u,
    invalid_semantic_data = 17u,
    insufficient_validation_workspace = 18u,
    incompatible_relocation = 19u,
    section_not_found = 20u
};

semantic_geometry_image_status_v1
query_semantic_geometry_image_requirements_v1(
    const semantic_geometry_image_build_request_v1 &request,
    semantic_geometry_image_requirements_v1 *requirements) noexcept;

semantic_geometry_image_status_v1 build_semantic_geometry_image_v1(
    const semantic_geometry_image_build_request_v1 &request,
    semantic_geometry_image_buffer_v1 buffer,
    semantic_geometry_image_validation_workspace_v1 validation_workspace,
    semantic_geometry_image_view_v1 *view) noexcept;

semantic_geometry_image_status_v1 validate_semantic_geometry_image_v1(
    const void *image,
    u64 image_bytes,
    semantic_geometry_image_validation_workspace_v1 validation_workspace,
    semantic_geometry_image_view_v1 *view) noexcept;

semantic_geometry_image_status_v1 rebind_semantic_geometry_image_v1(
    const semantic_geometry_image_view_v1 &validated_view,
    const void *new_image_base,
    u64 new_image_bytes,
    semantic_geometry_image_view_v1 *rebound_view) noexcept;

semantic_geometry_image_status_v1 find_semantic_geometry_section_v1(
    const semantic_geometry_image_view_v1 &validated_view,
    u32 section_kind,
    semantic_geometry_section_view_v1 *section) noexcept;

static_assert(
    std::is_trivially_copyable<semantic_geometry_optional_section_v1>::value,
    "CSG1 optional sections must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<semantic_geometry_image_build_request_v1>::value,
    "CSG1 build requests must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<semantic_geometry_image_view_v1>::value,
    "CSG1 relocated views must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<semantic_geometry_section_view_v1>::value,
    "CSG1 section views must remain pointer-copyable");

} // namespace cellerator::geometry::persistence
