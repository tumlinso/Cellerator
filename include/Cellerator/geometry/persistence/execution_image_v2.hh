#pragma once

#include "Cellerator/geometry/persistent_packing_payload.hh"

#include <Cellerator/execution/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellpack::persistence {

inline constexpr u32 execution_image_v2_schema_version = 2u;
inline constexpr u32 execution_image_v2_payload_kind = 0x43504532u; // "CPE2"
inline constexpr u32 execution_image_v2_alignment = 64u;
inline constexpr u32 execution_image_v2_endian_marker = 0x01020304u;
inline constexpr u32 invalid_directory_index = 0xffffffffu;

enum class execution_section_kind : u32 {
    domain_table = 1u,
    order_partition_table = 2u,
    relation_structure = 3u,
    semantic_geometry = 4u,
    initial_values = 5u,
    projection_payload = 6u,
    forward_value_map = 7u,
    transpose_value_map = 8u,
    hierarchy_partition_index = 9u,
    scheduling_summary = 10u,
    cpk1_v1_compatibility = 11u,
    extension = 0x80000000u
};

enum class execution_projection_kind : u32 {
    native_row_masked = 1u,
    native_feature_major = 2u,
    cta_macrotile = 3u,
    dense_fragment = 4u,
    csr = 5u,
    sell = 6u,
    bsr = 7u,
    blocked_ell = 8u,
    vendor_specific = 9u,
    transpose_backward = 10u,
    architecture_specific = 11u,
    extension = 0x80000000u
};

enum execution_directory_flags : u32 {
    directory_optional = 1u << 0u,
    directory_device_readable = 1u << 1u,
    projection_lazy_constructible = 1u << 2u,
    projection_forward_capable = 1u << 3u,
    projection_transpose_capable = 1u << 4u
};

// Fixed-width persistent axis record. These fields are semantic identities;
// pointer addresses and runtime-interned handles never enter the image.
struct execution_axis_record_v1 {
    u64 domain_low;
    u64 domain_high;
    u64 order_low;
    u64 order_high;
    u64 geometry_low;
    u64 geometry_high;
    u64 partition_low;
    u64 partition_high;
};

// The header and directories are cold load-time IR. Kernels consume only a
// prebound projection view and never parse this self-describing image.
struct execution_image_v2_header {
    unsigned char magic[8];
    u32 schema_version;
    u32 header_bytes;
    u32 endian;
    u32 alignment;
    u64 image_bytes;
    u64 image_identity;
    u64 structure_identity_low;
    u64 structure_identity_high;
    u64 structure_epoch;
    u64 semantic_geometry_identity_low;
    u64 semantic_geometry_identity_high;
    u64 projection_catalog_identity_low;
    u64 projection_catalog_identity_high;
    u64 initial_value_generation;
    execution_axis_record_v1 source_axis;
    execution_axis_record_v1 destination_axis;
    u32 section_count;
    u32 projection_count;
    u64 section_directory_offset;
    u64 projection_directory_offset;
};

struct execution_section_entry_v1 {
    execution_section_kind kind;
    u32 schema_version;
    u32 flags;
    u32 alignment;
    u64 identity_low;
    u64 identity_high;
    u64 offset;
    u64 bytes;
    u64 checksum;
    u32 element_count;
    u32 element_bytes;
};

struct execution_projection_entry_v1 {
    u64 identity_low;
    u64 identity_high;
    execution_projection_kind kind;
    u32 schema_version;
    u32 flags;
    u32 operation_family;
    std::uint16_t storage_type;
    std::uint16_t compute_type;
    std::uint16_t accumulation_type;
    std::uint16_t orientation;
    u32 architecture_class;
    u32 payload_section;
    u32 forward_map_section;
    u32 transpose_map_section;
    u32 scheduling_summary_section;
    u32 capability_section;
};

struct execution_section_source {
    execution_section_kind kind{};
    u32 schema_version = 0u;
    u32 flags = 0u;
    u32 alignment = execution_image_v2_alignment;
    u64 identity_low = 0u;
    u64 identity_high = 0u;
    const void *data = nullptr;
    std::size_t bytes = 0u;
    u32 element_count = 0u;
    u32 element_bytes = 0u;
};

struct execution_projection_source {
    execution_projection_entry_v1 entry{};
};

struct execution_image_v2_build_request {
    cellerator::execution::structure_id structure_identity{};
    u64 structure_epoch = 0u;
    cellerator::execution::geometry_id semantic_geometry_identity{};
    cellerator::execution::projection_id projection_catalog_identity{};
    u64 initial_value_generation = 0u;
    cellerator::execution::persistent_axis_identity source_axis{};
    cellerator::execution::persistent_axis_identity destination_axis{};
    const execution_section_source *sections = nullptr;
    u32 section_count = 0u;
    const execution_projection_source *projections = nullptr;
    u32 projection_count = 0u;
};

struct execution_image_v2_requirements {
    std::size_t image_bytes = 0u;
    std::size_t directory_bytes = 0u;
    std::size_t section_bytes = 0u;
    std::size_t alignment_padding_bytes = 0u;
};

struct execution_image_v2_buffer {
    void *image = nullptr;
    std::size_t capacity_bytes = 0u;
};

struct execution_image_v2_expected {
    cellerator::execution::structure_id structure_identity{};
    u64 structure_epoch = 0u;
    cellerator::execution::geometry_id semantic_geometry_identity{};
    cellerator::execution::projection_id projection_catalog_identity{};
    u64 image_identity = 0u;
};

struct execution_image_v2_view {
    execution_image_v2_header header{};
    const void *image_base = nullptr;
    std::size_t image_bytes = 0u;
    const execution_section_entry_v1 *sections = nullptr;
    const execution_projection_entry_v1 *projections = nullptr;
};

struct prebound_projection_view_v1 {
    execution_projection_entry_v1 descriptor{};
    const void *payload = nullptr;
    std::size_t payload_bytes = 0u;
    const void *forward_map = nullptr;
    std::size_t forward_map_bytes = 0u;
    const void *transpose_map = nullptr;
    std::size_t transpose_map_bytes = 0u;
    const void *scheduling_summary = nullptr;
    std::size_t scheduling_summary_bytes = 0u;
};

validation_result query_execution_image_v2_requirements_host(
    const execution_image_v2_build_request &request,
    execution_image_v2_requirements *out) noexcept;

validation_result build_execution_image_v2_host(
    const execution_image_v2_build_request &request,
    const execution_image_v2_buffer &buffer,
    execution_image_v2_view *out) noexcept;

validation_result validate_execution_image_v2_host(
    const void *image,
    std::size_t image_bytes,
    const execution_image_v2_expected &expected,
    execution_image_v2_view *out) noexcept;

// Rebinds a validated view to an equal-sized copy, including device memory.
// It performs no allocation, copy, image parsing, checksum, or dereference of
// the new address.
validation_result rebind_execution_image_v2(
    const execution_image_v2_view &validated_host_view,
    const void *new_image_base,
    std::size_t new_image_bytes,
    execution_image_v2_view *out) noexcept;

validation_result prebind_execution_projection_host(
    const execution_image_v2_view &validated_host_view,
    u32 projection_index,
    prebound_projection_view_v1 *out) noexcept;

// Reads only the already validated host directories. Hot pointers are formed
// from their validated section offsets plus destination_image_base, which may
// be a device address; the destination image is never dereferenced here.
validation_result prebind_execution_projection_for_base_host(
    const execution_image_v2_view &validated_host_view,
    u32 projection_index,
    const void *destination_image_base,
    std::size_t destination_image_bytes,
    prebound_projection_view_v1 *out) noexcept;

// CPK1 remains a combined v1 compatibility section. This validates it through
// the frozen v1 reader; it never converts the image to CSR, BELL, or v2 values.
validation_result load_cpk1_v1_compatibility_host(
    const execution_image_v2_view &validated_host_view,
    u32 projection_index,
    const persistent_packing_payload_compatibility &expected,
    persistent_packing_payload_view *out) noexcept;

static_assert(sizeof(execution_image_v2_header) == 256u,
    "execution image v2 header size is part of schema v2");
static_assert(sizeof(execution_section_entry_v1) == 64u,
    "execution section directory entry size is part of schema v2");
static_assert(sizeof(execution_projection_entry_v1) == 64u,
    "execution projection directory entry size is part of schema v2");
static_assert(std::is_trivially_copyable<execution_image_v2_header>::value,
    "execution image header must remain pointer-free");
static_assert(std::is_trivially_copyable<prebound_projection_view_v1>::value,
    "prebound projection view must remain device-copyable");

} // namespace cellpack::persistence
