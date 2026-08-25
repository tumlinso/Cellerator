#pragma once

#include "physical_csr.hh"

#include <Cellerator/execution/execution_contract.hh>

#include <CellPack/persistent_packing_payload.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math {

inline constexpr u32 feature_major_projection_schema_version = 1u;
inline constexpr u32 feature_major_projection_payload_kind = 0x464d5031u; // "FMP1"
inline constexpr u32 feature_major_projection_alignment = 64u;
inline constexpr u32 feature_major_projection_variant = 1u;
inline constexpr u32 feature_major_small_n_minimum = 1u;
inline constexpr u32 feature_major_small_n_maximum = 16u;

// Pointer-free payload for one feature-major projection of the frozen CP-BP
// geometry. A tile owns one row group. Its records are sorted by packed
// execution feature. Each record names participating rows with one 32-bit mask;
// compact values follow set row bits in increasing lane order. The source-value
// map is cold projection-construction metadata and is never traversed by the
// steady-state kernel.
struct feature_major_projection_payload_header {
    u32 schema_version = feature_major_projection_schema_version;
    u32 payload_kind = feature_major_projection_payload_kind;
    u32 header_bytes = sizeof(feature_major_projection_payload_header);
    u32 alignment = feature_major_projection_alignment;
    u64 payload_bytes = 0u;
    execution::structure_id structure_identity{};
    execution::projection_id projection_identity{};
    u64 structure_epoch = 0u;
    u64 source_payload_identity = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 ordering_identity = 0u;
    u64 global_row_begin = 0u;
    u64 row_domain_identity = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 tile_row_width = 0u;
    u32 tile_count = 0u;
    u32 feature_record_count = 0u;
    u32 nnz_count = 0u;
    u32 value_size_bytes = 0u;
    u32 reserved = 0u;
    u64 tile_feature_offsets_offset = 0u;
    u64 execution_feature_ids_offset = 0u;
    u64 participating_row_masks_offset = 0u;
    u64 feature_value_offsets_offset = 0u;
    u64 source_value_positions_offset = 0u;
};

struct feature_major_projection_requirements {
    std::size_t payload_bytes = 0u;
    std::size_t tile_feature_offset_count = 0u;
    std::size_t feature_record_count = 0u;
    std::size_t feature_value_offset_count = 0u;
    std::size_t source_value_position_count = 0u;
};

struct feature_major_projection_build_request {
    execution::structure_id structure_identity{};
    execution::structure_handle runtime_structure{};
    execution::structure_epoch structure_epoch_value{};
    execution::projection_id projection_identity{};
    execution::projection_handle runtime_projection{};
    cellpack::persistent_packing_payload_view source{};
};

struct feature_major_projection_buffer {
    void *payload = nullptr;
    std::size_t capacity_bytes = 0u;
};

// Hot typed view. Runtime handles are deliberately outside persistent bytes;
// they are rebound by the owning projection catalog/session.
struct feature_major_projection_view {
    feature_major_projection_payload_header header{};
    execution::structure_handle runtime_structure{};
    execution::projection_handle runtime_projection{};
    const void *payload_base = nullptr;
    const u32 *tile_feature_offsets = nullptr;
    const u32 *execution_feature_ids = nullptr;
    const u32 *participating_row_masks = nullptr;
    const u32 *feature_value_offsets = nullptr;
    const u32 *source_value_positions = nullptr;
};

struct feature_major_value_buffers {
    void *values = nullptr;
    std::size_t capacity_bytes = 0u;
};

physical_view_status query_feature_major_projection_requirements_host(
    const feature_major_projection_build_request &request,
    feature_major_projection_requirements *out) noexcept;

physical_view_status build_feature_major_projection_host(
    const feature_major_projection_build_request &request,
    const feature_major_projection_buffer &buffer,
    feature_major_projection_view *out) noexcept;

physical_view_status validate_feature_major_projection_payload_host(
    const void *payload,
    std::size_t payload_bytes,
    execution::structure_id expected_structure,
    execution::structure_epoch expected_epoch,
    execution::projection_id expected_projection,
    execution::structure_handle runtime_structure,
    execution::projection_handle runtime_projection,
    feature_major_projection_view *out) noexcept;

// Rebinds a host-validated payload to an equal-sized copy, including a device
// allocation. It performs no allocation, copy, parsing, or dereference of the
// destination address.
physical_view_status rebind_feature_major_projection(
    const feature_major_projection_view &validated_host_view,
    const void *new_payload_base,
    std::size_t new_payload_bytes,
    feature_major_projection_view *out) noexcept;

// Projection construction/value preparation helper. It reorders one source
// CPK1-local value generation into FMP1-local order using caller-owned host
// storage. This is never called from prepared execution.
physical_view_status pack_feature_major_values_host(
    const feature_major_projection_view &host_projection,
    const void *source_values,
    std::size_t source_value_bytes,
    const feature_major_value_buffers &buffers) noexcept;

static_assert(sizeof(feature_major_projection_payload_header) == 192u,
    "feature-major payload header is schema v1");
static_assert(std::is_trivially_copyable<feature_major_projection_payload_header>::value,
    "feature-major payload must remain pointer-free");
static_assert(std::is_trivially_copyable<feature_major_projection_view>::value,
    "feature-major view must remain pointer-copyable");

} // namespace cellerator::compute::math
