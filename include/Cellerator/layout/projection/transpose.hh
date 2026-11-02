#pragma once

#include <Cellerator/layout/projection/feature_major.hh>

#include <Cellerator/memory/workspace.hh>

#include <cstddef>
#include <type_traits>

namespace cellerator::layout::projection {

inline constexpr u32 transpose_projection_schema_version = 1u;
inline constexpr u32 transpose_projection_payload_kind = 0x43545031u; // "CTP1"
inline constexpr u32 transpose_projection_alignment = 64u;
inline constexpr u32 transpose_projection_variant = 1u;

// Pointer-free feature-row transpose of FMP1. Values remain owned by the
// forward projection's mutable value plane; forward_value_positions maps each
// transpose edge to that plane. The two logical maps preserve stable edge
// identity independently of either physical traversal.
struct TransposeMapV1 {
    u32 schema_version = transpose_projection_schema_version;
    u32 payload_kind = transpose_projection_payload_kind;
    u32 header_bytes = sizeof(TransposeMapV1);
    u32 alignment = transpose_projection_alignment;
    u64 payload_bytes = 0u;
    ::cellerator::structure::structure_id structure_identity{};
    ::cellerator::structure::projection_id projection_identity{};
    ::cellerator::structure::projection_id forward_projection_identity{};
    u64 structure_epoch = 0u;
    u64 source_payload_identity = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 ordering_identity = 0u;
    u64 row_domain_identity = 0u;
    u64 feature_axis_fingerprint = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    u32 value_size_bytes = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u64 feature_offsets_offset = 0u;
    u64 execution_row_ids_offset = 0u;
    u64 forward_value_positions_offset = 0u;
    u64 logical_to_transpose_offset = 0u;
    u64 transpose_to_logical_offset = 0u;
};

struct transpose_projection_requirements {
    std::size_t payload_bytes = 0u;
    std::size_t feature_offset_count = 0u;
    std::size_t edge_count = 0u;
    // CTP1 construction uses its output offset section as a reversible cursor
    // table, so prepared external scratch is exactly zero bytes.
    cellerator::memory::workspace_requirement construction_workspace{};
};

struct transpose_projection_build_request {
    ::cellerator::structure::projection_id projection_identity{};
    FeatureMajorViewV1 forward{};
};

struct transpose_projection_buffer {
    void *payload = nullptr;
    std::size_t capacity_bytes = 0u;
};

struct TransposeViewV1 {
    TransposeMapV1 header{};
    const void *payload_base = nullptr;
    const u32 *feature_offsets = nullptr;
    const u32 *execution_row_ids = nullptr;
    const u32 *forward_value_positions = nullptr;
    const u32 *logical_to_transpose = nullptr;
    const u32 *transpose_to_logical = nullptr;
};

physical_view_status query_transpose_projection_requirements_host(
    const transpose_projection_build_request &request,
    transpose_projection_requirements *out) noexcept;

physical_view_status build_transpose_projection_host(
    const transpose_projection_build_request &request,
    const transpose_projection_buffer &buffer,
    TransposeViewV1 *out) noexcept;

physical_view_status validate_transpose_projection_payload_host(
    const void *payload,
    std::size_t payload_bytes,
    ::cellerator::structure::structure_id expected_structure,
    ::cellerator::structure::structure_epoch expected_epoch,
    ::cellerator::structure::projection_id expected_projection,
    ::cellerator::structure::projection_id expected_forward_projection,
    TransposeViewV1 *out) noexcept;

physical_view_status rebind_transpose_projection(
    const TransposeViewV1 &validated_host_view,
    const void *new_payload_base,
    std::size_t new_payload_bytes,
    TransposeViewV1 *out) noexcept;

static_assert(sizeof(TransposeMapV1) == 192u,
    "transpose payload header is schema v1");
static_assert(std::is_trivially_copyable<TransposeMapV1>::value,
    "transpose payload must remain pointer-free");
static_assert(std::is_trivially_copyable<TransposeViewV1>::value,
    "transpose view must remain pointer-copyable");

} // namespace cellerator::layout::projection
