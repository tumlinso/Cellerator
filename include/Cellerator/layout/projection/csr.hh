#pragma once

#include <Cellerator/layout/projection/identity.hh>

#include <Cellerator/geometry/apply_plan.hh>
#include <Cellerator/geometry/persistent_packing_payload.hh>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cellerator::layout::projection {

inline constexpr u32 execution_csr_schema_version = 1u;
inline constexpr u32 execution_csr_structure_identity_version = 1u;

enum class physical_view_status_code : u32 {
    ok = 0u,
    invalid_argument = 1u,
    incompatible_identity = 2u,
    invalid_geometry = 3u,
    insufficient_capacity = 4u,
    overflow = 5u,
    unsupported_value_size = 6u,
    cuda_failure = 7u
};

struct physical_view_status {
    physical_view_status_code code = physical_view_status_code::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == physical_view_status_code::ok;
    }
};

// Derived row-compressed view in frozen execution-feature order. Ordered-plan
// adaptation aliases existing row offsets and values; lazy CPK1 materialization
// uses caller-owned buffers and never changes or duplicates the durable image.
struct CsrViewV1 {
    u32 schema_version = execution_csr_schema_version;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    u32 value_size_bytes = 0u;
    u64 row_domain_identity = 0u;
    SparseStructureIdentity structure{};
    FeatureOrderIdentity feature_order{};
    const u32 *row_offsets = nullptr;
    const u32 *execution_feature_ids = nullptr;
    const void *values = nullptr;
};

struct execution_csr_feature_buffers {
    std::size_t execution_feature_capacity = 0u;
    u32 *execution_feature_ids = nullptr;
};

struct lazy_execution_csr_requirements {
    std::size_t row_offset_count = 0u;
    std::size_t execution_feature_count = 0u;
    std::size_t value_bytes = 0u;
    std::size_t row_cursor_count = 0u;
    std::size_t total_bytes = 0u;
};

struct lazy_execution_csr_buffers {
    std::size_t row_offset_capacity = 0u;
    std::size_t execution_feature_capacity = 0u;
    std::size_t value_capacity_bytes = 0u;
    std::size_t row_cursor_capacity = 0u;
    u32 *row_offsets = nullptr;
    u32 *execution_feature_ids = nullptr;
    void *values = nullptr;
    u32 *row_cursors = nullptr;
};

namespace physical_csr_detail {

inline constexpr u64 fnv_offset = 1469598103934665603ull;
inline constexpr u64 fnv_prime = 1099511628211ull;

inline void hash_u64(u64 *hash, u64 value) noexcept {
    for (u32 byte = 0u; byte < 8u; ++byte) {
        *hash ^= (value >> (byte * 8u)) & 0xffu;
        *hash *= fnv_prime;
    }
}

inline u64 structure_identity(
    u64 global_row_begin,
    u64 row_domain_identity,
    u32 rows,
    u32 features,
    u32 nnz,
    const u32 *row_offsets,
    const u32 *feature_ids) noexcept {
    u64 hash = fnv_offset;
    hash_u64(&hash, execution_csr_structure_identity_version);
    hash_u64(&hash, global_row_begin);
    hash_u64(&hash, row_domain_identity);
    hash_u64(&hash, rows);
    hash_u64(&hash, features);
    hash_u64(&hash, nnz);
    for (u32 i = 0u; i <= rows; ++i) hash_u64(&hash, row_offsets[i]);
    for (u32 i = 0u; i < nnz; ++i) hash_u64(&hash, feature_ids[i]);
    return hash == 0u ? 1u : hash;
}

inline FeatureOrderIdentity packed_order(
    u32 features,
    u64 axis,
    u32 axis_version,
    u64 geometry) noexcept {
    FeatureOrderIdentity order;
    order.kind = feature_order_kind::packed;
    order.feature_count = features;
    order.feature_axis_identity_version = axis_version;
    order.feature_axis_identity = axis;
    order.packing_geometry_identity = geometry;
    return order;
}

inline bool valid_plan(
    const cellpack::feature_weighted_row_reduction_plan_view &plan) noexcept {
    return plan.semantic_plan_schema_version
            == cellpack::packing_plan_semantic_schema_version
        && plan.geometry_identity_version
            == cellpack::feature_block_geometry_identity_version
        && plan.feature_block_geometry_identity != 0u
        && (plan.feature_count == 0u || plan.feature_permutation != nullptr)
        && plan.feature_block_count != 0u
        && plan.feature_block_offsets != nullptr
        && plan.feature_block_offsets[0] == 0u
        && plan.feature_block_offsets[plan.feature_block_count] == plan.feature_count;
}

inline bool valid_payload_metadata(
    const cellpack::persistent_packing_payload_view &payload) noexcept {
    const auto &tiles = payload.tiles;
    return payload.payload_schema_version == cellpack::persistent_packing_payload_schema_version
        && payload.payload_kind == cellpack::persistent_packing_payload_kind
        && payload.payload_identity != 0u && valid_plan(payload.plan)
        && cellpack::packed_extents_fit(tiles)
        && tiles.layout.features == payload.plan.feature_count
        && tiles.layout.feature_blocks == payload.plan.feature_block_count
        && tiles.feature_block_geometry_identity
            == payload.plan.feature_block_geometry_identity
        && tiles.layout.tile_row_width != 0u && tiles.layout.tile_row_width <= 32u
        && tiles.layout.tiles == tiles.layout.rows / tiles.layout.tile_row_width
            + (tiles.layout.rows % tiles.layout.tile_row_width != 0u ? 1u : 0u)
        && tiles.value_size_bytes != 0u
        && tiles.layout.tile_block_offsets != nullptr
        && tiles.layout.block_row_entry_offsets != nullptr
        && tiles.layout.row_block_value_offsets != nullptr
        && tiles.layout.tile_block_offsets[tiles.layout.tiles] == tiles.layout.tile_blocks
        && tiles.layout.block_row_entry_offsets[tiles.layout.tile_blocks]
            == tiles.layout.row_block_entries
        && tiles.layout.row_block_value_offsets[tiles.layout.row_block_entries]
            == tiles.layout.stored_values
        && (tiles.layout.tile_blocks == 0u
            || (tiles.layout.block_ids != nullptr
                && tiles.layout.participating_row_masks != nullptr))
        && (tiles.layout.row_block_entries == 0u
            || tiles.layout.feature_masks != nullptr)
        && (tiles.layout.stored_values == 0u || tiles.values != nullptr);
}

} // namespace physical_csr_detail

physical_view_status build_execution_csr_view_host(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const cellpack::ordered_plan_partition_view &ordered,
    const execution_csr_feature_buffers &buffers,
    CsrViewV1 *out) noexcept;

inline physical_view_status query_lazy_execution_csr_requirements(
    const cellpack::persistent_packing_payload_view &payload,
    lazy_execution_csr_requirements *out) noexcept {
    if (out == nullptr || !physical_csr_detail::valid_payload_metadata(payload)) {
        return {physical_view_status_code::invalid_argument,
            "lazy CSR requirements need a validated host CPK1 view"};
    }
    const auto &tiles = payload.tiles;
    if (tiles.layout.stored_values != 0u
        && tiles.value_size_bytes > std::numeric_limits<std::size_t>::max()
            / tiles.layout.stored_values) {
        return {physical_view_status_code::overflow,
            "lazy CSR value byte count overflows"};
    }
    lazy_execution_csr_requirements result;
    result.row_offset_count = static_cast<std::size_t>(tiles.layout.rows) + 1u;
    result.execution_feature_count = tiles.layout.stored_values;
    result.value_bytes = static_cast<std::size_t>(tiles.layout.stored_values)
        * tiles.value_size_bytes;
    result.row_cursor_count = tiles.layout.rows;
    const std::size_t index_count = result.row_offset_count
        + result.execution_feature_count + result.row_cursor_count;
    if (index_count > (std::numeric_limits<std::size_t>::max() - result.value_bytes)
            / sizeof(u32)) {
        return {physical_view_status_code::overflow,
            "lazy CSR total byte count overflows"};
    }
    result.total_bytes = index_count * sizeof(u32) + result.value_bytes;
    *out = result;
    return {};
}

// `payload` must be a validated host CPK1 view. Reconstruction is deliberately
// opt-in and caller-owned; normal native-tile consumers continue using CPK1.
physical_view_status materialize_execution_csr_from_cpk1_host(
    const cellpack::persistent_packing_payload_view &payload,
    const lazy_execution_csr_buffers &buffers,
    CsrViewV1 *out) noexcept;

static_assert(std::is_trivially_copyable<CsrViewV1>::value,
    "execution CSR view must remain pointer-copyable");

} // namespace cellerator::layout::projection
