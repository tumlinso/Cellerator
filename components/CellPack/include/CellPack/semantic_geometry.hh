#pragma once

#include "CellPack/persistent_packing_payload.hh"

#include <Cellerator/execution/execution_contract.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellpack {

inline constexpr u32 cp_bp_v1_semantic_geometry_schema_version = 1u;
inline constexpr u32 cp_bp_v1_compatibility_adapter_schema_version = 1u;
inline constexpr u32 semantic_geometry_row_nnz_bucket_count = 8u;
inline constexpr u32 semantic_geometry_mask_bucket_count = 33u;

enum class semantic_statistic : u64 {
    row_nnz = 1ull << 0u,
    block_occupancy = 1ull << 1u,
    module_occupancy = 1ull << 2u,
    row_mask_popcount = 1ull << 3u,
    feature_mask_popcount = 1ull << 4u,
    feature_reuse = 1ull << 5u,
    lane_imbalance = 1ull << 6u,
    metadata_value_ratio = 1ull << 7u,
    partial_block_occupancy = 1ull << 8u,
    dense_fragment_candidates = 1ull << 9u,
    heavy_rows = 1ull << 10u,
    forward_locality = 1ull << 11u,
    transpose_locality = 1ull << 12u,
    cross_partition_edges = 1ull << 13u,
    module_activation_frequency = 1ull << 14u,
    quantization_range = 1ull << 15u,
    quantization_outliers = 1ull << 16u
};

constexpr u64 statistic_mask(semantic_statistic statistic) noexcept {
    return static_cast<u64>(statistic);
}

struct semantic_statistics_manifest {
    u64 hot_summary_mask = 0u;
    u64 cold_sidecar_mask = 0u;
    u64 requires_external_semantics_mask = 0u;
};

struct semantic_geometry_hot_summary {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 feature_block_count = 0u;
    u32 row_group_count = 0u;
    u32 tile_count = 0u;
    u32 tile_block_count = 0u;
    u32 maximum_row_nnz = 0u;
    u32 heavy_row_count = 0u;
    u32 dense_fragment_candidate_count = 0u;
    u32 reserved = 0u;
    u64 nnz_count = 0u;
    u64 projection_metadata_bytes = 0u;
    u64 projection_value_bytes = 0u;
    double mean_row_nnz = 0.0;
    double mean_block_occupancy = 0.0;
    double mean_feature_reuse = 0.0;
    double mean_lane_imbalance = 0.0;
    double metadata_to_value_ratio = 0.0;
};

struct semantic_geometry_cold_sidecar {
    u64 available_mask = 0u;
    u64 row_nnz_histogram[semantic_geometry_row_nnz_bucket_count]{};
    u64 row_mask_popcount_histogram[semantic_geometry_mask_bucket_count]{};
    u64 feature_mask_popcount_histogram[semantic_geometry_mask_bucket_count]{};
    double partial_block_occupancy_sum = 0.0;
    u64 partial_block_sample_count = 0u;
};

struct cp_bp_v1_semantic_geometry_view {
    u32 schema_version = cp_bp_v1_semantic_geometry_schema_version;
    cellerator::execution::axis_identity row_axis{};
    cellerator::execution::axis_identity feature_axis{};
    u64 v1_feature_block_geometry_identity = 0u;
    u64 v1_ordering_identity = 0u;
    u64 v1_payload_identity = 0u;
    u64 cost_policy_identity = 0u;
    packing_exact_objective_kind v1_objective_kind =
        packing_exact_objective_kind::row_active_block_references;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 feature_block_count = 0u;
    u32 row_group_count = 0u;
    const u32 *feature_block_offsets = nullptr;
    const u32 *feature_to_block = nullptr;
    const u32 *feature_to_local = nullptr;
    const u32 *row_group_offsets = nullptr;
    const u32 *execution_to_canonical_feature = nullptr;
    const u32 *canonical_to_execution_feature = nullptr;
    const u32 *execution_to_canonical_row = nullptr;
    const u32 *canonical_to_execution_row = nullptr;
};

struct cp_bp_v1_adapter_request {
    cellerator::execution::axis_identity row_axis{};
    cellerator::execution::axis_identity feature_axis{};
    cellerator::execution::structure_handle structure{};
    cellerator::execution::structure_epoch structure_epoch_value{};
    cellerator::execution::projection_catalog_handle projection_catalog{};
    cellerator::execution::projection_handle projection{};
    cellerator::execution::value_generation value_generation_value{};
    cellerator::execution::device_location value_location{};
    cellerator::execution::value_numeric_policy numeric{};
};

// Every pointer aliases the validated v1 payload. The adapter owns no bytes and
// never thaws, mutates, decodes, or canonicalizes the source image.
struct cp_bp_v1_compatibility_adapter {
    u32 schema_version = cp_bp_v1_compatibility_adapter_schema_version;
    cp_bp_v1_semantic_geometry_view geometry{};
    cellerator::execution::relation_structure structure{};
    cellerator::execution::value_plane values{};
    cellerator::execution::projection_handle projection{};
    persistent_packing_payload_view payload{};
    feature_weighted_row_reduction_plan_view direct_plan{};
    local_cell_order_view direct_order{};
    warp_tile_view direct_tiles{};
};

semantic_statistics_manifest cp_bp_semantic_statistics_manifest() noexcept;

validation_result build_cp_bp_v1_compatibility_adapter_host(
    const persistent_packing_payload_view &payload,
    const cp_bp_v1_adapter_request &request,
    cp_bp_v1_compatibility_adapter *out) noexcept;

// CP-BP v1 stores rows and features physically, but its logical relation is
// always feature source -> row destination. Projection orientation must not
// redefine that relation or its stable logical edge identity.
validation_result validate_cp_bp_v1_compatibility_adapter_host(
    const cp_bp_v1_compatibility_adapter &adapter) noexcept;

validation_result evaluate_cp_bp_v1_semantic_statistics_host(
    const cp_bp_v1_compatibility_adapter &adapter,
    semantic_geometry_hot_summary *hot,
    semantic_geometry_cold_sidecar *cold) noexcept;

static_assert(std::is_trivially_copyable<cp_bp_v1_semantic_geometry_view>::value,
    "CP-BP semantic geometry view must remain pointer-copyable");
static_assert(std::is_trivially_copyable<cp_bp_v1_compatibility_adapter>::value,
    "CP-BP compatibility adapter must remain pointer-copyable");

} // namespace cellpack
