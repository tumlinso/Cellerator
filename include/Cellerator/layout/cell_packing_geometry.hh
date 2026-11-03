#pragma once

#include <Cellerator/geometry/persistent_packing_payload.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::layout {

// Physical CP-BP packing metadata, not a claim about biological dependencies.
inline constexpr std::uint32_t cell_packing_geometry_schema_version = 1u;
inline constexpr std::uint32_t packing_row_nnz_bucket_count = 8u;
inline constexpr std::uint32_t packing_mask_bucket_count = 33u;

enum class PackingStatistic : std::uint64_t {
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

constexpr std::uint64_t statistic_mask(PackingStatistic statistic) noexcept {
    return static_cast<std::uint64_t>(statistic);
}

struct PackingStatisticsManifest {
    std::uint64_t hot_summary_mask = 0u;
    std::uint64_t cold_sidecar_mask = 0u;
    // These statistics require separately supplied meaning/data; packing
    // metadata alone cannot establish them.
    std::uint64_t requires_external_semantics_mask = 0u;
};

struct CellPackingHotSummary {
    std::uint32_t row_count = 0u, feature_count = 0u, feature_block_count = 0u;
    std::uint32_t row_group_count = 0u, tile_count = 0u, tile_block_count = 0u;
    std::uint32_t maximum_row_nnz = 0u, heavy_row_count = 0u;
    std::uint32_t dense_fragment_candidate_count = 0u, reserved = 0u;
    std::uint64_t nnz_count = 0u, projection_metadata_bytes = 0u, projection_value_bytes = 0u;
    double mean_row_nnz = 0.0, mean_block_occupancy = 0.0, mean_feature_reuse = 0.0;
    double mean_lane_imbalance = 0.0, metadata_to_value_ratio = 0.0;
};

struct CellPackingColdSidecar {
    std::uint64_t available_mask = 0u;
    std::uint64_t row_nnz_histogram[packing_row_nnz_bucket_count]{};
    std::uint64_t row_mask_popcount_histogram[packing_mask_bucket_count]{};
    std::uint64_t feature_mask_popcount_histogram[packing_mask_bucket_count]{};
    double partial_block_occupancy_sum = 0.0;
    std::uint64_t partial_block_sample_count = 0u;
};

// Physical maps and image identities only. Runtime axis/structure/value
// associations are separately supplied by execution::CellPackingBinding.
struct CellPackingGeometryView {
    std::uint32_t schema_version = cell_packing_geometry_schema_version;
    std::uint64_t v1_feature_block_geometry_identity = 0u, v1_ordering_identity = 0u;
    std::uint64_t v1_payload_identity = 0u, cost_policy_identity = 0u;
    cellpack::packing_exact_objective_kind v1_objective_kind =
        cellpack::packing_exact_objective_kind::row_active_block_references;
    std::uint32_t row_count = 0u, feature_count = 0u, feature_block_count = 0u, row_group_count = 0u;
    const std::uint32_t *feature_block_offsets = nullptr, *feature_to_block = nullptr, *feature_to_local = nullptr;
    const std::uint32_t *row_group_offsets = nullptr;
    const std::uint32_t *execution_to_canonical_feature = nullptr, *canonical_to_execution_feature = nullptr;
    const std::uint32_t *execution_to_canonical_row = nullptr, *canonical_to_execution_row = nullptr;
};

// Borrowed host-resident physical data; pointer arrays must have their declared
// capacities. No runtime handle or scientific interpretation is required.
struct CellPackingStatisticsSource {
    cellpack::feature_weighted_row_reduction_plan_view plan{};
    cellpack::local_cell_order_view order{};
    cellpack::warp_tile_view tiles{};
    std::uint32_t row_group_count = 0u;
};

PackingStatisticsManifest cell_packing_statistics_manifest() noexcept;
cellpack::validation_result evaluate_cell_packing_statistics_host(
    const CellPackingStatisticsSource&, CellPackingHotSummary*, CellPackingColdSidecar*) noexcept;

static_assert(std::is_trivially_copyable<CellPackingGeometryView>::value,
              "physical packing views must remain pointer-copyable");
static_assert(std::is_trivially_copyable<CellPackingStatisticsSource>::value,
              "physical statistics sources must remain borrowed views");
} // namespace cellerator::layout
