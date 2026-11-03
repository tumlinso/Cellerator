#include "Cellerator/layout/cell_packing_geometry.hh"

#include <algorithm>
#include <limits>

namespace cellerator::layout {
using namespace cellpack;
namespace {
u32 popcount(u32 value) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<u32>(__builtin_popcount(value));
#else
    u32 count = 0u;
    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
#endif
}

u32 row_nnz_bucket(u32 count) noexcept {
    if (count < 2u) return count;
    if (count < 4u) return 2u;
    if (count < 8u) return 3u;
    if (count < 16u) return 4u;
    if (count < 32u) return 5u;
    if (count < 64u) return 6u;
    return 7u;
}

u64 metadata_bytes(const warp_tile_view &tiles) noexcept {
    const u64 words = static_cast<u64>(tiles.tile_count) + 1u
        + 2u * tiles.tile_block_count
        + static_cast<u64>(tiles.tile_block_count) + 1u
        + tiles.row_block_entry_count
        + static_cast<u64>(tiles.row_block_entry_count) + 1u;
    return words * sizeof(u32);
}

bool valid_statistics_source(const CellPackingStatisticsSource &source) noexcept {
    const auto &plan = source.plan;
    const auto &order = source.order;
    const auto &tiles = source.tiles;
    if (plan.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || plan.geometry_identity_version != feature_block_geometry_identity_version
        || plan.feature_count == 0u || plan.feature_block_count == 0u
        || plan.feature_block_geometry_identity == 0u || plan.feature_block_offsets == nullptr
        || order.order_schema_version != local_cell_order_schema_version
        || order.ordering_identity == 0u || order.row_count == 0u
        || order.row_nnz_counts == nullptr || source.row_group_count == 0u
        || tiles.tile_schema_version != warp_tile_schema_version || tiles.tile_identity == 0u
        || tiles.ordering_identity != order.ordering_identity
        || tiles.feature_block_geometry_identity != plan.feature_block_geometry_identity
        || tiles.row_count != order.row_count || tiles.feature_count != plan.feature_count
        || tiles.feature_block_count != plan.feature_block_count || tiles.value_size_bytes == 0u)
        return false;
    if ((tiles.tile_block_count != 0u
            && (tiles.tile_block_ids == nullptr || tiles.tile_block_cell_masks == nullptr
                || tiles.block_row_entry_offsets == nullptr))
        || (tiles.row_block_entry_count != 0u && tiles.row_block_gene_masks == nullptr))
        return false;
    for (u32 block = 0; block < plan.feature_block_count; ++block) {
        const u32 begin = plan.feature_block_offsets[block];
        const u32 end = plan.feature_block_offsets[block + 1u];
        if (begin > end || end > plan.feature_count || end - begin > 32u) return false;
    }
    return true;
}
} // namespace

PackingStatisticsManifest cell_packing_statistics_manifest() noexcept {
    PackingStatisticsManifest result;
    result.hot_summary_mask = statistic_mask(PackingStatistic::row_nnz)
        | statistic_mask(PackingStatistic::block_occupancy)
        | statistic_mask(PackingStatistic::feature_reuse)
        | statistic_mask(PackingStatistic::lane_imbalance)
        | statistic_mask(PackingStatistic::metadata_value_ratio)
        | statistic_mask(PackingStatistic::dense_fragment_candidates)
        | statistic_mask(PackingStatistic::heavy_rows);
    result.cold_sidecar_mask = result.hot_summary_mask
        | statistic_mask(PackingStatistic::module_occupancy)
        | statistic_mask(PackingStatistic::row_mask_popcount)
        | statistic_mask(PackingStatistic::feature_mask_popcount)
        | statistic_mask(PackingStatistic::partial_block_occupancy)
        | statistic_mask(PackingStatistic::forward_locality)
        | statistic_mask(PackingStatistic::transpose_locality)
        | statistic_mask(PackingStatistic::cross_partition_edges)
        | statistic_mask(PackingStatistic::module_activation_frequency)
        | statistic_mask(PackingStatistic::quantization_range)
        | statistic_mask(PackingStatistic::quantization_outliers);
    result.requires_external_semantics_mask =
        statistic_mask(PackingStatistic::module_occupancy)
        | statistic_mask(PackingStatistic::forward_locality)
        | statistic_mask(PackingStatistic::transpose_locality)
        | statistic_mask(PackingStatistic::cross_partition_edges)
        | statistic_mask(PackingStatistic::module_activation_frequency)
        | statistic_mask(PackingStatistic::quantization_range)
        | statistic_mask(PackingStatistic::quantization_outliers);
    return result;
}

validation_result evaluate_cell_packing_statistics_host(
    const CellPackingStatisticsSource &source,
    CellPackingHotSummary *hot,
    CellPackingColdSidecar *cold) noexcept {
    if (hot == nullptr || cold == nullptr) return validation_error(
        validation_code::null_pointer, invalid_id, "packing statistics output is null");
    if (!valid_statistics_source(source)) return validation_error(
        validation_code::invalid_matrix_view, invalid_id, "invalid cell packing statistics source");
    CellPackingHotSummary hot_result;
    CellPackingColdSidecar cold_result;
    const warp_tile_view &tiles = source.tiles;
    const local_cell_order_view &order = source.order;
    hot_result.row_count = order.row_count;
    hot_result.feature_count = source.plan.feature_count;
    hot_result.feature_block_count = source.plan.feature_block_count;
    hot_result.row_group_count = source.row_group_count;
    hot_result.tile_count = tiles.tile_count;
    hot_result.tile_block_count = tiles.tile_block_count;
    hot_result.nnz_count = tiles.nnz_count;
    hot_result.projection_metadata_bytes = metadata_bytes(tiles);
    hot_result.projection_value_bytes = static_cast<u64>(tiles.nnz_count)
        * tiles.value_size_bytes;
    if (hot_result.projection_value_bytes != 0u) {
        hot_result.metadata_to_value_ratio =
            static_cast<double>(hot_result.projection_metadata_bytes)
            / static_cast<double>(hot_result.projection_value_bytes);
    }

    u64 total_row_nnz = 0u;
    for (u32 row = 0; row < order.row_count; ++row) {
        const u32 count = order.row_nnz_counts[row];
        total_row_nnz += count;
        hot_result.maximum_row_nnz = std::max(hot_result.maximum_row_nnz, count);
        ++cold_result.row_nnz_histogram[row_nnz_bucket(count)];
    }
    if (order.row_count != 0u) {
        hot_result.mean_row_nnz = static_cast<double>(total_row_nnz) / order.row_count;
        for (u32 row = 0; row < order.row_count; ++row) {
            if (static_cast<double>(order.row_nnz_counts[row])
                > 2.0 * hot_result.mean_row_nnz) ++hot_result.heavy_row_count;
        }
    }

    double occupancy_sum = 0.0, reuse_sum = 0.0, imbalance_sum = 0.0;
    for (u32 descriptor = 0; descriptor < tiles.tile_block_count; ++descriptor) {
        if (tiles.block_row_entry_offsets[descriptor]
                > tiles.block_row_entry_offsets[descriptor + 1u]
            || tiles.block_row_entry_offsets[descriptor + 1u]
                > tiles.row_block_entry_count
            || tiles.tile_block_ids[descriptor] >= source.plan.feature_block_count) {
            return validation_error(validation_code::invalid_offsets, descriptor,
                "CP-BP tile descriptor is outside packing geometry");
        }
        const u32 active_rows = popcount(tiles.tile_block_cell_masks[descriptor]);
        u32 feature_union = 0u, descriptor_nnz = 0u;
        u32 minimum_lane_nnz = std::numeric_limits<u32>::max(), maximum_lane_nnz = 0u;
        const u32 entry_begin = tiles.block_row_entry_offsets[descriptor];
        const u32 entry_end = tiles.block_row_entry_offsets[descriptor + 1u];
        for (u32 entry = entry_begin; entry < entry_end; ++entry) {
            const u32 mask = tiles.row_block_gene_masks[entry];
            const u32 lane_nnz = popcount(mask);
            feature_union |= mask;
            descriptor_nnz += lane_nnz;
            minimum_lane_nnz = std::min(minimum_lane_nnz, lane_nnz);
            maximum_lane_nnz = std::max(maximum_lane_nnz, lane_nnz);
        }
        const u32 active_features = popcount(feature_union);
        ++cold_result.row_mask_popcount_histogram[active_rows];
        ++cold_result.feature_mask_popcount_histogram[active_features];
        const u32 block = tiles.tile_block_ids[descriptor];
        const u32 block_width = source.plan.feature_block_offsets[block + 1u]
            - source.plan.feature_block_offsets[block];
        const u64 capacity = static_cast<u64>(active_rows) * block_width;
        const double occupancy = capacity == 0u ? 0.0
            : static_cast<double>(descriptor_nnz) / static_cast<double>(capacity);
        occupancy_sum += occupancy;
        cold_result.partial_block_occupancy_sum += occupancy;
        ++cold_result.partial_block_sample_count;
        reuse_sum += active_features == 0u ? 0.0
            : static_cast<double>(descriptor_nnz) / active_features;
        imbalance_sum += entry_begin == entry_end ? 0.0
            : static_cast<double>(maximum_lane_nnz - minimum_lane_nnz);
        if (capacity >= 16u && occupancy >= 0.5) {
            ++hot_result.dense_fragment_candidate_count;
        }
    }
    if (tiles.tile_block_count != 0u) {
        const double count = static_cast<double>(tiles.tile_block_count);
        hot_result.mean_block_occupancy = occupancy_sum / count;
        hot_result.mean_feature_reuse = reuse_sum / count;
        hot_result.mean_lane_imbalance = imbalance_sum / count;
    }
    cold_result.available_mask = cell_packing_statistics_manifest().cold_sidecar_mask
        & ~cell_packing_statistics_manifest().requires_external_semantics_mask;
    *hot = hot_result;
    *cold = cold_result;
    return validation_ok();
}
} // namespace cellerator::layout
