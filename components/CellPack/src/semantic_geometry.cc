#include "CellPack/semantic_geometry.hh"

#include <algorithm>
#include <limits>

namespace cellpack {
namespace {

using namespace cellerator::execution;

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

bool valid_payload_metadata(const persistent_packing_payload_view &payload) noexcept {
    const auto &plan = payload.plan;
    const auto &order = payload.order;
    const auto &tiles = payload.tiles;
    if (payload.payload_schema_version != persistent_packing_payload_schema_version
        || payload.payload_kind != persistent_packing_payload_kind
        || payload.payload_identity == 0u || payload.image_base == nullptr
        || payload.image_bytes == 0u || payload.cost_policy_identity == 0u)
        return false;
    if (plan.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || plan.geometry_identity_version != feature_block_geometry_identity_version
        || plan.feature_count == 0u || plan.feature_block_count == 0u
        || plan.feature_block_geometry_identity == 0u
        || plan.feature_block_offsets == nullptr || plan.feature_permutation == nullptr
        || payload.inverse_feature_permutation == nullptr
        || payload.feature_to_block == nullptr || payload.feature_to_local == nullptr)
        return false;
    if (order.order_schema_version != local_cell_order_schema_version
        || order.ordering_identity == 0u || order.row_count == 0u
        || order.row_permutation == nullptr || order.inverse_row_permutation == nullptr
        || order.row_nnz_counts == nullptr || payload.row_group_count == 0u
        || payload.row_group_offsets == nullptr)
        return false;
    if (tiles.tile_schema_version != warp_tile_schema_version
        || tiles.tile_identity == 0u || tiles.ordering_identity != order.ordering_identity
        || tiles.feature_block_geometry_identity != plan.feature_block_geometry_identity
        || tiles.row_count != order.row_count || tiles.feature_count != plan.feature_count
        || tiles.feature_block_count != plan.feature_block_count
        || tiles.value_size_bytes == 0u)
        return false;
    if ((tiles.tile_count != 0u && tiles.tile_block_offsets == nullptr)
        || (tiles.tile_block_count != 0u
            && (tiles.tile_block_ids == nullptr || tiles.tile_block_cell_masks == nullptr
                || tiles.block_row_entry_offsets == nullptr))
        || (tiles.row_block_entry_count != 0u && tiles.row_block_gene_masks == nullptr)
        || (tiles.nnz_count != 0u
            && (tiles.row_block_value_offsets == nullptr || tiles.values == nullptr)))
        return false;
    return true;
}

u64 metadata_bytes(const warp_tile_view &tiles) noexcept {
    const u64 words = static_cast<u64>(tiles.tile_count) + 1u
        + 2u * tiles.tile_block_count
        + static_cast<u64>(tiles.tile_block_count) + 1u
        + tiles.row_block_entry_count
        + static_cast<u64>(tiles.row_block_entry_count) + 1u;
    return words * sizeof(u32);
}

} // namespace

semantic_statistics_manifest cp_bp_semantic_statistics_manifest() noexcept {
    semantic_statistics_manifest result;
    result.hot_summary_mask = statistic_mask(semantic_statistic::row_nnz)
        | statistic_mask(semantic_statistic::block_occupancy)
        | statistic_mask(semantic_statistic::feature_reuse)
        | statistic_mask(semantic_statistic::lane_imbalance)
        | statistic_mask(semantic_statistic::metadata_value_ratio)
        | statistic_mask(semantic_statistic::dense_fragment_candidates)
        | statistic_mask(semantic_statistic::heavy_rows);
    result.cold_sidecar_mask = result.hot_summary_mask
        | statistic_mask(semantic_statistic::module_occupancy)
        | statistic_mask(semantic_statistic::row_mask_popcount)
        | statistic_mask(semantic_statistic::feature_mask_popcount)
        | statistic_mask(semantic_statistic::partial_block_occupancy)
        | statistic_mask(semantic_statistic::forward_locality)
        | statistic_mask(semantic_statistic::transpose_locality)
        | statistic_mask(semantic_statistic::cross_partition_edges)
        | statistic_mask(semantic_statistic::module_activation_frequency)
        | statistic_mask(semantic_statistic::quantization_range)
        | statistic_mask(semantic_statistic::quantization_outliers);
    result.requires_external_semantics_mask =
        statistic_mask(semantic_statistic::module_occupancy)
        | statistic_mask(semantic_statistic::forward_locality)
        | statistic_mask(semantic_statistic::transpose_locality)
        | statistic_mask(semantic_statistic::cross_partition_edges)
        | statistic_mask(semantic_statistic::module_activation_frequency)
        | statistic_mask(semantic_statistic::quantization_range)
        | statistic_mask(semantic_statistic::quantization_outliers);
    return result;
}

validation_result build_cp_bp_v1_compatibility_adapter_host(
    const persistent_packing_payload_view &payload,
    const cp_bp_v1_adapter_request &request,
    cp_bp_v1_compatibility_adapter *out) noexcept {
    if (out == nullptr) return validation_error(
        validation_code::null_pointer, invalid_id, "CP-BP adapter output is null");
    if (!valid_payload_metadata(payload)) return validation_error(
        validation_code::invalid_matrix_view, invalid_id, "CPK1 metadata is invalid");
    if (!valid_axis_identity(request.row_axis)
        || !valid_axis_identity(request.feature_axis)
        || !valid_handle(request.structure) || !valid_handle(request.projection)
        || !valid_projection_catalog(request.projection_catalog)
        || request.structure_epoch_value.value == 0u
        || request.value_generation_value.value == 0u
        || !valid_location(request.value_location)) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "CP-BP adapter identity or residency is invalid");
    }
    if (request.numeric.storage == numeric_type::invalid
        || request.numeric.dequantized == numeric_type::invalid
        || request.numeric.accumulation == numeric_type::invalid) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CP-BP adapter numeric policy is invalid");
    }
    const u64 value_bytes = static_cast<u64>(payload.tiles.nnz_count)
        * payload.tiles.value_size_bytes;
    if (payload.tiles.nnz_count != 0u
        && value_bytes / payload.tiles.nnz_count != payload.tiles.value_size_bytes) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CP-BP v1 value byte count overflows");
    }

    cp_bp_v1_compatibility_adapter result;
    result.geometry.row_axis = request.row_axis;
    result.geometry.feature_axis = request.feature_axis;
    result.geometry.v1_feature_block_geometry_identity =
        payload.plan.feature_block_geometry_identity;
    result.geometry.v1_ordering_identity = payload.order.ordering_identity;
    result.geometry.v1_payload_identity = payload.payload_identity;
    result.geometry.cost_policy_identity = payload.cost_policy_identity;
    result.geometry.v1_objective_kind = payload.objective_kind;
    result.geometry.row_count = payload.order.row_count;
    result.geometry.feature_count = payload.plan.feature_count;
    result.geometry.feature_block_count = payload.plan.feature_block_count;
    result.geometry.row_group_count = payload.row_group_count;
    result.geometry.feature_block_offsets = payload.plan.feature_block_offsets;
    result.geometry.feature_to_block = payload.feature_to_block;
    result.geometry.feature_to_local = payload.feature_to_local;
    result.geometry.row_group_offsets = payload.row_group_offsets;
    result.geometry.execution_to_canonical_feature = payload.plan.feature_permutation;
    result.geometry.canonical_to_execution_feature = payload.inverse_feature_permutation;
    result.geometry.execution_to_canonical_row = payload.order.row_permutation;
    result.geometry.canonical_to_execution_row = payload.order.inverse_row_permutation;

    result.structure.identity = request.structure;
    result.structure.epoch = request.structure_epoch_value;
    result.structure.source_axis = request.row_axis;
    result.structure.destination_axis = request.feature_axis;
    result.structure.projections = request.projection_catalog;
    result.structure.logical_edge_count = payload.tiles.nnz_count;
    result.values.structure = request.structure;
    result.values.structure_epoch_value = request.structure_epoch_value;
    result.values.values = const_cast<void *>(payload.tiles.values);
    result.values.location = request.value_location;
    result.values.numeric = request.numeric;
    result.values.quantization.kind = quantization_kind::none;
    result.values.quantization.scale_type = numeric_type::invalid;
    result.values.quantization.offset_type = numeric_type::invalid;
    result.values.layout = value_layout_kind::projection_local_order;
    result.values.generation = request.value_generation_value;
    result.values.element_count = payload.tiles.nnz_count;
    result.values.value_bytes = value_bytes;
    result.projection = request.projection;
    result.payload = payload;
    result.direct_plan = payload.plan;
    result.direct_order = payload.order;
    result.direct_tiles = payload.tiles;
    if (validate_relation_structure(result.structure) != lifetime_validation_code::ok
        || validate_value_plane(result.structure, result.values)
            != lifetime_validation_code::ok) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "CP-BP adapter lifetime contract is invalid");
    }
    *out = result;
    return validation_ok();
}

validation_result evaluate_cp_bp_v1_semantic_statistics_host(
    const cp_bp_v1_compatibility_adapter &adapter,
    semantic_geometry_hot_summary *hot,
    semantic_geometry_cold_sidecar *cold) noexcept {
    if (hot == nullptr || cold == nullptr) return validation_error(
        validation_code::null_pointer, invalid_id, "semantic statistics output is null");
    if (adapter.schema_version != cp_bp_v1_compatibility_adapter_schema_version
        || !valid_payload_metadata(adapter.payload)
        || adapter.direct_tiles.values != adapter.payload.tiles.values) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CP-BP adapter is invalid or no longer aliases CPK1");
    }
    semantic_geometry_hot_summary hot_result;
    semantic_geometry_cold_sidecar cold_result;
    const warp_tile_view &tiles = adapter.direct_tiles;
    const local_cell_order_view &order = adapter.direct_order;
    hot_result.row_count = order.row_count;
    hot_result.feature_count = adapter.direct_plan.feature_count;
    hot_result.feature_block_count = adapter.direct_plan.feature_block_count;
    hot_result.row_group_count = adapter.geometry.row_group_count;
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
            || tiles.tile_block_ids[descriptor] >= adapter.direct_plan.feature_block_count) {
            return validation_error(validation_code::invalid_offsets, descriptor,
                "CP-BP tile descriptor is outside semantic geometry");
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
        const u32 block_width = adapter.direct_plan.feature_block_offsets[block + 1u]
            - adapter.direct_plan.feature_block_offsets[block];
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
    cold_result.available_mask = cp_bp_semantic_statistics_manifest().cold_sidecar_mask
        & ~cp_bp_semantic_statistics_manifest().requires_external_semantics_mask;
    *hot = hot_result;
    *cold = cold_result;
    return validation_ok();
}

} // namespace cellpack
