#pragma once

#include <Cellerator/compute/projection/physical_csr.hh>

#include <Cellerator/geometry/persistent_packing_payload.hh>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace cellerator::compute::math {

inline constexpr u32 native_tile_view_schema_version = 1u;

// One scheduling record per existing CPK1 tile-block descriptor. The dense
// workload is the number of row/feature pairs covered by the descriptor's
// active rows and union feature mask; nnz is the exact compact work present.
struct native_tile_block_metrics {
    u32 active_rows = 0u;
    u32 active_features = 0u;
    u32 nnz = 0u;
    u32 dense_workload = 0u;
    double density = 0.0;
    double feature_reuse = 0.0;
};

struct native_tile_requirements {
    std::size_t union_mask_count = 0u;
    std::size_t packed_offset_count = 0u;
    std::size_t block_metric_count = 0u;
    std::size_t sidecar_bytes = 0u;
};

struct native_tile_buffers {
    std::size_t union_mask_capacity = 0u;
    u32 *union_gene_masks = nullptr;
    std::size_t packed_offset_capacity = 0u;
    u32 *packed_value_offsets = nullptr;
    std::size_t block_metric_capacity = 0u;
    native_tile_block_metrics *block_metrics = nullptr;
};

// Read-only math projection. plan/order/tiles and every compact value pointer
// alias the frozen CellPack objects; only the caller-owned sidecars are new.
struct native_tile_view {
    u32 schema_version = native_tile_view_schema_version;
    cellpack::feature_weighted_row_reduction_plan_view plan{};
    cellpack::local_cell_order_view order{};
    cellpack::warp_tile_view tiles{};
    const u32 *union_gene_masks = nullptr;
    const u32 *packed_value_offsets = nullptr;
    const native_tile_block_metrics *block_metrics = nullptr;
    u64 dense_workload = 0u;
};

// Exact logical coordinate for one existing compact value. value aliases the
// corresponding bytes in tiles.values; the decoder never materializes values.
struct native_tile_decoded_value {
    u32 value_index = 0u;
    u32 execution_row = 0u;
    u32 canonical_row = 0u;
    u64 global_row = 0u;
    u32 execution_feature = 0u;
    u32 canonical_feature = 0u;
    const void *value = nullptr;
};

namespace native_tile_detail {
physical_view_status validate_source(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const cellpack::warp_tile_view &tiles) noexcept;
} // namespace native_tile_detail

inline physical_view_status query_native_tile_requirements_host(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const cellpack::warp_tile_view &tiles,
    native_tile_requirements *out) noexcept {
    if (out == nullptr) return {
        physical_view_status_code::invalid_argument, "null requirements output"};
    const auto checked = native_tile_detail::validate_source(plan, order, tiles);
    if (!checked) return checked;
    native_tile_requirements result;
    result.union_mask_count = tiles.tile_block_count;
    result.packed_offset_count = static_cast<std::size_t>(tiles.tile_block_count) + 1u;
    result.block_metric_count = tiles.tile_block_count;
    if (result.union_mask_count > std::numeric_limits<std::size_t>::max() / sizeof(u32)
        || result.packed_offset_count > std::numeric_limits<std::size_t>::max() / sizeof(u32)
        || result.block_metric_count > std::numeric_limits<std::size_t>::max()
            / sizeof(native_tile_block_metrics)) {
        return {physical_view_status_code::overflow, "sidecar size overflow"};
    }
    result.sidecar_bytes = (result.union_mask_count + result.packed_offset_count)
        * sizeof(u32) + result.block_metric_count * sizeof(native_tile_block_metrics);
    *out = result;
    return {};
}

physical_view_status build_native_tile_view_host(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const cellpack::warp_tile_view &tiles,
    const native_tile_buffers &buffers,
    native_tile_view *out) noexcept;

inline physical_view_status query_native_tile_requirements_from_cpk1_host(
    const cellpack::persistent_packing_payload_view &payload,
    native_tile_requirements *out) noexcept {
    if (!physical_csr_detail::valid_payload_metadata(payload)) {
        return {physical_view_status_code::invalid_argument, "invalid CPK1 view"};
    }
    return query_native_tile_requirements_host(
        payload.plan, payload.order, payload.tiles, out);
}

inline physical_view_status build_native_tile_view_from_cpk1_host(
    const cellpack::persistent_packing_payload_view &payload,
    const native_tile_buffers &buffers,
    native_tile_view *out) noexcept {
    if (!physical_csr_detail::valid_payload_metadata(payload)) {
        return {physical_view_status_code::invalid_argument, "invalid CPK1 view"};
    }
    return build_native_tile_view_host(
        payload.plan, payload.order, payload.tiles, buffers, out);
}

physical_view_status decode_native_tile_value_host(
    const native_tile_view &view,
    u32 value_index,
    native_tile_decoded_value *out) noexcept;

static_assert(std::is_trivially_copyable<native_tile_view>::value,
    "native tile view must remain pointer-copyable");

} // namespace cellerator::compute::math
