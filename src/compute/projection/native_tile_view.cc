#include <Cellerator/compute/projection/native_tile_view.hh>

#include <limits>

namespace cellerator::compute::math {
namespace {

using status = physical_view_status;
using code = physical_view_status_code;
using plan_view = cellpack::feature_weighted_row_reduction_plan_view;
using order_view = cellpack::local_cell_order_view;
using tile_view = cellpack::warp_tile_view;
using requirements = native_tile_requirements;
using buffers = native_tile_buffers;
using native_view = native_tile_view;
using decoded_value = native_tile_decoded_value;

status fail(code value, const char *message) noexcept {
    return {value, message};
}

u32 popcount(u32 value) noexcept {
    return static_cast<u32>(__builtin_popcount(value));
}

u32 low_mask(u32 width) noexcept {
    return width == 32u ? std::numeric_limits<u32>::max() : (1u << width) - 1u;
}

} // namespace

namespace native_tile_detail {

status validate_source(
    const plan_view &plan,
    const order_view &order,
    const tile_view &tiles) noexcept {
    if (!physical_csr_detail::valid_plan(plan)) return fail(
        code::invalid_argument, "invalid native plan");
    if (order.order_schema_version != cellpack::local_cell_order_schema_version
        || order.signature_algorithm_version
            != cellpack::local_cell_signature_algorithm_version
        || order.ordering_identity == 0u || order.group_width == 0u
        || order.group_width > cellpack::warp_tile_cell_mask_bits
        || order.row_count != tiles.row_count
        || order.feature_block_count != plan.feature_block_count
        || order.feature_block_geometry_identity != plan.feature_block_geometry_identity
        || order.ordering_identity != tiles.ordering_identity
        || order.global_row_begin != tiles.global_row_begin
        || order.full_row_count != tiles.full_row_count
        || order.row_domain_identity != tiles.row_domain_identity
        || (order.row_count != 0u
            && (order.row_permutation == nullptr
                || order.inverse_row_permutation == nullptr))) {
        return fail(code::incompatible_identity, "incompatible native order");
    }
    if (tiles.tile_schema_version != cellpack::warp_tile_schema_version
        || tiles.record_schema_version != cellpack::cell_block_record_schema_version
        || tiles.semantic_plan_schema_version != plan.semantic_plan_schema_version
        || tiles.geometry_identity_version != plan.geometry_identity_version
        || tiles.order_schema_version != order.order_schema_version
        || tiles.tile_identity == 0u
        || tiles.feature_block_geometry_identity != plan.feature_block_geometry_identity
        || tiles.feature_count != plan.feature_count
        || tiles.feature_block_count != plan.feature_block_count
        || tiles.tile_row_width != order.group_width
        || tiles.tile_count != tiles.row_count / tiles.tile_row_width
            + (tiles.row_count % tiles.tile_row_width != 0u ? 1u : 0u)
        || tiles.value_size_bytes == 0u || tiles.tile_block_offsets == nullptr
        || tiles.block_row_entry_offsets == nullptr
        || tiles.row_block_value_offsets == nullptr
        || tiles.tile_block_offsets[0] != 0u
        || tiles.tile_block_offsets[tiles.tile_count] != tiles.tile_block_count
        || tiles.block_row_entry_offsets[0] != 0u
        || tiles.block_row_entry_offsets[tiles.tile_block_count]
            != tiles.row_block_entry_count
        || tiles.row_block_value_offsets[0] != 0u
        || tiles.row_block_value_offsets[tiles.row_block_entry_count] != tiles.nnz_count
        || (tiles.tile_block_count != 0u
            && (tiles.tile_block_ids == nullptr
                || tiles.tile_block_cell_masks == nullptr))
        || (tiles.row_block_entry_count != 0u
            && tiles.row_block_gene_masks == nullptr)
        || (tiles.nnz_count != 0u && tiles.values == nullptr)) {
        return fail(code::invalid_geometry, "invalid native metadata");
    }
    if (tiles.nnz_count != 0u
        && tiles.value_size_bytes > std::numeric_limits<std::size_t>::max()
            / tiles.nnz_count) {
        return fail(code::overflow, "native value size overflow");
    }
    for (u32 execution = 0u; execution < order.row_count; ++execution) {
        const u32 canonical = order.row_permutation[execution];
        if (canonical >= order.row_count
            || order.inverse_row_permutation[canonical] != execution) {
            return fail(code::incompatible_identity, "invalid row permutation");
        }
    }
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 begin = tiles.tile_block_offsets[tile];
        const u32 end = tiles.tile_block_offsets[tile + 1u];
        if (end < begin || end > tiles.tile_block_count) return fail(
            code::invalid_geometry, "invalid tile offsets");
        const u32 rows_left = tiles.row_count - tile * tiles.tile_row_width;
        const u32 valid_cells = low_mask(rows_left < tiles.tile_row_width
            ? rows_left : tiles.tile_row_width);
        u32 previous_block = std::numeric_limits<u32>::max();
        for (u32 descriptor = begin; descriptor < end; ++descriptor) {
            const u32 block = tiles.tile_block_ids[descriptor];
            const u32 cells = tiles.tile_block_cell_masks[descriptor];
            const u32 entry_begin = tiles.block_row_entry_offsets[descriptor];
            const u32 entry_end = tiles.block_row_entry_offsets[descriptor + 1u];
            if (block >= plan.feature_block_count || cells == 0u
                || (cells & ~valid_cells) != 0u
                || (descriptor != begin && block <= previous_block)
                || entry_end < entry_begin || entry_end > tiles.row_block_entry_count
                || entry_end - entry_begin != popcount(cells)) {
                return fail(code::invalid_geometry, "invalid block descriptor");
            }
            previous_block = block;
            const u32 block_begin = plan.feature_block_offsets[block];
            const u32 block_end = plan.feature_block_offsets[block + 1u];
            if (block_end <= block_begin || block_end - block_begin > 32u) return fail(
                code::invalid_geometry, "invalid block width");
            const u32 valid_genes = low_mask(block_end - block_begin);
            for (u32 entry = entry_begin; entry < entry_end; ++entry) {
                const u32 genes = tiles.row_block_gene_masks[entry];
                const u32 value_begin = tiles.row_block_value_offsets[entry];
                const u32 value_end = tiles.row_block_value_offsets[entry + 1u];
                if (genes == 0u || (genes & ~valid_genes) != 0u
                    || value_end < value_begin || value_end > tiles.nnz_count
                    || value_end - value_begin != popcount(genes)) {
                    return fail(code::invalid_geometry, "invalid gene offsets");
                }
            }
        }
    }
    return {};
}

} // namespace native_tile_detail

physical_view_status build_native_tile_view_host(
    const plan_view &plan,
    const order_view &order,
    const tile_view &tiles,
    const buffers &buffers,
    native_view *out) noexcept {
    requirements required;
    auto checked = query_native_tile_requirements_host(plan, order, tiles, &required);
    if (!checked || out == nullptr) return checked ? fail(
        code::invalid_argument, "null native output") : checked;
    if (buffers.union_mask_capacity < required.union_mask_count
        || buffers.packed_offset_capacity < required.packed_offset_count
        || buffers.block_metric_capacity < required.block_metric_count
        || (required.union_mask_count != 0u && buffers.union_gene_masks == nullptr)
        || buffers.packed_value_offsets == nullptr
        || (required.block_metric_count != 0u && buffers.block_metrics == nullptr)) {
        return fail(code::insufficient_capacity, "small sidecar buffers");
    }
    u64 total_workload = 0u;
    for (u32 descriptor = 0u; descriptor < tiles.tile_block_count; ++descriptor) {
        const u32 entry_begin = tiles.block_row_entry_offsets[descriptor];
        const u32 entry_end = tiles.block_row_entry_offsets[descriptor + 1u];
        u32 union_mask = 0u;
        for (u32 entry = entry_begin; entry < entry_end; ++entry)
            union_mask |= tiles.row_block_gene_masks[entry];
        const u32 value_begin = tiles.row_block_value_offsets[entry_begin];
        const u32 value_end = tiles.row_block_value_offsets[entry_end];
        const u32 active_rows = popcount(tiles.tile_block_cell_masks[descriptor]);
        const u32 active_features = popcount(union_mask);
        const u32 nnz = value_end - value_begin;
        const u32 workload = active_rows * active_features;
        buffers.union_gene_masks[descriptor] = union_mask;
        buffers.packed_value_offsets[descriptor] = value_begin;
        buffers.block_metrics[descriptor] = {active_rows, active_features, nnz, workload,
            workload == 0u ? 0.0 : static_cast<double>(nnz) / workload,
            active_features == 0u ? 0.0 : static_cast<double>(nnz) / active_features};
        total_workload += workload;
    }
    buffers.packed_value_offsets[tiles.tile_block_count] = tiles.nnz_count;
    native_view result;
    result.plan = plan;
    result.order = order;
    result.tiles = tiles;
    result.union_gene_masks = buffers.union_gene_masks;
    result.packed_value_offsets = buffers.packed_value_offsets;
    result.block_metrics = buffers.block_metrics;
    result.dense_workload = total_workload;
    *out = result;
    return {};
}

physical_view_status decode_native_tile_value_host(
    const native_view &view,
    u32 value_index,
    decoded_value *out) noexcept {
    if (out == nullptr || view.schema_version != native_tile_view_schema_version
        || value_index >= view.tiles.nnz_count) {
        return fail(code::invalid_argument, "invalid decode request");
    }
    const auto checked = native_tile_detail::validate_source(
        view.plan, view.order, view.tiles);
    if (!checked) return checked;
    if (view.packed_value_offsets == nullptr) {
        return fail(code::invalid_geometry, "invalid native sidecars");
    }
    for (u32 descriptor = 0u; descriptor <= view.tiles.tile_block_count; ++descriptor) {
        const u32 entry = view.tiles.block_row_entry_offsets[descriptor];
        if (view.packed_value_offsets[descriptor]
            != view.tiles.row_block_value_offsets[entry]) {
            return fail(code::invalid_geometry, "invalid native sidecars");
        }
    }
    u32 descriptor = 0u;
    while (view.packed_value_offsets[descriptor + 1u] <= value_index) ++descriptor;
    if (descriptor >= view.tiles.tile_block_count) return fail(
        code::invalid_geometry, "invalid packed offsets");
    u32 tile = 0u;
    while (view.tiles.tile_block_offsets[tile + 1u] <= descriptor) ++tile;
    u32 entry = view.tiles.block_row_entry_offsets[descriptor];
    while (view.tiles.row_block_value_offsets[entry + 1u] <= value_index) ++entry;
    const u32 entry_rank = entry - view.tiles.block_row_entry_offsets[descriptor];
    const u32 cell_mask = view.tiles.tile_block_cell_masks[descriptor];
    u32 lane = 0u, seen = 0u;
    for (; lane < view.tiles.tile_row_width; ++lane) {
        if ((cell_mask & (1u << lane)) == 0u) continue;
        if (seen++ == entry_rank) break;
    }
    u32 gene_rank = value_index - view.tiles.row_block_value_offsets[entry];
    u32 gene = 0u;
    for (u32 mask = view.tiles.row_block_gene_masks[entry]; mask != 0u; ++gene, mask >>= 1u) {
        if ((mask & 1u) != 0u && gene_rank-- == 0u) break;
    }
    const u32 execution_row = tile * view.tiles.tile_row_width + lane;
    const u32 block = view.tiles.tile_block_ids[descriptor];
    const u32 execution_feature = view.plan.feature_block_offsets[block] + gene;
    decoded_value result;
    result.value_index = value_index;
    result.execution_row = execution_row;
    result.canonical_row = view.order.row_permutation[execution_row];
    result.global_row = view.tiles.global_row_begin + result.canonical_row;
    result.execution_feature = execution_feature;
    result.canonical_feature = view.plan.feature_permutation[execution_feature];
    result.value = static_cast<const unsigned char *>(view.tiles.values)
        + static_cast<std::size_t>(value_index) * view.tiles.value_size_bytes;
    *out = result;
    return {};
}

} // namespace cellerator::compute::math
