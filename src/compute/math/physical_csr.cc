#include <Cellerator/compute/math/physical_csr.hh>

#include <algorithm>
#include <cstring>
#include <limits>

namespace cellerator::compute::math {
namespace {

physical_view_status fail(physical_view_status_code code, const char *message) noexcept {
    return {code, message};
}

} // namespace

physical_view_status build_execution_csr_view_host(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const cellpack::ordered_plan_partition_view &ordered,
    const execution_csr_feature_buffers &buffers,
    execution_csr_view *out) noexcept {
    if (out == nullptr || !physical_csr_detail::valid_plan(plan)) {
        return fail(physical_view_status_code::invalid_argument,
            "execution CSR requires a valid frozen plan view and output");
    }
    if (ordered.semantic_plan_schema_version != plan.semantic_plan_schema_version
        || ordered.feature_count != plan.feature_count
        || ordered.feature_axis_fingerprint == 0u
        || ordered.feature_axis_fingerprint_version == 0u
        || ordered.row_offsets == nullptr
        || (ordered.nnz_count != 0u && ordered.value_size_bytes == 0u)
        || (ordered.nnz_count != 0u
            && (ordered.block_ids == nullptr || ordered.local_feature_ids == nullptr
                || ordered.canonical_feature_ids == nullptr || ordered.values == nullptr))) {
        return fail(physical_view_status_code::incompatible_identity,
            "ordered partition disagrees with the frozen feature plan");
    }
    if (buffers.execution_feature_capacity < ordered.nnz_count
        || (ordered.nnz_count != 0u && buffers.execution_feature_ids == nullptr)) {
        return fail(physical_view_status_code::insufficient_capacity,
            "execution feature buffer is too small");
    }
    if (ordered.row_offsets[0] != 0u
        || ordered.row_offsets[ordered.row_count] != ordered.nnz_count) {
        return fail(physical_view_status_code::invalid_geometry,
            "ordered CSR row offsets do not span nnz");
    }
    for (u32 row = 0u; row < ordered.row_count; ++row) {
        const u32 begin = ordered.row_offsets[row], end = ordered.row_offsets[row + 1u];
        if (end < begin || end > ordered.nnz_count) {
            return fail(physical_view_status_code::invalid_geometry,
                "ordered CSR row offsets are not monotonic");
        }
        u32 previous = std::numeric_limits<u32>::max();
        for (u32 entry = begin; entry < end; ++entry) {
            const u32 block = ordered.block_ids[entry];
            if (block >= plan.feature_block_count) {
                return fail(physical_view_status_code::invalid_geometry,
                    "ordered feature block is out of range");
            }
            const u32 block_begin = plan.feature_block_offsets[block];
            const u32 block_end = plan.feature_block_offsets[block + 1u];
            const u32 local = ordered.local_feature_ids[entry];
            if (local >= block_end - block_begin) {
                return fail(physical_view_status_code::invalid_geometry,
                    "ordered local feature is out of range");
            }
            const u32 execution = block_begin + local;
            if (plan.feature_permutation[execution] != ordered.canonical_feature_ids[entry]
                || (entry != begin && execution <= previous)) {
                return fail(physical_view_status_code::incompatible_identity,
                    "ordered feature identity does not map to packed order");
            }
            buffers.execution_feature_ids[entry] = execution;
            previous = execution;
        }
    }
    execution_csr_view result;
    result.global_row_begin = ordered.global_row_begin;
    result.full_row_count = ordered.full_row_count;
    result.row_count = ordered.row_count;
    result.feature_count = ordered.feature_count;
    result.nnz_count = ordered.nnz_count;
    result.value_size_bytes = ordered.value_size_bytes;
    result.row_domain_identity = ordered.row_domain_identity;
    result.feature_order = physical_csr_detail::packed_order(ordered.feature_count,
        ordered.feature_axis_fingerprint, ordered.feature_axis_fingerprint_version,
        plan.feature_block_geometry_identity);
    result.row_offsets = ordered.row_offsets;
    result.execution_feature_ids = buffers.execution_feature_ids;
    result.values = ordered.values;
    result.structure.identity_version = execution_csr_structure_identity_version;
    result.structure.value = physical_csr_detail::structure_identity(result.global_row_begin,
        result.row_domain_identity, result.row_count, result.feature_count,
        result.nnz_count, result.row_offsets, result.execution_feature_ids);
    *out = result;
    return {};
}

physical_view_status materialize_execution_csr_from_cpk1_host(
    const cellpack::persistent_packing_payload_view &payload,
    const lazy_execution_csr_buffers &buffers,
    execution_csr_view *out) noexcept {
    lazy_execution_csr_requirements required;
    physical_view_status status = query_lazy_execution_csr_requirements(payload, &required);
    if (!status || out == nullptr) return status ? fail(
        physical_view_status_code::invalid_argument, "lazy CSR output is null") : status;
    if (buffers.row_offset_capacity < required.row_offset_count
        || buffers.execution_feature_capacity < required.execution_feature_count
        || buffers.value_capacity_bytes < required.value_bytes
        || buffers.row_cursor_capacity < required.row_cursor_count
        || buffers.row_offsets == nullptr
        || (required.execution_feature_count != 0u
            && (buffers.execution_feature_ids == nullptr || buffers.values == nullptr))
        || (required.row_cursor_count != 0u && buffers.row_cursors == nullptr)) {
        return fail(physical_view_status_code::insufficient_capacity,
            "lazy CSR caller-owned buffers are too small");
    }
    const auto &tiles = payload.tiles;
    std::fill(buffers.row_offsets, buffers.row_offsets + required.row_offset_count, 0u);
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 descriptor_begin = tiles.tile_block_offsets[tile];
        const u32 descriptor_end = tiles.tile_block_offsets[tile + 1u];
        const u32 remaining = tiles.row_count - tile * tiles.tile_row_width;
        const u32 lanes = std::min(tiles.tile_row_width, remaining);
        const u32 valid_cell_mask = lanes == 32u
            ? std::numeric_limits<u32>::max() : (1u << lanes) - 1u;
        if (descriptor_end < descriptor_begin || descriptor_end > tiles.tile_block_count) {
            return fail(physical_view_status_code::invalid_geometry,
                "CPK1 tile descriptor offsets are invalid");
        }
        u32 previous_block = std::numeric_limits<u32>::max();
        for (u32 descriptor = descriptor_begin; descriptor < descriptor_end; ++descriptor) {
            const u32 block = tiles.tile_block_ids[descriptor];
            const u32 cell_mask = tiles.tile_block_cell_masks[descriptor];
            u32 entry = tiles.block_row_entry_offsets[descriptor];
            const u32 entry_end = tiles.block_row_entry_offsets[descriptor + 1u];
            if (block >= payload.plan.feature_block_count || cell_mask == 0u
                || (descriptor != descriptor_begin && block <= previous_block)
                || (cell_mask & ~valid_cell_mask) != 0u
                || entry_end < entry || entry_end > tiles.row_block_entry_count) {
                return fail(physical_view_status_code::invalid_geometry,
                    "CPK1 tile descriptor is invalid");
            }
            previous_block = block;
            const u32 block_begin = payload.plan.feature_block_offsets[block];
            const u32 block_end = payload.plan.feature_block_offsets[block + 1u];
            if (block_end <= block_begin || block_end - block_begin > 32u) {
                return fail(physical_view_status_code::invalid_geometry,
                    "CPK1 feature-block width is invalid");
            }
            const u32 block_width = block_end - block_begin;
            const u32 valid_gene_mask = block_width == 32u
                ? std::numeric_limits<u32>::max() : (1u << block_width) - 1u;
            for (u32 lane = 0u; lane < tiles.tile_row_width; ++lane) {
                if ((cell_mask & (1u << lane)) == 0u) continue;
                const u32 row = tile * tiles.tile_row_width + lane;
                if (row >= tiles.row_count || entry >= entry_end) {
                    return fail(physical_view_status_code::invalid_geometry,
                        "CPK1 tile row entry is invalid");
                }
                const u32 gene_mask = tiles.row_block_gene_masks[entry];
                const u32 value_begin = tiles.row_block_value_offsets[entry];
                const u32 value_end = tiles.row_block_value_offsets[++entry];
                const u32 count = static_cast<u32>(__builtin_popcount(gene_mask));
                if (gene_mask == 0u
                    || (gene_mask & ~valid_gene_mask) != 0u
                    || value_end < value_begin || value_end > tiles.nnz_count
                    || value_end - value_begin != count) {
                    return fail(physical_view_status_code::invalid_geometry,
                        "CPK1 row-block gene/value geometry is invalid");
                }
                if (buffers.row_offsets[row + 1u] > tiles.nnz_count
                    || count > tiles.nnz_count - buffers.row_offsets[row + 1u]) {
                    return fail(physical_view_status_code::overflow,
                        "CPK1 row nnz count overflows");
                }
                buffers.row_offsets[row + 1u] += count;
            }
            if (entry != entry_end) return fail(
                physical_view_status_code::invalid_geometry,
                "CPK1 row-entry count disagrees with the cell mask");
        }
    }
    for (u32 row = 0u; row < tiles.row_count; ++row) {
        buffers.row_offsets[row + 1u] += buffers.row_offsets[row];
        buffers.row_cursors[row] = buffers.row_offsets[row];
    }
    if (buffers.row_offsets[tiles.row_count] != tiles.nnz_count) {
        return fail(physical_view_status_code::invalid_geometry,
            "CPK1 decoded row counts do not span nnz");
    }
    const auto *source_values = static_cast<const unsigned char *>(tiles.values);
    auto *target_values = static_cast<unsigned char *>(buffers.values);
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        for (u32 descriptor = tiles.tile_block_offsets[tile];
             descriptor < tiles.tile_block_offsets[tile + 1u]; ++descriptor) {
            const u32 block = tiles.tile_block_ids[descriptor];
            const u32 block_begin = payload.plan.feature_block_offsets[block];
            u32 entry = tiles.block_row_entry_offsets[descriptor];
            for (u32 lane = 0u; lane < tiles.tile_row_width; ++lane) {
                if ((tiles.tile_block_cell_masks[descriptor] & (1u << lane)) == 0u) continue;
                const u32 row = tile * tiles.tile_row_width + lane;
                const u32 gene_mask = tiles.row_block_gene_masks[entry];
                u32 value = tiles.row_block_value_offsets[entry++];
                for (u32 local = 0u; local < 32u; ++local) {
                    if ((gene_mask & (1u << local)) == 0u) continue;
                    const u32 destination = buffers.row_cursors[row]++;
                    buffers.execution_feature_ids[destination] = block_begin + local;
                    std::memcpy(target_values + static_cast<std::size_t>(destination)
                            * tiles.value_size_bytes,
                        source_values + static_cast<std::size_t>(value++)
                            * tiles.value_size_bytes,
                        tiles.value_size_bytes);
                }
            }
        }
    }
    execution_csr_view result;
    result.global_row_begin = tiles.global_row_begin;
    result.full_row_count = tiles.full_row_count;
    result.row_count = tiles.row_count;
    result.feature_count = tiles.feature_count;
    result.nnz_count = tiles.nnz_count;
    result.value_size_bytes = tiles.value_size_bytes;
    result.row_domain_identity = tiles.row_domain_identity;
    result.feature_order = physical_csr_detail::packed_order(tiles.feature_count,
        tiles.feature_axis_fingerprint, tiles.feature_axis_fingerprint_version,
        payload.plan.feature_block_geometry_identity);
    result.row_offsets = buffers.row_offsets;
    result.execution_feature_ids = buffers.execution_feature_ids;
    result.values = buffers.values;
    result.structure.identity_version = execution_csr_structure_identity_version;
    result.structure.value = physical_csr_detail::structure_identity(result.global_row_begin,
        result.row_domain_identity, result.row_count, result.feature_count,
        result.nnz_count, result.row_offsets, result.execution_feature_ids);
    *out = result;
    return {};
}

} // namespace cellerator::compute::math
