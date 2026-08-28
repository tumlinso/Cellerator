#include "Cellerator/geometry/warp_tiles.hh"

#include <algorithm>
#include <cstring>
#include <limits>

// Phase C keeps the query/build/validate/decode traversals together so the
// exact compact rank and identity rules cannot drift between host entrypoints.
// CUDA construction is deliberately a separate Phase D translation unit.
namespace cellpack {
namespace {

u64 splitmix64(u64 value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

u32 popcount_u32(u32 value) noexcept {
    u32 count = 0u;
    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

u32 tile_count_for_rows(u32 row_count, u32 width) noexcept {
    return row_count / width + (row_count % width != 0u ? 1u : 0u);
}

u32 valid_lane_mask(u32 count) noexcept {
    return count == warp_tile_cell_mask_bits
        ? std::numeric_limits<u32>::max()
        : ((1u << count) - 1u);
}

bool multiply_overflows(std::size_t lhs, std::size_t rhs, std::size_t *out) noexcept {
    if (out == nullptr || (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return true;
    }
    *out = lhs * rhs;
    return false;
}

validation_result validate_inputs(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order) {
    validation_result status = validate_cell_block_record_view_host(plan, records);
    if (!status) return status;
    status = validate_local_cell_order_view_host(records, order);
    if (!status) return status;
    if (order.group_width == 0u || order.group_width > warp_tile_cell_mask_bits) {
        return validation_error(validation_code::invalid_matrix_view, order.group_width,
            "warp-tile v1 requires one to 32 execution rows per tile");
    }
    return validation_ok();
}

u64 compute_tile_identity(
    const cell_block_record_view &records,
    const local_cell_order_view &order) noexcept {
    u64 identity = splitmix64(records.feature_block_geometry_identity);
    identity = splitmix64(identity ^ (static_cast<u64>(warp_tile_schema_version) << 32u)
        ^ records.record_schema_version);
    identity = splitmix64(identity ^ order.ordering_identity);
    identity = splitmix64(identity ^ records.row_domain_identity);
    identity = splitmix64(identity ^ records.global_row_begin);
    identity = splitmix64(identity ^ (static_cast<u64>(records.row_count) << 32u)
        ^ records.full_row_count);
    identity = splitmix64(identity ^ (static_cast<u64>(records.feature_count) << 32u)
        ^ records.feature_block_count);
    identity = splitmix64(identity ^ (static_cast<u64>(order.group_width) << 32u)
        ^ records.value_size_bytes);
    return identity == 0u ? 1u : identity;
}

u32 next_tile_block(
    const u32 *cursors,
    const u32 *ends,
    const cell_block_record_view &records,
    u32 lane_count) noexcept {
    u32 block = invalid_id;
    for (u32 lane = 0u; lane < lane_count; ++lane) {
        if (cursors[lane] != ends[lane]) {
            block = std::min(block, records.record_block_ids[cursors[lane]]);
        }
    }
    return block;
}

void initialize_tile_cursors(
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    u32 tile_begin,
    u32 lane_count,
    u32 *cursors,
    u32 *ends) noexcept {
    for (u32 lane = 0u; lane < lane_count; ++lane) {
        const u32 row = order.row_permutation[tile_begin + lane];
        cursors[lane] = records.row_record_offsets[row];
        ends[lane] = records.row_record_offsets[row + 1u];
    }
}

validation_result validate_output_buffers(
    const cell_block_record_view &records,
    const warp_tile_requirements &required,
    const warp_tile_buffers &buffers) {
    if (buffers.tile_block_offset_capacity < required.tile_block_offset_count
        || buffers.tile_block_capacity < required.tile_block_count
        || buffers.block_row_entry_offset_capacity < required.block_row_entry_offset_count
        || buffers.row_block_entry_capacity < required.row_block_entry_count
        || buffers.row_block_value_offset_capacity < required.row_block_value_offset_count
        || buffers.value_capacity_bytes < required.value_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "warp-tile output capacity is insufficient");
    }
    if (buffers.tile_block_offsets == nullptr
        || buffers.block_row_entry_offsets == nullptr
        || buffers.row_block_value_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile terminal offset output is null");
    }
    if (required.tile_block_count != 0u
        && (buffers.tile_block_ids == nullptr || buffers.tile_block_cell_masks == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile block descriptor output is null");
    }
    if (required.row_block_entry_count != 0u && buffers.row_block_gene_masks == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile row-block gene-mask output is null");
    }
    if (records.nnz_count != 0u && buffers.values == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile value output is null");
    }

    u32 *outputs[] = {
        buffers.tile_block_offsets,
        buffers.tile_block_ids,
        buffers.tile_block_cell_masks,
        buffers.block_row_entry_offsets,
        buffers.row_block_gene_masks,
        buffers.row_block_value_offsets
    };
    const u32 *inputs[] = {
        records.row_record_offsets,
        records.record_block_ids,
        records.record_gene_masks,
        records.record_value_offsets
    };
    for (std::size_t lhs = 0u; lhs < sizeof(outputs) / sizeof(outputs[0]); ++lhs) {
        if (outputs[lhs] == nullptr) continue;
        for (std::size_t rhs = lhs + 1u; rhs < sizeof(outputs) / sizeof(outputs[0]); ++rhs) {
            if (outputs[lhs] == outputs[rhs]) {
                return validation_error(validation_code::invalid_matrix_view, invalid_id,
                    "warp-tile output arrays must be distinct");
            }
        }
        for (const u32 *input : inputs) {
            if (input != nullptr && outputs[lhs] == input) {
                return validation_error(validation_code::invalid_matrix_view, invalid_id,
                    "warp-tile construction is out-of-place");
            }
        }
    }
    if (records.nnz_count != 0u && buffers.values == records.values) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "warp-tile value construction is out-of-place");
    }
    return validation_ok();
}

void set_view(
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_requirements &required,
    const warp_tile_buffers &buffers,
    warp_tile_view *out) {
    warp_tile_view result;
    result.tile_schema_version = warp_tile_schema_version;
    result.record_schema_version = records.record_schema_version;
    result.semantic_plan_schema_version = records.semantic_plan_schema_version;
    result.geometry_identity_version = records.geometry_identity_version;
    result.order_schema_version = order.order_schema_version;
    result.tile_identity = compute_tile_identity(records, order);
    result.feature_block_geometry_identity = records.feature_block_geometry_identity;
    result.ordering_identity = order.ordering_identity;
    result.global_row_begin = records.global_row_begin;
    result.full_row_count = records.full_row_count;
    result.row_count = records.row_count;
    result.feature_count = records.feature_count;
    result.feature_block_count = records.feature_block_count;
    result.tile_row_width = order.group_width;
    result.tile_count = tile_count_for_rows(records.row_count, order.group_width);
    result.nnz_count = records.nnz_count;
    result.tile_block_count = required.tile_block_count;
    result.row_block_entry_count = required.row_block_entry_count;
    result.value_size_bytes = records.value_size_bytes;
    result.feature_axis_fingerprint = records.feature_axis_fingerprint;
    result.feature_axis_fingerprint_version = records.feature_axis_fingerprint_version;
    result.row_domain_identity = records.row_domain_identity;
    result.tile_block_offsets = buffers.tile_block_offsets;
    result.tile_block_ids = buffers.tile_block_ids;
    result.tile_block_cell_masks = buffers.tile_block_cell_masks;
    result.block_row_entry_offsets = buffers.block_row_entry_offsets;
    result.row_block_gene_masks = buffers.row_block_gene_masks;
    result.row_block_value_offsets = buffers.row_block_value_offsets;
    result.values = buffers.values;
    *out = result;
}

} // namespace

u64 warp_tile_identity(
    const cell_block_record_view &records,
    const local_cell_order_view &order) noexcept {
    return compute_tile_identity(records, order);
}

validation_result query_warp_tile_requirements_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    warp_tile_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile requirements output is null");
    }
    const validation_result status = validate_inputs(plan, records, order);
    if (!status) return status;

    u32 tile_block_count = 0u, observed_entries = 0u;
    const u32 tile_count = tile_count_for_rows(records.row_count, order.group_width);
    for (u32 tile = 0u; tile < tile_count; ++tile) {
        const u32 tile_begin = tile * order.group_width;
        const u32 lane_count = std::min(order.group_width, records.row_count - tile_begin);
        u32 cursors[warp_tile_cell_mask_bits] = {}, ends[warp_tile_cell_mask_bits] = {};
        initialize_tile_cursors(records, order, tile_begin, lane_count, cursors, ends);
        for (u32 block = next_tile_block(cursors, ends, records, lane_count);
             block != invalid_id;
             block = next_tile_block(cursors, ends, records, lane_count)) {
            if (tile_block_count == std::numeric_limits<u32>::max()) {
                return validation_error(validation_code::integer_overflow, invalid_id,
                    "warp-tile block count overflows uint32");
            }
            ++tile_block_count;
            for (u32 lane = 0u; lane < lane_count; ++lane) {
                if (cursors[lane] != ends[lane]
                    && records.record_block_ids[cursors[lane]] == block) {
                    ++cursors[lane];
                    ++observed_entries;
                }
            }
        }
    }
    if (observed_entries != records.record_count) {
        return validation_error(validation_code::invalid_offsets, observed_entries,
            "warp-tile union traversal did not consume every cell-block record");
    }
    std::size_t value_bytes = 0u;
    if (multiply_overflows(records.nnz_count, records.value_size_bytes, &value_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "warp-tile value byte count overflows size_t");
    }
    warp_tile_requirements result;
    result.tile_block_offset_count = static_cast<std::size_t>(tile_count) + 1u;
    result.tile_block_count = tile_block_count;
    result.block_row_entry_offset_count = static_cast<std::size_t>(tile_block_count) + 1u;
    result.row_block_entry_count = records.record_count;
    result.row_block_value_offset_count = static_cast<std::size_t>(records.record_count) + 1u;
    result.value_bytes = value_bytes;
    *out = result;
    return validation_ok();
}

validation_result build_warp_tiles_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_buffers &buffers,
    warp_tile_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile view output is null");
    }
    warp_tile_requirements required;
    validation_result status = query_warp_tile_requirements_host(plan, records, order, &required);
    if (!status) return status;
    status = validate_output_buffers(records, required, buffers);
    if (!status) return status;

    const auto *source_values = static_cast<const unsigned char *>(records.values);
    auto *output_values = static_cast<unsigned char *>(buffers.values);
    const u32 tile_count = tile_count_for_rows(records.row_count, order.group_width);
    u32 tile_block = 0u, row_entry = 0u, output_value = 0u;
    for (u32 tile = 0u; tile < tile_count; ++tile) {
        buffers.tile_block_offsets[tile] = tile_block;
        const u32 tile_begin = tile * order.group_width;
        const u32 lane_count = std::min(order.group_width, records.row_count - tile_begin);
        u32 cursors[warp_tile_cell_mask_bits] = {}, ends[warp_tile_cell_mask_bits] = {};
        initialize_tile_cursors(records, order, tile_begin, lane_count, cursors, ends);
        for (u32 block = next_tile_block(cursors, ends, records, lane_count);
             block != invalid_id;
             block = next_tile_block(cursors, ends, records, lane_count)) {
            buffers.tile_block_ids[tile_block] = block;
            buffers.block_row_entry_offsets[tile_block] = row_entry;
            u32 cell_mask = 0u;
            for (u32 lane = 0u; lane < lane_count; ++lane) {
                if (cursors[lane] == ends[lane]
                    || records.record_block_ids[cursors[lane]] != block) continue;
                const u32 record = cursors[lane]++;
                cell_mask |= 1u << lane;
                buffers.row_block_gene_masks[row_entry] = records.record_gene_masks[record];
                buffers.row_block_value_offsets[row_entry] = output_value;
                const u32 source_begin = records.record_value_offsets[record];
                const u32 value_count = records.record_value_offsets[record + 1u] - source_begin;
                std::size_t copy_bytes = 0u;
                if (multiply_overflows(value_count, records.value_size_bytes, &copy_bytes)) {
                    return validation_error(validation_code::integer_overflow, record,
                        "warp-tile row-block value byte count overflows size_t");
                }
                if (copy_bytes != 0u) {
                    std::memcpy(output_values
                            + static_cast<std::size_t>(output_value) * records.value_size_bytes,
                        source_values
                            + static_cast<std::size_t>(source_begin) * records.value_size_bytes,
                        copy_bytes);
                }
                output_value += value_count;
                ++row_entry;
            }
            buffers.tile_block_cell_masks[tile_block] = cell_mask;
            ++tile_block;
        }
    }
    buffers.tile_block_offsets[tile_count] = tile_block;
    buffers.block_row_entry_offsets[tile_block] = row_entry;
    buffers.row_block_value_offsets[row_entry] = output_value;
    if (tile_block != required.tile_block_count
        || row_entry != required.row_block_entry_count
        || output_value != records.nnz_count) {
        return validation_error(validation_code::invalid_offsets, invalid_id,
            "warp-tile construction did not emit its queried requirements");
    }
    set_view(records, order, required, buffers, out);
    return validate_warp_tile_view_host(plan, records, order, *out);
}

validation_result validate_warp_tile_view_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles) {
    validation_result status = validate_inputs(plan, records, order);
    if (!status) return status;
    const u32 expected_tile_count = tile_count_for_rows(records.row_count, order.group_width);
    if (tiles.tile_schema_version != warp_tile_schema_version
        || tiles.record_schema_version != records.record_schema_version
        || tiles.semantic_plan_schema_version != records.semantic_plan_schema_version
        || tiles.geometry_identity_version != records.geometry_identity_version
        || tiles.order_schema_version != order.order_schema_version) {
        return validation_error(validation_code::unsupported_version,
            tiles.tile_schema_version, "warp-tile version is unsupported");
    }
    if (tiles.tile_identity != compute_tile_identity(records, order)
        || tiles.feature_block_geometry_identity != records.feature_block_geometry_identity
        || tiles.ordering_identity != order.ordering_identity
        || tiles.global_row_begin != records.global_row_begin
        || tiles.full_row_count != records.full_row_count
        || tiles.row_count != records.row_count
        || tiles.feature_count != records.feature_count
        || tiles.feature_block_count != records.feature_block_count
        || tiles.tile_row_width != order.group_width
        || tiles.tile_count != expected_tile_count
        || tiles.nnz_count != records.nnz_count
        || tiles.row_block_entry_count != records.record_count
        || tiles.value_size_bytes != records.value_size_bytes
        || tiles.feature_axis_fingerprint != records.feature_axis_fingerprint
        || tiles.feature_axis_fingerprint_version != records.feature_axis_fingerprint_version
        || tiles.row_domain_identity != records.row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "warp-tile identity is incompatible with its plan, records, or order");
    }
    if (tiles.tile_block_offsets == nullptr
        || tiles.block_row_entry_offsets == nullptr
        || tiles.row_block_value_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile terminal offsets are null");
    }
    if (tiles.tile_block_count != 0u
        && (tiles.tile_block_ids == nullptr || tiles.tile_block_cell_masks == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile block descriptors are null");
    }
    if (tiles.row_block_entry_count != 0u && tiles.row_block_gene_masks == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile row-block gene masks are null");
    }
    if (tiles.nnz_count != 0u && tiles.values == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile values are null");
    }
    if (tiles.tile_block_offsets[0] != 0u
        || tiles.block_row_entry_offsets[0] != 0u
        || tiles.row_block_value_offsets[0] != 0u) {
        return validation_error(validation_code::invalid_offsets, 0u,
            "warp-tile offsets must start at zero");
    }

    const auto *source_values = static_cast<const unsigned char *>(records.values);
    const auto *tile_values = static_cast<const unsigned char *>(tiles.values);
    u32 observed_entries = 0u;
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 descriptor_begin = tiles.tile_block_offsets[tile];
        const u32 descriptor_end = tiles.tile_block_offsets[tile + 1u];
        if (descriptor_end < descriptor_begin || descriptor_end > tiles.tile_block_count) {
            return validation_error(validation_code::invalid_offsets, tile,
                "warp-tile block offsets are not monotonic");
        }
        const u32 tile_begin = tile * tiles.tile_row_width;
        const u32 lane_count = std::min(tiles.tile_row_width, tiles.row_count - tile_begin);
        const u32 lane_mask = valid_lane_mask(lane_count);
        u32 cursors[warp_tile_cell_mask_bits] = {}, ends[warp_tile_cell_mask_bits] = {};
        initialize_tile_cursors(records, order, tile_begin, lane_count, cursors, ends);
        for (u32 descriptor = descriptor_begin; descriptor < descriptor_end; ++descriptor) {
            const u32 expected_block = next_tile_block(cursors, ends, records, lane_count);
            if (expected_block == invalid_id || tiles.tile_block_ids[descriptor] != expected_block) {
                return validation_error(validation_code::invalid_offsets, descriptor,
                    "warp-tile dictionary is not the exact sorted row-block union");
            }
            const u32 entry_begin = tiles.block_row_entry_offsets[descriptor];
            const u32 entry_end = tiles.block_row_entry_offsets[descriptor + 1u];
            if (entry_end < entry_begin || entry_end > tiles.row_block_entry_count) {
                return validation_error(validation_code::invalid_offsets, descriptor,
                    "warp-tile block-to-row-entry offsets are not monotonic");
            }
            const u32 cell_mask = tiles.tile_block_cell_masks[descriptor];
            if (cell_mask == 0u || (cell_mask & ~lane_mask) != 0u) {
                return validation_error(validation_code::invalid_matrix_view, descriptor,
                    "warp-tile cell mask is empty or outside the tile tail");
            }
            u32 expected_cell_mask = 0u, entry = entry_begin;
            for (u32 lane = 0u; lane < lane_count; ++lane) {
                if (cursors[lane] == ends[lane]
                    || records.record_block_ids[cursors[lane]] != expected_block) continue;
                const u32 record = cursors[lane]++;
                expected_cell_mask |= 1u << lane;
                if (entry >= entry_end
                    || tiles.row_block_gene_masks[entry] != records.record_gene_masks[record]) {
                    return validation_error(validation_code::invalid_plan_geometry, entry,
                        "warp-tile row-block gene mask disagrees with its source record");
                }
                const u32 tile_value_begin = tiles.row_block_value_offsets[entry];
                const u32 tile_value_end = tiles.row_block_value_offsets[entry + 1u];
                const u32 source_value_begin = records.record_value_offsets[record];
                const u32 source_value_count = records.record_value_offsets[record + 1u]
                    - source_value_begin;
                if (tile_value_end < tile_value_begin || tile_value_end > tiles.nnz_count
                    || tile_value_end - tile_value_begin != source_value_count
                    || source_value_count != popcount_u32(tiles.row_block_gene_masks[entry])) {
                    return validation_error(validation_code::invalid_offsets, entry,
                        "warp-tile value offsets disagree with row-block mask rank");
                }
                std::size_t compare_bytes = 0u;
                if (multiply_overflows(source_value_count, tiles.value_size_bytes, &compare_bytes)) {
                    return validation_error(validation_code::integer_overflow, entry,
                        "warp-tile row-block value byte count overflows size_t");
                }
                if (compare_bytes != 0u
                    && std::memcmp(tile_values
                            + static_cast<std::size_t>(tile_value_begin) * tiles.value_size_bytes,
                        source_values
                            + static_cast<std::size_t>(source_value_begin) * tiles.value_size_bytes,
                        compare_bytes) != 0) {
                    return validation_error(validation_code::invalid_matrix_view, entry,
                        "warp-tile compact value bytes disagree with their source record");
                }
                ++entry;
                ++observed_entries;
            }
            if (cell_mask != expected_cell_mask
                || entry != entry_end
                || entry_end - entry_begin != popcount_u32(cell_mask)) {
                return validation_error(validation_code::invalid_offsets, descriptor,
                    "warp-tile cell mask and compact row-entry range disagree");
            }
        }
        for (u32 lane = 0u; lane < lane_count; ++lane) {
            if (cursors[lane] != ends[lane]) {
                return validation_error(validation_code::invalid_offsets, tile,
                    "warp-tile dictionary omitted a source row block");
            }
        }
    }
    if (tiles.tile_block_offsets[tiles.tile_count] != tiles.tile_block_count
        || tiles.block_row_entry_offsets[tiles.tile_block_count]
            != tiles.row_block_entry_count
        || tiles.row_block_value_offsets[tiles.row_block_entry_count] != tiles.nnz_count
        || observed_entries != tiles.row_block_entry_count) {
        return validation_error(validation_code::invalid_offsets, invalid_id,
            "warp-tile terminal offsets disagree with declared counts");
    }
    std::size_t ignored = 0u;
    if (multiply_overflows(tiles.nnz_count, tiles.value_size_bytes, &ignored)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "warp-tile value byte count overflows size_t");
    }
    return validation_ok();
}

validation_result decode_warp_tiles_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    const warp_tile_decode_workspace &workspace,
    const warp_tile_decode_buffers &buffers,
    decoded_warp_tile_partition_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "decoded warp-tile partition output is null");
    }
    validation_result status = validate_warp_tile_view_host(plan, records, order, tiles);
    if (!status) return status;
    std::size_t value_bytes = 0u;
    if (multiply_overflows(tiles.nnz_count, tiles.value_size_bytes, &value_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "decoded warp-tile value byte count overflows size_t");
    }
    if (workspace.row_capacity < tiles.row_count
        || buffers.row_offset_capacity < static_cast<std::size_t>(tiles.row_count) + 1u
        || buffers.entry_capacity < tiles.nnz_count
        || buffers.value_capacity_bytes < value_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "decoded warp-tile output or workspace capacity is insufficient");
    }
    if (buffers.row_offsets == nullptr
        || (tiles.row_count != 0u && workspace.row_cursors == nullptr)
        || (tiles.nnz_count != 0u
            && (buffers.canonical_feature_ids == nullptr || buffers.values == nullptr))) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "decoded warp-tile output or workspace array is null");
    }
    if ((workspace.row_cursors != nullptr && workspace.row_cursors == buffers.row_offsets)
        || (workspace.row_cursors != nullptr
            && workspace.row_cursors == buffers.canonical_feature_ids)
        || (buffers.canonical_feature_ids != nullptr
            && buffers.row_offsets == buffers.canonical_feature_ids)
        || (tiles.nnz_count != 0u && buffers.values == tiles.values)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "warp-tile decoding requires distinct out-of-place arrays");
    }

    std::fill(buffers.row_offsets, buffers.row_offsets + tiles.row_count + 1u, 0u);
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 tile_begin = tile * tiles.tile_row_width;
        for (u32 descriptor = tiles.tile_block_offsets[tile];
             descriptor < tiles.tile_block_offsets[tile + 1u]; ++descriptor) {
            u32 entry = tiles.block_row_entry_offsets[descriptor];
            const u32 cell_mask = tiles.tile_block_cell_masks[descriptor];
            for (u32 lane = 0u; lane < tiles.tile_row_width; ++lane) {
                if ((cell_mask & (1u << lane)) == 0u) continue;
                const u32 row = order.row_permutation[tile_begin + lane];
                buffers.row_offsets[row + 1u] += popcount_u32(tiles.row_block_gene_masks[entry]);
                ++entry;
            }
        }
    }
    for (u32 row = 0u; row < tiles.row_count; ++row) {
        buffers.row_offsets[row + 1u] += buffers.row_offsets[row];
        workspace.row_cursors[row] = buffers.row_offsets[row];
    }
    if (buffers.row_offsets[tiles.row_count] != tiles.nnz_count) {
        return validation_error(validation_code::invalid_offsets, tiles.row_count,
            "decoded warp-tile row counts do not sum to nnz");
    }

    const auto *source_values = static_cast<const unsigned char *>(tiles.values);
    auto *output_values = static_cast<unsigned char *>(buffers.values);
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 tile_begin = tile * tiles.tile_row_width;
        for (u32 descriptor = tiles.tile_block_offsets[tile];
             descriptor < tiles.tile_block_offsets[tile + 1u]; ++descriptor) {
            const u32 block = tiles.tile_block_ids[descriptor];
            const u32 block_begin = plan.feature_block_offsets()[block];
            const u32 block_width = plan.feature_block_offsets()[block + 1u] - block_begin;
            u32 entry = tiles.block_row_entry_offsets[descriptor];
            const u32 cell_mask = tiles.tile_block_cell_masks[descriptor];
            for (u32 lane = 0u; lane < tiles.tile_row_width; ++lane) {
                if ((cell_mask & (1u << lane)) == 0u) continue;
                const u32 row = order.row_permutation[tile_begin + lane];
                const u32 gene_mask = tiles.row_block_gene_masks[entry];
                u32 source_value = tiles.row_block_value_offsets[entry];
                for (u32 local = 0u; local < block_width; ++local) {
                    if ((gene_mask & (1u << local)) == 0u) continue;
                    const u32 output_entry = workspace.row_cursors[row]++;
                    buffers.canonical_feature_ids[output_entry] =
                        plan.feature_permutation()[block_begin + local];
                    std::memcpy(output_values
                            + static_cast<std::size_t>(output_entry) * tiles.value_size_bytes,
                        source_values
                            + static_cast<std::size_t>(source_value) * tiles.value_size_bytes,
                        tiles.value_size_bytes);
                    ++source_value;
                }
                ++entry;
            }
        }
    }
    for (u32 row = 0u; row < tiles.row_count; ++row) {
        if (workspace.row_cursors[row] != buffers.row_offsets[row + 1u]) {
            return validation_error(validation_code::invalid_offsets, row,
                "decoded warp-tile row cursor disagrees with its output range");
        }
    }

    decoded_warp_tile_partition_view result;
    result.global_row_begin = tiles.global_row_begin;
    result.full_row_count = tiles.full_row_count;
    result.row_count = tiles.row_count;
    result.feature_count = tiles.feature_count;
    result.nnz_count = tiles.nnz_count;
    result.value_size_bytes = tiles.value_size_bytes;
    result.feature_axis_fingerprint = tiles.feature_axis_fingerprint;
    result.feature_axis_fingerprint_version = tiles.feature_axis_fingerprint_version;
    result.row_domain_identity = tiles.row_domain_identity;
    result.row_offsets = buffers.row_offsets;
    result.canonical_feature_ids = buffers.canonical_feature_ids;
    result.values = buffers.values;
    *out = result;
    return validation_ok();
}

validation_result evaluate_warp_tile_metrics_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    warp_tile_metrics *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "warp-tile metrics output is null");
    }
    const validation_result status = validate_warp_tile_view_host(plan, records, order, tiles);
    if (!status) return status;
    warp_tile_metrics result;
    result.tile_count = tiles.tile_count;
    result.tile_block_count = tiles.tile_block_count;
    result.row_block_entry_count = tiles.row_block_entry_count;
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        result.maximum_tile_block_union = std::max(result.maximum_tile_block_union,
            tiles.tile_block_offsets[tile + 1u] - tiles.tile_block_offsets[tile]);
    }
    result.metadata_bytes =
        (static_cast<u64>(tiles.tile_count) + 1u) * sizeof(u32)
        + static_cast<u64>(tiles.tile_block_count) * 2u * sizeof(u32)
        + (static_cast<u64>(tiles.tile_block_count) + 1u) * sizeof(u32)
        + static_cast<u64>(tiles.row_block_entry_count) * sizeof(u32)
        + (static_cast<u64>(tiles.row_block_entry_count) + 1u) * sizeof(u32);
    result.value_bytes = static_cast<u64>(tiles.nnz_count) * tiles.value_size_bytes;
    result.total_bytes = result.metadata_bytes + result.value_bytes;
    result.source_record_metadata_bytes =
        (static_cast<u64>(records.row_count) + 1u) * sizeof(u32)
        + static_cast<u64>(records.record_count) * 2u * sizeof(u32)
        + (static_cast<u64>(records.record_count) + 1u) * sizeof(u32);
    *out = result;
    return validation_ok();
}

} // namespace cellpack
