#include "CellPack/local_cell_ordering.hh"

#include <algorithm>
#include <limits>

namespace cellpack {
namespace {

u64 splitmix64(u64 value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

bool valid_kind(local_cell_order_kind kind) noexcept {
    return kind == local_cell_order_kind::inferred_minhash
        || kind == local_cell_order_kind::original
        || kind == local_cell_order_kind::deterministic_random
        || kind == local_cell_order_kind::row_nnz_descending;
}

validation_result validate_config(const local_cell_order_config &config) {
    if (!valid_kind(config.kind)) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "local-cell order kind is unsupported");
    }
    if (config.window_size == 0u || config.group_width == 0u
        || config.group_width > config.window_size
        || config.window_size % config.group_width != 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "local-cell windows must contain an integral nonzero number of groups");
    }
    return validation_ok();
}

validation_result validate_records(const cell_block_record_view &records) {
    if (records.record_schema_version != cell_block_record_schema_version
        || records.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || records.geometry_identity_version != feature_block_geometry_identity_version) {
        return validation_error(validation_code::unsupported_version,
            records.record_schema_version, "cell-block record version is unsupported");
    }
    if (records.feature_block_geometry_identity == 0u || records.row_domain_identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "cell-block record semantic identity is missing");
    }
    const u64 row_end = records.global_row_begin + static_cast<u64>(records.row_count);
    if (row_end < records.global_row_begin || row_end > records.full_row_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "cell-block record partition is outside the row domain");
    }
    if (records.row_record_offsets == nullptr || records.record_value_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record offsets are null");
    }
    if (records.record_count != 0u && records.record_block_ids == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record block ids are null");
    }
    if (records.row_record_offsets[0] != 0u || records.record_value_offsets[0] != 0u) {
        return validation_error(validation_code::invalid_offsets, 0u,
            "cell-block record offsets must start at zero");
    }
    for (u32 row = 0u; row < records.row_count; ++row) {
        const u32 begin = records.row_record_offsets[row];
        const u32 end = records.row_record_offsets[row + 1u];
        if (end < begin || end > records.record_count) {
            return validation_error(validation_code::invalid_offsets, row,
                "cell-block row offsets are not monotonic");
        }
        u32 previous = invalid_id;
        for (u32 record = begin; record < end; ++record) {
            const u32 block = records.record_block_ids[record];
            if (block >= records.feature_block_count
                || (previous != invalid_id && block <= previous)) {
                return validation_error(validation_code::invalid_offsets, record,
                    "cell-block ids are not strictly increasing per row");
            }
            if (records.record_value_offsets[record + 1u]
                < records.record_value_offsets[record]) {
                return validation_error(validation_code::invalid_offsets, record,
                    "cell-block value offsets are not monotonic");
            }
            previous = block;
        }
    }
    if (records.row_record_offsets[records.row_count] != records.record_count
        || records.record_value_offsets[records.record_count] != records.nnz_count) {
        return validation_error(validation_code::invalid_offsets, records.record_count,
            "cell-block record terminal offset disagrees with its count");
    }
    return validation_ok();
}

u64 inferred_signature(
    const u32 *block_ids,
    u32 begin,
    u32 end,
    u64 seed) noexcept {
    if (begin == end) return std::numeric_limits<u64>::max();
    u64 packed = 0u;
    for (u32 lane = 0u; lane < local_cell_signature_lane_count; ++lane) {
        u64 minimum = std::numeric_limits<u64>::max();
        const u64 lane_seed = splitmix64(seed ^ (static_cast<u64>(lane) << 32u));
        for (u32 record = begin; record < end; ++record) {
            minimum = std::min(minimum,
                splitmix64(lane_seed ^ static_cast<u64>(block_ids[record])));
        }
        packed = (packed << 16u) | ((minimum >> 48u) & 0xffffu);
    }
    return packed;
}

u64 compute_ordering_identity(
    const cell_block_record_view &records,
    const local_cell_order_config &config) noexcept {
    u64 identity = splitmix64(records.feature_block_geometry_identity);
    identity = splitmix64(identity ^ (static_cast<u64>(local_cell_order_schema_version) << 32u)
        ^ local_cell_signature_algorithm_version);
    identity = splitmix64(identity ^ records.row_domain_identity);
    identity = splitmix64(identity ^ records.global_row_begin);
    identity = splitmix64(identity ^ (static_cast<u64>(records.row_count) << 32u)
        ^ records.full_row_count);
    identity = splitmix64(identity ^ (static_cast<u64>(config.window_size) << 32u)
        ^ config.group_width);
    identity = splitmix64(identity ^ config.seed ^ static_cast<u32>(config.kind));
    return identity == 0u ? 1u : identity;
}

void row_keys(
    const cell_block_record_view &records,
    const local_cell_order_config &config,
    u32 row,
    u64 *primary,
    u32 *secondary,
    u32 *active_count,
    u32 *nnz_count) noexcept {
    const u32 begin = records.row_record_offsets[row];
    const u32 end = records.row_record_offsets[row + 1u];
    const u32 active = end - begin;
    const u32 nnz = records.record_value_offsets[end] - records.record_value_offsets[begin];
    *active_count = active;
    *nnz_count = nnz;
    *secondary = active;
    switch (config.kind) {
    case local_cell_order_kind::inferred_minhash:
        *primary = inferred_signature(records.record_block_ids, begin, end, config.seed);
        break;
    case local_cell_order_kind::original:
        *primary = row;
        *secondary = 0u;
        break;
    case local_cell_order_kind::deterministic_random:
        *primary = splitmix64(config.seed ^ (records.global_row_begin + row));
        *secondary = 0u;
        break;
    case local_cell_order_kind::row_nnz_descending:
        *primary = std::numeric_limits<u64>::max() - nnz;
        break;
    }
}

validation_result validate_buffers(
    u32 row_count,
    const local_cell_order_buffers &buffers) {
    if (buffers.row_capacity < row_count) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "local-cell order output capacity is insufficient");
    }
    if (row_count != 0u
        && (buffers.primary_keys == nullptr || buffers.secondary_keys == nullptr
            || buffers.active_block_counts == nullptr || buffers.row_nnz_counts == nullptr
            || buffers.row_permutation == nullptr
            || buffers.inverse_row_permutation == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "local-cell order output array is null");
    }
    return validation_ok();
}

void set_view(
    const cell_block_record_view &records,
    const local_cell_order_config &config,
    const local_cell_order_buffers &buffers,
    local_cell_order_view *out) {
    local_cell_order_view result;
    result.order_schema_version = local_cell_order_schema_version;
    result.signature_algorithm_version = local_cell_signature_algorithm_version;
    result.kind = config.kind;
    result.window_size = config.window_size;
    result.group_width = config.group_width;
    result.seed = config.seed;
    result.ordering_identity = compute_ordering_identity(records, config);
    result.global_row_begin = records.global_row_begin;
    result.full_row_count = records.full_row_count;
    result.row_count = records.row_count;
    result.feature_block_count = records.feature_block_count;
    result.feature_block_geometry_identity = records.feature_block_geometry_identity;
    result.row_domain_identity = records.row_domain_identity;
    result.primary_keys = buffers.primary_keys;
    result.secondary_keys = buffers.secondary_keys;
    result.active_block_counts = buffers.active_block_counts;
    result.row_nnz_counts = buffers.row_nnz_counts;
    result.row_permutation = buffers.row_permutation;
    result.inverse_row_permutation = buffers.inverse_row_permutation;
    *out = result;
}

} // namespace

u64 local_cell_order_identity(
    const cell_block_record_view &records,
    const local_cell_order_config &config) noexcept {
    return compute_ordering_identity(records, config);
}

validation_result query_local_cell_order_requirements_host(
    const cell_block_record_view &records,
    local_cell_order_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "local-cell order requirements output is null");
    }
    const validation_result status = validate_records(records);
    if (!status) return status;
    out->row_capacity = records.row_count;
    out->block_epoch_capacity = records.feature_block_count;
    return validation_ok();
}

validation_result build_local_cell_order_host(
    const cell_block_record_view &records,
    const local_cell_order_config &config,
    const local_cell_order_buffers &buffers,
    local_cell_order_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "local-cell order view output is null");
    }
    validation_result status = validate_records(records);
    if (!status) return status;
    status = validate_config(config);
    if (!status) return status;
    status = validate_buffers(records.row_count, buffers);
    if (!status) return status;

    for (u32 row = 0u; row < records.row_count; ++row) {
        row_keys(records, config, row, buffers.primary_keys + row,
            buffers.secondary_keys + row, buffers.active_block_counts + row,
            buffers.row_nnz_counts + row);
        buffers.row_permutation[row] = row;
    }
    for (u32 window_begin = 0u; window_begin < records.row_count;) {
        const u32 remaining = records.row_count - window_begin;
        const u32 window_end = window_begin + std::min(config.window_size, remaining);
        std::sort(buffers.row_permutation + window_begin,
            buffers.row_permutation + window_end,
            [&](u32 lhs, u32 rhs) {
                if (buffers.primary_keys[lhs] != buffers.primary_keys[rhs])
                    return buffers.primary_keys[lhs] < buffers.primary_keys[rhs];
                if (buffers.secondary_keys[lhs] != buffers.secondary_keys[rhs])
                    return buffers.secondary_keys[lhs] < buffers.secondary_keys[rhs];
                return lhs < rhs;
            });
        window_begin = window_end;
    }
    for (u32 execution = 0u; execution < records.row_count; ++execution) {
        buffers.inverse_row_permutation[buffers.row_permutation[execution]] = execution;
    }
    set_view(records, config, buffers, out);
    return validate_local_cell_order_view_host(records, *out);
}

validation_result validate_local_cell_order_view_host(
    const cell_block_record_view &records,
    const local_cell_order_view &order) {
    validation_result status = validate_records(records);
    if (!status) return status;
    local_cell_order_config config{order.kind, order.window_size, order.group_width, order.seed};
    status = validate_config(config);
    if (!status) return status;
    if (order.order_schema_version != local_cell_order_schema_version
        || order.signature_algorithm_version != local_cell_signature_algorithm_version) {
        return validation_error(validation_code::unsupported_version,
            order.order_schema_version, "local-cell order version is unsupported");
    }
    if (order.ordering_identity != compute_ordering_identity(records, config)
        || order.global_row_begin != records.global_row_begin
        || order.full_row_count != records.full_row_count
        || order.row_count != records.row_count
        || order.feature_block_count != records.feature_block_count
        || order.feature_block_geometry_identity != records.feature_block_geometry_identity
        || order.row_domain_identity != records.row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "local-cell order identity is incompatible with its records");
    }
    if (order.row_count != 0u
        && (order.primary_keys == nullptr || order.secondary_keys == nullptr
            || order.active_block_counts == nullptr || order.row_nnz_counts == nullptr
            || order.row_permutation == nullptr || order.inverse_row_permutation == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "local-cell order array is null");
    }
    for (u32 row = 0u; row < order.row_count; ++row) {
        u64 primary = 0u;
        u32 secondary = 0u, active = 0u, nnz = 0u;
        row_keys(records, config, row, &primary, &secondary, &active, &nnz);
        if (order.primary_keys[row] != primary || order.secondary_keys[row] != secondary
            || order.active_block_counts[row] != active || order.row_nnz_counts[row] != nnz) {
            return validation_error(validation_code::invalid_signature, row,
                "local-cell row signature or count is incorrect");
        }
    }
    for (u32 execution = 0u; execution < order.row_count; ++execution) {
        const u32 row = order.row_permutation[execution];
        if (row >= order.row_count || order.inverse_row_permutation[row] != execution) {
            return validation_error(validation_code::invalid_permutation, execution,
                "local-cell permutation and inverse disagree");
        }
        if (row / config.window_size != execution / config.window_size) {
            return validation_error(validation_code::invalid_permutation, execution,
                "local-cell permutation crosses a window boundary");
        }
        if (execution % config.window_size != 0u) {
            const u32 previous = order.row_permutation[execution - 1u];
            const bool sorted = order.primary_keys[previous] < order.primary_keys[row]
                || (order.primary_keys[previous] == order.primary_keys[row]
                    && (order.secondary_keys[previous] < order.secondary_keys[row]
                        || (order.secondary_keys[previous] == order.secondary_keys[row]
                            && previous < row)));
            if (!sorted) {
                return validation_error(validation_code::invalid_permutation, execution,
                    "local-cell order is not the deterministic tuple order");
            }
        }
    }
    return validation_ok();
}

validation_result evaluate_local_cell_order_metrics_host(
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const local_cell_order_metric_workspace &workspace,
    local_cell_order_metrics *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "local-cell order metrics output is null");
    }
    validation_result status = validate_local_cell_order_view_host(records, order);
    if (!status) return status;
    if (workspace.block_epoch_capacity < records.feature_block_count
        || (records.feature_block_count != 0u && workspace.block_epochs == nullptr)) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "local-cell metric workspace is insufficient");
    }
    if (records.feature_block_count != 0u) {
        std::fill(workspace.block_epochs,
            workspace.block_epochs + records.feature_block_count, 0u);
    }
    local_cell_order_metrics result;
    result.group_width = order.group_width;
    u32 epoch = 0u;
    for (u32 group_begin = 0u; group_begin < order.row_count;) {
        const u32 group_end = group_begin
            + std::min(order.group_width, order.row_count - group_begin);
        if (++epoch == 0u) {
            if (records.feature_block_count != 0u) {
                std::fill(workspace.block_epochs,
                    workspace.block_epochs + records.feature_block_count, 0u);
            }
            epoch = 1u;
        }
        u32 group_union = 0u;
        for (u32 execution = group_begin; execution < group_end; ++execution) {
            const u32 row = order.row_permutation[execution];
            const u32 begin = records.row_record_offsets[row];
            const u32 end = records.row_record_offsets[row + 1u];
            result.total_active_block_references += end - begin;
            for (u32 record = begin; record < end; ++record) {
                const u32 block = records.record_block_ids[record];
                if (workspace.block_epochs[block] != epoch) {
                    workspace.block_epochs[block] = epoch;
                    ++group_union;
                }
            }
        }
        result.total_group_block_union_references += group_union;
        result.maximum_group_block_union = std::max(result.maximum_group_block_union, group_union);
        ++result.group_count;
        group_begin = group_end;
    }
    result.block_id_metadata_bytes = result.total_group_block_union_references * sizeof(u32);
    *out = result;
    return validation_ok();
}

} // namespace cellpack
