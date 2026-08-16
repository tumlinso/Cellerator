#include "CellPack/evaluator.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cellpack {
namespace {

validation_result validate_permutation_pair(
    const u32 *permutation,
    const u32 *inverse,
    u32 count,
    const char *message) {
    if (permutation == nullptr && inverse == nullptr) return validation_ok();
    if (permutation == nullptr || inverse == nullptr) {
        return validation_error(validation_code::invalid_permutation, invalid_id, message);
    }
    for (u32 execution = 0u; execution < count; ++execution) {
        const u32 canonical = permutation[execution];
        if (canonical >= count || inverse[canonical] != execution) {
            return validation_error(validation_code::invalid_permutation, execution, message);
        }
    }
    for (u32 canonical = 0u; canonical < count; ++canonical) {
        const u32 execution = inverse[canonical];
        if (execution >= count || permutation[execution] != canonical) {
            return validation_error(validation_code::invalid_permutation, canonical, message);
        }
    }
    return validation_ok();
}

validation_result validate_boundaries(
    const u32 *offsets,
    u32 group_count,
    u32 axis_count,
    const char *message) {
    if (axis_count == 0u) {
        return group_count == 0u
            ? validation_ok()
            : validation_error(validation_code::invalid_plan_geometry, group_count, message);
    }
    if (group_count == 0u || offsets == nullptr || offsets[0] != 0u || offsets[group_count] != axis_count) {
        return validation_error(validation_code::invalid_plan_geometry, group_count, message);
    }
    for (u32 group = 0u; group < group_count; ++group) {
        if (offsets[group + 1u] <= offsets[group]) {
            return validation_error(validation_code::invalid_plan_geometry, group, message);
        }
    }
    return validation_ok();
}

u32 find_group(const u32 *offsets, u32 group_count, u32 execution_position) {
    u32 low = 0u, high = group_count;
    while (low + 1u < high) {
        const u32 middle = low + ((high - low) >> 1u);
        if (execution_position < offsets[middle]) high = middle;
        else low = middle;
    }
    return low;
}

u32 execution_position(const u32 *inverse, u32 canonical) {
    return inverse == nullptr ? canonical : inverse[canonical];
}

bool checked_mul_size(std::size_t lhs, std::size_t rhs, std::size_t *out) {
    if (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs) return false;
    *out = lhs * rhs;
    return true;
}

bool checked_add_size(std::size_t lhs, std::size_t rhs, std::size_t *out) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

bool checked_mul_u64(u64 lhs, u64 rhs, u64 *out) {
    if (lhs != 0u && rhs > std::numeric_limits<u64>::max() / lhs) return false;
    *out = lhs * rhs;
    return true;
}

bool checked_add_u64(u64 lhs, u64 rhs, u64 *out) {
    if (rhs > std::numeric_limits<u64>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

void observe(count_distribution *distribution, u64 value) {
    if (distribution->sample_count == 0u) {
        distribution->minimum = value;
        distribution->maximum = value;
    } else {
        distribution->minimum = std::min(distribution->minimum, value);
        distribution->maximum = std::max(distribution->maximum, value);
    }
    ++distribution->sample_count;
    distribution->total += value;
    const double converted = static_cast<double>(value);
    distribution->squared_total += converted * converted;
}

void observe(real_distribution *distribution, double value) {
    if (distribution->sample_count == 0u) {
        distribution->minimum = value;
        distribution->maximum = value;
    } else {
        distribution->minimum = std::min(distribution->minimum, value);
        distribution->maximum = std::max(distribution->maximum, value);
    }
    ++distribution->sample_count;
    distribution->total += value;
    distribution->squared_total += value * value;
}

validation_result validate_buffers(
    const packing_evaluation_requirements &requirements,
    const packing_evaluation_workspace_view &workspace,
    const packing_occupancy_buffers &buffers) {
    if (requirements.workspace_entry_capacity != 0u
        && (workspace.entries == nullptr || workspace.entry_capacity < requirements.workspace_entry_capacity)) {
        return validation_error(validation_code::insufficient_capacity, workspace.entry_capacity, "PackingPlan evaluator workspace is too small");
    }
    if (requirements.occupied_tile_capacity != 0u
        && (buffers.occupied_tiles == nullptr || buffers.occupied_tile_capacity < requirements.occupied_tile_capacity)) {
        return validation_error(validation_code::insufficient_capacity, buffers.occupied_tile_capacity, "PackingPlan occupied-tile output is too small");
    }
    if (requirements.execution_row_capacity != 0u
        && (buffers.active_feature_blocks_per_execution_row == nullptr
            || buffers.execution_row_capacity < requirements.execution_row_capacity)) {
        return validation_error(validation_code::insufficient_capacity, buffers.execution_row_capacity, "PackingPlan per-row output is too small");
    }
    if (requirements.row_group_capacity != 0u
        && (buffers.row_groups == nullptr || buffers.row_group_capacity < requirements.row_group_capacity)) {
        return validation_error(validation_code::insufficient_capacity, buffers.row_group_capacity, "PackingPlan row-group output is too small");
    }
    return validation_ok();
}

} // namespace

validation_result validate_csr_support_view(const csr_support_view &source) {
    if (source.row_count != 0u && source.row_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "CSR support row offsets are null");
    }
    if (source.nnz_count != 0u && source.feature_ids == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "CSR support feature ids are null");
    }
    if (source.row_count == 0u) {
        return source.nnz_count == 0u
            ? validation_ok()
            : validation_error(validation_code::invalid_matrix_view, invalid_id, "empty CSR support row axis cannot contain nonzeros");
    }
    if (source.feature_count == 0u && source.nnz_count != 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "empty CSR support feature axis cannot contain nonzeros");
    }
    if (source.row_offsets[0] != 0u) {
        return validation_error(validation_code::invalid_matrix_view, 0u, "CSR support row offsets must start at zero");
    }
    for (u32 row = 0u; row < source.row_count; ++row) {
        const u32 begin = source.row_offsets[row], end = source.row_offsets[row + 1u];
        if (end < begin || end > source.nnz_count) {
            return validation_error(validation_code::invalid_matrix_view, row, "CSR support row offsets are not monotonic");
        }
        u32 previous_feature = invalid_id;
        for (u32 entry = begin; entry < end; ++entry) {
            const u32 feature = source.feature_ids[entry];
            if (feature >= source.feature_count) {
                return validation_error(validation_code::invalid_matrix_view, entry, "CSR support feature id is outside matrix bounds");
            }
            if (previous_feature != invalid_id && feature <= previous_feature) {
                return validation_error(validation_code::invalid_matrix_view, entry, "CSR support feature ids must be strictly increasing within each row");
            }
            previous_feature = feature;
        }
    }
    if (source.row_offsets[source.row_count] != source.nnz_count) {
        return validation_error(validation_code::invalid_matrix_view, source.row_count, "CSR support final row offset does not match nnz count");
    }
    return validation_ok();
}

validation_result validate_packing_plan_view(const packing_plan_view &plan) {
    validation_result result = validate_permutation_pair(
        plan.row_permutation,
        plan.inverse_row_permutation,
        plan.row_count,
        "PackingPlan row permutation pair is invalid");
    if (!result) return result;
    result = validate_permutation_pair(
        plan.feature_permutation,
        plan.inverse_feature_permutation,
        plan.feature_count,
        "PackingPlan feature permutation pair is invalid");
    if (!result) return result;
    result = validate_boundaries(
        plan.row_group_offsets,
        plan.row_group_count,
        plan.row_count,
        "PackingPlan row-group boundaries do not partition the row execution axis");
    if (!result) return result;
    return validate_boundaries(
        plan.feature_block_offsets,
        plan.feature_block_count,
        plan.feature_count,
        "PackingPlan feature-block boundaries do not partition the feature execution axis");
}

validation_result prepare_csr_support(const csr_support_view &source, prepared_csr_support *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "prepared CSR support output is null");
    }
    validation_result result = validate_csr_support_view(source);
    if (!result) return result;
    out->support = source;
    out->validated = true;
    return validation_ok();
}

packing_plan_view make_packing_plan_view(const static_plan &plan) {
    packing_plan_view view;
    view.row_count = plan.desc.row_count;
    view.feature_count = plan.desc.feature_count;
    view.row_permutation = plan.row_permutation.empty() ? nullptr : plan.row_permutation.data();
    view.inverse_row_permutation = plan.inverse_row_permutation.empty() ? nullptr : plan.inverse_row_permutation.data();
    view.feature_permutation = plan.feature_permutation.empty() ? nullptr : plan.feature_permutation.data();
    view.inverse_feature_permutation = plan.inverse_feature_permutation.empty() ? nullptr : plan.inverse_feature_permutation.data();
    view.row_group_count = static_cast<u32>(plan.row_groups.size());
    view.row_group_offsets = plan.row_group_offsets.empty() ? nullptr : plan.row_group_offsets.data();
    view.feature_block_count = static_cast<u32>(plan.modules.size());
    view.feature_block_offsets = plan.feature_block_offsets.empty() ? nullptr : plan.feature_block_offsets.data();
    return view;
}

validation_result query_packing_evaluation_requirements(
    const csr_support_view &source,
    const packing_plan_view &plan,
    packing_evaluation_requirements *out) {
    prepared_csr_support prepared;
    validation_result result = prepare_csr_support(source, &prepared);
    if (!result) return result;
    return query_packing_evaluation_requirements(prepared, plan, out);
}

validation_result query_packing_evaluation_requirements(
    const prepared_csr_support &prepared,
    const packing_plan_view &plan,
    packing_evaluation_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "PackingPlan evaluation requirements output is null");
    }
    if (!prepared.validated) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "CSR support must be prepared before repeated PackingPlan evaluation");
    }
    const csr_support_view &source = prepared.support;
    validation_result result;
    result = validate_packing_plan_view(plan);
    if (!result) return result;
    if (source.row_count != plan.row_count || source.feature_count != plan.feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "CSR support dimensions do not match PackingPlan geometry");
    }

    packing_evaluation_requirements requirements;
    requirements.workspace_entry_capacity = source.nnz_count;
    requirements.logical_tile_count = static_cast<u64>(plan.row_group_count) * static_cast<u64>(plan.feature_block_count);
    requirements.occupied_tile_capacity = static_cast<u32>(std::min<u64>(source.nnz_count, requirements.logical_tile_count));
    requirements.execution_row_capacity = source.row_count;
    requirements.row_group_capacity = plan.row_group_count;

    if (!checked_mul_size(source.nnz_count, sizeof(packing_evaluation_entry), &requirements.temporary_workspace_bytes)) {
        return validation_error(validation_code::integer_overflow, source.nnz_count, "PackingPlan temporary workspace byte count overflows size_t");
    }
    std::size_t tile_bytes = 0u, row_bytes = 0u, group_bytes = 0u, output_bytes = 0u;
    if (!checked_mul_size(requirements.occupied_tile_capacity, sizeof(occupied_tile_occupancy), &tile_bytes)
        || !checked_mul_size(requirements.execution_row_capacity, sizeof(u32), &row_bytes)
        || !checked_mul_size(requirements.row_group_capacity, sizeof(row_group_occupancy), &group_bytes)
        || !checked_add_size(tile_bytes, row_bytes, &output_bytes)
        || !checked_add_size(output_bytes, group_bytes, &output_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id, "PackingPlan output buffer byte count overflows size_t");
    }
    requirements.output_buffer_bytes = output_bytes;
    *out = requirements;
    return validation_ok();
}

validation_result evaluate_packing_plan(
    const csr_support_view &source,
    const packing_plan_view &plan,
    const packing_evaluation_workspace_view &workspace,
    const packing_occupancy_buffers &buffers,
    packing_occupancy_result *out) {
    prepared_csr_support prepared;
    validation_result result = prepare_csr_support(source, &prepared);
    if (!result) return result;
    return evaluate_packing_plan(prepared, plan, workspace, buffers, out);
}

validation_result evaluate_packing_plan(
    const prepared_csr_support &prepared,
    const packing_plan_view &plan,
    const packing_evaluation_workspace_view &workspace,
    const packing_occupancy_buffers &buffers,
    packing_occupancy_result *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "PackingPlan occupancy output is null");
    }
    const csr_support_view &source = prepared.support;
    packing_evaluation_requirements requirements;
    validation_result result = query_packing_evaluation_requirements(prepared, plan, &requirements);
    if (!result) return result;
    result = validate_buffers(requirements, workspace, buffers);
    if (!result) return result;

    for (u32 row = 0u; row < source.row_count; ++row) {
        buffers.active_feature_blocks_per_execution_row[row] = 0u;
    }
    for (u32 group = 0u; group < plan.row_group_count; ++group) {
        row_group_occupancy &group_out = buffers.row_groups[group];
        group_out = row_group_occupancy{};
        group_out.row_group = group;
        group_out.row_count = plan.row_group_offsets[group + 1u] - plan.row_group_offsets[group];
    }

    u32 entry_count = 0u;
    for (u32 canonical_row = 0u; canonical_row < source.row_count; ++canonical_row) {
        const u32 execution_row = execution_position(plan.inverse_row_permutation, canonical_row);
        const u32 row_group = find_group(plan.row_group_offsets, plan.row_group_count, execution_row);
        for (u32 entry = source.row_offsets[canonical_row]; entry < source.row_offsets[canonical_row + 1u]; ++entry) {
            const u32 execution_feature = execution_position(plan.inverse_feature_permutation, source.feature_ids[entry]);
            const u32 feature_block = find_group(plan.feature_block_offsets, plan.feature_block_count, execution_feature);
            packing_evaluation_entry &mapped = workspace.entries[entry_count++];
            mapped.tile_id = static_cast<u64>(row_group) * static_cast<u64>(plan.feature_block_count)
                + static_cast<u64>(feature_block);
            mapped.execution_row = execution_row;
            mapped.reserved = 0u;
        }
    }

    if (entry_count > 1u) {
        std::sort(workspace.entries, workspace.entries + entry_count, [](const packing_evaluation_entry &lhs, const packing_evaluation_entry &rhs) {
            if (lhs.tile_id != rhs.tile_id) return lhs.tile_id < rhs.tile_id;
            return lhs.execution_row < rhs.execution_row;
        });
    }

    packing_occupancy_result evaluation;
    evaluation.row_count = source.row_count;
    evaluation.feature_count = source.feature_count;
    evaluation.row_group_count = plan.row_group_count;
    evaluation.feature_block_count = plan.feature_block_count;
    evaluation.occupied_tiles = buffers.occupied_tiles;
    evaluation.active_feature_blocks_per_execution_row = buffers.active_feature_blocks_per_execution_row;
    evaluation.row_groups = buffers.row_groups;
    evaluation.totals.total_nnz = source.nnz_count;
    evaluation.totals.logical_tile_count = requirements.logical_tile_count;

    u32 begin = 0u;
    while (begin < entry_count) {
        u32 end = begin + 1u;
        while (end < entry_count && workspace.entries[end].tile_id == workspace.entries[begin].tile_id) ++end;
        const u64 tile_id = workspace.entries[begin].tile_id;
        const u32 row_group = static_cast<u32>(tile_id / plan.feature_block_count);
        const u32 feature_block = static_cast<u32>(tile_id % plan.feature_block_count);
        const u32 row_count = plan.row_group_offsets[row_group + 1u] - plan.row_group_offsets[row_group];
        const u32 feature_count = plan.feature_block_offsets[feature_block + 1u] - plan.feature_block_offsets[feature_block];
        const u64 logical_slots = static_cast<u64>(row_count) * static_cast<u64>(feature_count);
        const u64 tile_nnz = static_cast<u64>(end - begin);
        if (tile_nnz > logical_slots) {
            return validation_error(validation_code::invalid_matrix_view, begin, "PackingPlan tile contains more structural entries than logical slots");
        }

        u32 participating_rows = 0u, previous_row = invalid_id;
        for (u32 entry = begin; entry < end; ++entry) {
            const u32 execution_row = workspace.entries[entry].execution_row;
            if (execution_row != previous_row) {
                ++participating_rows;
                ++buffers.active_feature_blocks_per_execution_row[execution_row];
                previous_row = execution_row;
            }
        }

        occupied_tile_occupancy &tile = buffers.occupied_tiles[evaluation.occupied_tile_count++];
        tile.row_group = row_group;
        tile.feature_block = feature_block;
        tile.participating_rows = participating_rows;
        tile.reserved = 0u;
        tile.nnz = tile_nnz;
        tile.logical_slots = logical_slots;
        tile.dense_padding = logical_slots - tile_nnz;
        tile.density = logical_slots == 0u ? 0.0 : static_cast<double>(tile_nnz) / static_cast<double>(logical_slots);
        tile.row_participation = row_count == 0u ? 0.0 : static_cast<double>(participating_rows) / static_cast<double>(row_count);

        row_group_occupancy &group = buffers.row_groups[row_group];
        ++group.active_feature_blocks;
        group.nnz += tile_nnz;
        group.participating_row_block_references += participating_rows;
        group.occupied_dense_slots += logical_slots;
        group.dense_padding += tile.dense_padding;

        evaluation.totals.occupied_dense_slots += logical_slots;
        evaluation.totals.dense_padding += tile.dense_padding;
        evaluation.totals.row_active_block_references += participating_rows;
        observe(&evaluation.nnz_per_occupied_tile, tile_nnz);
        observe(&evaluation.tile_density, tile.density);
        observe(&evaluation.participating_rows_per_occupied_tile, participating_rows);
        observe(&evaluation.feature_block_reuse, tile.row_participation);
        observe(&evaluation.dense_padding_per_occupied_tile, tile.dense_padding);
        begin = end;
    }

    evaluation.totals.occupied_tile_count = evaluation.occupied_tile_count;
    evaluation.totals.empty_tile_count = evaluation.totals.logical_tile_count - evaluation.totals.occupied_tile_count;
    evaluation.totals.row_group_active_block_references = evaluation.totals.occupied_tile_count;
    for (u32 row = 0u; row < source.row_count; ++row) {
        observe(&evaluation.active_feature_blocks_per_row, buffers.active_feature_blocks_per_execution_row[row]);
    }
    for (u32 group = 0u; group < plan.row_group_count; ++group) {
        observe(&evaluation.active_feature_blocks_per_row_group, buffers.row_groups[group].active_feature_blocks);
    }

    *out = evaluation;
    return validation_ok();
}

validation_result estimate_packing_cost(
    const packing_occupancy_result &occupancy,
    const packing_cost_model &model,
    packing_cost_estimate *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "PackingPlan cost output is null");
    }
    if (!std::isfinite(model.byte_weight) || model.byte_weight < 0.0
        || !std::isfinite(model.occupied_tile_weight) || model.occupied_tile_weight < 0.0
        || !std::isfinite(model.row_active_block_weight) || model.row_active_block_weight < 0.0) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "PackingPlan cost weights must be finite and nonnegative");
    }

    packing_cost_estimate estimate;
    estimate.value_slots = model.dense_values_within_occupied_tiles
        ? occupancy.totals.occupied_dense_slots
        : occupancy.totals.total_nnz;
    if (!checked_mul_u64(estimate.value_slots, model.value_bytes, &estimate.value_bytes)
        || !checked_mul_u64(occupancy.totals.total_nnz, model.per_nnz_index_bytes, &estimate.per_nnz_index_bytes)
        || !checked_mul_u64(occupancy.totals.occupied_tile_count, model.occupied_tile_metadata_bytes, &estimate.occupied_tile_metadata_bytes)
        || !checked_mul_u64(occupancy.totals.row_active_block_references, model.row_active_block_metadata_bytes, &estimate.row_active_block_metadata_bytes)
        || !checked_mul_u64(occupancy.row_group_count, model.row_group_metadata_bytes, &estimate.row_group_metadata_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id, "PackingPlan cost component overflows uint64");
    }
    u64 total = 0u;
    if (!checked_add_u64(total, estimate.value_bytes, &total)
        || !checked_add_u64(total, estimate.per_nnz_index_bytes, &total)
        || !checked_add_u64(total, estimate.occupied_tile_metadata_bytes, &total)
        || !checked_add_u64(total, estimate.row_active_block_metadata_bytes, &total)
        || !checked_add_u64(total, estimate.row_group_metadata_bytes, &total)) {
        return validation_error(validation_code::integer_overflow, invalid_id, "PackingPlan total cost overflows uint64");
    }
    estimate.total_bytes = total;
    estimate.score = model.byte_weight * static_cast<double>(estimate.total_bytes)
        + model.occupied_tile_weight * static_cast<double>(occupancy.totals.occupied_tile_count)
        + model.row_active_block_weight * static_cast<double>(occupancy.totals.row_active_block_references);
    *out = estimate;
    return validation_ok();
}

count_distribution merge_count_distributions(const count_distribution &lhs, const count_distribution &rhs) {
    if (lhs.sample_count == 0u) return rhs;
    if (rhs.sample_count == 0u) return lhs;
    count_distribution merged;
    merged.sample_count = lhs.sample_count + rhs.sample_count;
    merged.minimum = std::min(lhs.minimum, rhs.minimum);
    merged.maximum = std::max(lhs.maximum, rhs.maximum);
    merged.total = lhs.total + rhs.total;
    merged.squared_total = lhs.squared_total + rhs.squared_total;
    return merged;
}

real_distribution merge_real_distributions(const real_distribution &lhs, const real_distribution &rhs) {
    if (lhs.sample_count == 0u) return rhs;
    if (rhs.sample_count == 0u) return lhs;
    real_distribution merged;
    merged.sample_count = lhs.sample_count + rhs.sample_count;
    merged.minimum = std::min(lhs.minimum, rhs.minimum);
    merged.maximum = std::max(lhs.maximum, rhs.maximum);
    merged.total = lhs.total + rhs.total;
    merged.squared_total = lhs.squared_total + rhs.squared_total;
    return merged;
}

} // namespace cellpack
