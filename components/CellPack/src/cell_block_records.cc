#include "CellPack/cell_block_records.hh"

#include <cstring>
#include <limits>

namespace cellpack {
namespace {

bool multiply_overflows(std::size_t lhs, std::size_t rhs, std::size_t *out) noexcept {
    if (out == nullptr || (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return true;
    }
    *out = lhs * rhs;
    return false;
}

u32 popcount_u32(u32 value) noexcept {
    u32 count = 0u;
    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

u32 valid_mask_for_width(u32 width) noexcept {
    return width == cell_block_gene_mask_bits
        ? std::numeric_limits<u32>::max()
        : ((1u << width) - 1u);
}

validation_result validate_plan_for_cell_blocks(const frozen_packing_plan &plan) {
    const validation_result status = plan.validate();
    if (!status) return status;
    if (plan.identity().row_domain_kind != packing_row_domain_kind::full_dataset_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "cell-block records require a full-domain frozen plan");
    }
    if (plan.maximum_feature_block_width() > cell_block_gene_mask_bits) {
        return validation_error(validation_code::invalid_plan_geometry,
            plan.maximum_feature_block_width(),
            "cell-block record v1 supports at most 32 features per block");
    }
    if (plan.feature_block_geometry_identity() == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "frozen plan feature-block geometry identity is missing");
    }
    return validation_ok();
}

validation_result validate_record_metadata(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records) {
    validation_result status = validate_plan_for_cell_blocks(plan);
    if (!status) return status;
    if (records.record_schema_version != cell_block_record_schema_version
        || records.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || records.geometry_identity_version != feature_block_geometry_identity_version) {
        return validation_error(validation_code::unsupported_version,
            records.record_schema_version, "cell-block record version is unsupported");
    }
    if (records.feature_block_geometry_identity != plan.feature_block_geometry_identity()) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "cell-block records do not match the frozen feature-block geometry");
    }
    if (records.full_row_count != plan.row_count()
        || records.feature_count != plan.feature_count()
        || records.feature_block_count != plan.feature_block_count()
        || records.feature_axis_fingerprint != plan.identity().feature_axis_fingerprint
        || records.feature_axis_fingerprint_version
            != plan.identity().feature_axis_fingerprint_version
        || records.row_domain_identity != plan.identity().row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "cell-block record dataset identity is incompatible with the frozen plan");
    }
    if (records.value_size_bytes == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "cell-block record value size is zero");
    }
    const u64 row_end = records.global_row_begin + static_cast<u64>(records.row_count);
    if (row_end < records.global_row_begin || row_end > records.full_row_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "cell-block record partition is outside the full row domain");
    }
    return validation_ok();
}

validation_result validate_record_buffers(
    const ordered_plan_partition_view &source,
    const cell_block_record_requirements &required,
    const cell_block_record_buffers &buffers) {
    if (buffers.row_record_offset_capacity < required.row_record_offset_count
        || buffers.record_capacity < required.record_count
        || buffers.record_value_offset_capacity < required.record_value_offset_count
        || buffers.value_capacity_bytes < required.value_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "cell-block record output capacity is insufficient");
    }
    if (buffers.row_record_offsets == nullptr || buffers.record_value_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record offset output is null");
    }
    if (required.record_count != 0u
        && (buffers.record_block_ids == nullptr || buffers.record_gene_masks == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record descriptor output is null");
    }
    if (source.nnz_count != 0u && buffers.values == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record value output is null");
    }
    if (source.nnz_count != 0u && source.values == buffers.values) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "cell-block record construction is out-of-place");
    }
    if (buffers.row_record_offsets == buffers.record_value_offsets
        || (required.record_count != 0u
            && (buffers.row_record_offsets == buffers.record_block_ids
                || buffers.row_record_offsets == buffers.record_gene_masks
                || buffers.record_value_offsets == buffers.record_block_ids
                || buffers.record_value_offsets == buffers.record_gene_masks
                || buffers.record_block_ids == buffers.record_gene_masks))) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "cell-block record output arrays must be distinct");
    }
    return validation_ok();
}

void set_record_view(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source,
    u32 record_count,
    const cell_block_record_buffers &buffers,
    cell_block_record_view *out) {
    cell_block_record_view result;
    result.record_schema_version = cell_block_record_schema_version;
    result.semantic_plan_schema_version = packing_plan_semantic_schema_version;
    result.geometry_identity_version = feature_block_geometry_identity_version;
    result.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    result.global_row_begin = source.global_row_begin;
    result.full_row_count = source.full_row_count;
    result.row_count = source.row_count;
    result.feature_count = source.feature_count;
    result.feature_block_count = plan.feature_block_count();
    result.nnz_count = source.nnz_count;
    result.record_count = record_count;
    result.value_size_bytes = source.value_size_bytes;
    result.feature_axis_fingerprint = source.feature_axis_fingerprint;
    result.feature_axis_fingerprint_version = source.feature_axis_fingerprint_version;
    result.row_domain_identity = source.row_domain_identity;
    result.row_record_offsets = buffers.row_record_offsets;
    result.record_block_ids = buffers.record_block_ids;
    result.record_gene_masks = buffers.record_gene_masks;
    result.record_value_offsets = buffers.record_value_offsets;
    result.values = buffers.values;
    *out = result;
}

} // namespace

validation_result validate_ordered_plan_partition_for_cell_blocks_host(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source) {
    validation_result status = validate_plan_for_cell_blocks(plan);
    if (!status) return status;
    if (source.semantic_plan_schema_version != packing_plan_semantic_schema_version) {
        return validation_error(validation_code::unsupported_version,
            source.semantic_plan_schema_version,
            "ordered partition semantic plan version is unsupported");
    }
    if (source.full_row_count != plan.row_count()
        || source.feature_count != plan.feature_count()
        || source.feature_axis_fingerprint != plan.identity().feature_axis_fingerprint
        || source.feature_axis_fingerprint_version != plan.identity().feature_axis_fingerprint_version
        || source.row_domain_identity != plan.identity().row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "ordered partition is incompatible with the frozen plan");
    }
    if (source.value_size_bytes == 0u || source.row_offsets == nullptr) {
        return validation_error(source.row_offsets == nullptr
                ? validation_code::null_pointer : validation_code::invalid_matrix_view,
            invalid_id, "ordered partition row offsets or value size is invalid");
    }
    const u64 row_end = source.global_row_begin + static_cast<u64>(source.row_count);
    if (row_end < source.global_row_begin || row_end > source.full_row_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "ordered partition is outside the full row domain");
    }
    if (source.nnz_count != 0u
        && (source.block_ids == nullptr || source.local_feature_ids == nullptr
            || source.canonical_feature_ids == nullptr || source.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "ordered partition entry arrays are null");
    }
    if (source.row_offsets[0] != 0u) {
        return validation_error(validation_code::invalid_offsets, 0u,
            "ordered partition row offsets must start at zero");
    }
    for (u32 row = 0u; row < source.row_count; ++row) {
        const u32 begin = source.row_offsets[row], end = source.row_offsets[row + 1u];
        if (end < begin || end > source.nnz_count) {
            return validation_error(validation_code::invalid_offsets, row,
                "ordered partition row offsets are not monotonic");
        }
        u64 previous_key = 0u;
        bool have_previous = false;
        for (u32 entry = begin; entry < end; ++entry) {
            const u32 block = source.block_ids[entry], local = source.local_feature_ids[entry];
            if (block >= plan.feature_block_count()) {
                return validation_error(validation_code::invalid_plan_geometry, entry,
                    "ordered partition block id is outside the frozen plan");
            }
            const u32 block_begin = plan.feature_block_offsets()[block];
            const u32 block_width = plan.feature_block_offsets()[block + 1u] - block_begin;
            if (local >= block_width) {
                return validation_error(validation_code::invalid_plan_geometry, entry,
                    "ordered partition local feature id is outside its block");
            }
            if (source.canonical_feature_ids[entry]
                != plan.feature_permutation()[block_begin + local]) {
                return validation_error(validation_code::invalid_plan_geometry, entry,
                    "ordered partition canonical feature disagrees with block geometry");
            }
            const u64 key = (static_cast<u64>(block) << 32u) | local;
            if (have_previous && key <= previous_key) {
                return validation_error(validation_code::invalid_offsets, entry,
                    "ordered partition block/local coordinates are not strictly increasing per row");
            }
            previous_key = key;
            have_previous = true;
        }
    }
    if (source.row_offsets[source.row_count] != source.nnz_count) {
        return validation_error(validation_code::invalid_offsets, source.row_count,
            "ordered partition final row offset does not match nnz count");
    }
    return validation_ok();
}

validation_result query_cell_block_record_requirements_host(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source,
    cell_block_record_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record requirements output is null");
    }
    const validation_result status =
        validate_ordered_plan_partition_for_cell_blocks_host(plan, source);
    if (!status) return status;
    u32 record_count = 0u;
    for (u32 row = 0u; row < source.row_count; ++row) {
        u32 previous_block = invalid_id;
        for (u32 entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u]; ++entry) {
            if (source.block_ids[entry] != previous_block) {
                ++record_count;
                previous_block = source.block_ids[entry];
            }
        }
    }
    std::size_t value_bytes = 0u;
    if (multiply_overflows(source.nnz_count, source.value_size_bytes, &value_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "cell-block record value byte count overflows size_t");
    }
    cell_block_record_requirements result;
    result.row_record_offset_count = static_cast<std::size_t>(source.row_count) + 1u;
    result.record_count = record_count;
    result.record_value_offset_count = static_cast<std::size_t>(record_count) + 1u;
    result.value_bytes = value_bytes;
    *out = result;
    return validation_ok();
}

validation_result build_cell_block_records_host(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source,
    const cell_block_record_buffers &buffers,
    cell_block_record_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record view output is null");
    }
    cell_block_record_requirements required;
    validation_result status = query_cell_block_record_requirements_host(plan, source, &required);
    if (!status) return status;
    status = validate_record_buffers(source, required, buffers);
    if (!status) return status;

    u32 record = 0u;
    for (u32 row = 0u; row < source.row_count; ++row) {
        buffers.row_record_offsets[row] = record;
        u32 current_block = invalid_id;
        for (u32 entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u]; ++entry) {
            const u32 block = source.block_ids[entry], local = source.local_feature_ids[entry];
            if (block != current_block) {
                buffers.record_block_ids[record] = block;
                buffers.record_gene_masks[record] = 0u;
                buffers.record_value_offsets[record] = entry;
                current_block = block;
                ++record;
            }
            buffers.record_gene_masks[record - 1u] |= 1u << local;
        }
    }
    buffers.row_record_offsets[source.row_count] = record;
    buffers.record_value_offsets[record] = source.nnz_count;
    if (required.value_bytes != 0u) {
        std::memcpy(buffers.values, source.values, required.value_bytes);
    }
    set_record_view(plan, source, record, buffers, out);
    return validate_cell_block_record_view_host(plan, *out);
}

validation_result validate_cell_block_record_view_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records) {
    validation_result status = validate_record_metadata(plan, records);
    if (!status) return status;
    if (records.row_record_offsets == nullptr || records.record_value_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record offsets are null");
    }
    if (records.record_count != 0u
        && (records.record_block_ids == nullptr || records.record_gene_masks == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record descriptors are null");
    }
    if (records.nnz_count != 0u && records.values == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "cell-block record values are null");
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
                "cell-block row-to-record offsets are not monotonic");
        }
        u32 previous_block = invalid_id;
        for (u32 record = begin; record < end; ++record) {
            const u32 block = records.record_block_ids[record];
            if (block >= plan.feature_block_count()
                || (previous_block != invalid_id && block <= previous_block)) {
                return validation_error(validation_code::invalid_offsets, record,
                    "cell-block record ids are not strictly increasing per row");
            }
            const u32 width = plan.feature_block_offsets()[block + 1u]
                - plan.feature_block_offsets()[block];
            const u32 mask = records.record_gene_masks[record];
            if (mask == 0u || (mask & ~valid_mask_for_width(width)) != 0u) {
                return validation_error(validation_code::invalid_plan_geometry, record,
                    "cell-block gene mask is empty or outside its block width");
            }
            const u32 value_begin = records.record_value_offsets[record];
            const u32 value_end = records.record_value_offsets[record + 1u];
            if (value_end < value_begin || value_end > records.nnz_count
                || value_end - value_begin != popcount_u32(mask)) {
                return validation_error(validation_code::invalid_offsets, record,
                    "cell-block value offsets disagree with the gene-mask rank");
            }
            previous_block = block;
        }
    }
    if (records.row_record_offsets[records.row_count] != records.record_count
        || records.record_value_offsets[records.record_count] != records.nnz_count) {
        return validation_error(validation_code::invalid_offsets, records.record_count,
            "cell-block terminal offset does not match its declared count");
    }
    std::size_t ignored = 0u;
    if (multiply_overflows(records.nnz_count, records.value_size_bytes, &ignored)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "cell-block record value byte count overflows size_t");
    }
    return validation_ok();
}

validation_result decode_cell_block_records_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const cell_block_decode_buffers &buffers,
    decoded_cell_block_partition_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "decoded cell-block partition output is null");
    }
    validation_result status = validate_cell_block_record_view_host(plan, records);
    if (!status) return status;
    std::size_t value_bytes = 0u;
    if (multiply_overflows(records.nnz_count, records.value_size_bytes, &value_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "decoded cell-block value byte count overflows size_t");
    }
    if (buffers.row_offset_capacity < static_cast<std::size_t>(records.row_count) + 1u
        || buffers.entry_capacity < records.nnz_count
        || buffers.value_capacity_bytes < value_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "decoded cell-block output capacity is insufficient");
    }
    if (buffers.row_offsets == nullptr
        || (records.nnz_count != 0u
            && (buffers.canonical_feature_ids == nullptr || buffers.values == nullptr))) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "decoded cell-block output arrays are null");
    }
    if (records.nnz_count != 0u && buffers.values == records.values) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "cell-block decoding is out-of-place");
    }

    for (u32 row = 0u; row <= records.row_count; ++row) {
        buffers.row_offsets[row] =
            records.record_value_offsets[records.row_record_offsets[row]];
    }
    for (u32 record = 0u; record < records.record_count; ++record) {
        const u32 block = records.record_block_ids[record];
        const u32 block_begin = plan.feature_block_offsets()[block];
        const u32 width = plan.feature_block_offsets()[block + 1u] - block_begin;
        const u32 mask = records.record_gene_masks[record];
        u32 output_entry = records.record_value_offsets[record];
        for (u32 local = 0u; local < width; ++local) {
            if ((mask & (1u << local)) != 0u) {
                buffers.canonical_feature_ids[output_entry++] =
                    plan.feature_permutation()[block_begin + local];
            }
        }
    }
    if (value_bytes != 0u) std::memcpy(buffers.values, records.values, value_bytes);

    decoded_cell_block_partition_view result;
    result.global_row_begin = records.global_row_begin;
    result.full_row_count = records.full_row_count;
    result.row_count = records.row_count;
    result.feature_count = records.feature_count;
    result.nnz_count = records.nnz_count;
    result.value_size_bytes = records.value_size_bytes;
    result.feature_axis_fingerprint = records.feature_axis_fingerprint;
    result.feature_axis_fingerprint_version = records.feature_axis_fingerprint_version;
    result.row_domain_identity = records.row_domain_identity;
    result.row_offsets = buffers.row_offsets;
    result.canonical_feature_ids = buffers.canonical_feature_ids;
    result.values = buffers.values;
    *out = result;
    return validation_ok();
}

} // namespace cellpack
