#include "CellPack/apply_plan.hh"

#include <algorithm>
#include <cstring>
#include <limits>

namespace cellpack {
namespace {

bool multiply_overflows(std::size_t lhs, std::size_t rhs, std::size_t *out) {
    if (out == nullptr) return true;
    if (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs) return true;
    *out = lhs * rhs;
    return false;
}

validation_result validate_context_and_partition(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source) {
    validation_result status = plan.validate();
    if (!status) return status;
    if (context.full_row_count == 0u || context.feature_count == 0u
        || context.feature_axis_fingerprint == 0u
        || context.feature_axis_fingerprint_version == 0u
        || context.row_domain_identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "plan application context must identify a nonempty full row/feature domain");
    }
    packing_plan_compatibility expected;
    expected.row_count = context.full_row_count;
    expected.feature_count = context.feature_count;
    expected.feature_axis_fingerprint = context.feature_axis_fingerprint;
    expected.feature_axis_fingerprint_version = context.feature_axis_fingerprint_version;
    expected.row_domain_kind = packing_row_domain_kind::full_dataset_identity;
    expected.row_domain_identity = context.row_domain_identity;
    status = plan.validate_compatibility(expected);
    if (!status) return status;
    if (source.feature_count != context.feature_count || source.value_size_bytes == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "plan application source feature/value shape is incompatible");
    }
    if (source.row_count == 0u && source.nnz_count != 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "empty plan application row axis cannot contain nonzeros");
    }
    const u64 row_end = source.global_row_begin + static_cast<u64>(source.row_count);
    if (row_end < source.global_row_begin || row_end > context.full_row_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "plan application partition is outside the full row domain");
    }
    if (source.row_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "plan application row offsets are null");
    }
    if (source.nnz_count != 0u
        && (source.canonical_feature_ids == nullptr || source.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "plan application feature ids or values are null");
    }
    std::size_t value_bytes = 0u;
    if (multiply_overflows(source.nnz_count, source.value_size_bytes, &value_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "plan application value byte count overflows size_t");
    }
    return validation_ok();
}

validation_result validate_buffers(
    const plan_application_source_view &source,
    const plan_application_buffers &buffers) {
    const std::size_t row_offsets = static_cast<std::size_t>(source.row_count) + 1u;
    std::size_t value_bytes = 0u;
    if (multiply_overflows(source.nnz_count, source.value_size_bytes, &value_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "plan application value byte count overflows size_t");
    }
    if (buffers.row_offset_capacity < row_offsets
        || buffers.entry_capacity < source.nnz_count
        || buffers.value_capacity_bytes < value_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "plan application output capacity is insufficient");
    }
    if (buffers.row_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "plan application output row offsets are null");
    }
    if (source.nnz_count != 0u
        && (buffers.block_ids == nullptr || buffers.local_feature_ids == nullptr
            || buffers.canonical_feature_ids == nullptr || buffers.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "plan application output arrays are null");
    }
    if (source.nnz_count != 0u
        && (source.canonical_feature_ids == buffers.canonical_feature_ids
            || source.values == buffers.values
            || buffers.block_ids == buffers.local_feature_ids
            || buffers.block_ids == buffers.canonical_feature_ids
            || buffers.local_feature_ids == buffers.canonical_feature_ids)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "plan application is out-of-place and requires distinct entry buffers");
    }
    return validation_ok();
}

void set_result_view(
    const plan_application_context &context,
    const plan_application_source_view &source,
    const plan_application_buffers &buffers,
    ordered_plan_partition_view *out) {
    ordered_plan_partition_view result;
    result.semantic_plan_schema_version = packing_plan_semantic_schema_version;
    result.global_row_begin = source.global_row_begin;
    result.full_row_count = context.full_row_count;
    result.row_count = source.row_count;
    result.feature_count = source.feature_count;
    result.nnz_count = source.nnz_count;
    result.value_size_bytes = source.value_size_bytes;
    result.feature_axis_fingerprint = context.feature_axis_fingerprint;
    result.feature_axis_fingerprint_version = context.feature_axis_fingerprint_version;
    result.row_domain_identity = context.row_domain_identity;
    result.row_offsets = buffers.row_offsets;
    result.block_ids = buffers.block_ids;
    result.local_feature_ids = buffers.local_feature_ids;
    result.canonical_feature_ids = buffers.canonical_feature_ids;
    result.values = buffers.values;
    *out = result;
}

} // namespace

validation_result validate_plan_application_metadata(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source) {
    return validate_context_and_partition(plan, context, source);
}

validation_result validate_plan_application_source_host(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source) {
    validation_result status = validate_context_and_partition(plan, context, source);
    if (!status) return status;
    if (source.row_offsets[0] != 0u) {
        return validation_error(validation_code::invalid_matrix_view, 0u,
            "plan application CSR row offsets must start at zero");
    }
    for (u32 row = 0u; row < source.row_count; ++row) {
        const u32 begin = source.row_offsets[row], end = source.row_offsets[row + 1u];
        if (end < begin || end > source.nnz_count) {
            return validation_error(validation_code::invalid_matrix_view, row,
                "plan application CSR row offsets are not monotonic");
        }
        u32 previous = invalid_id;
        for (u32 entry = begin; entry < end; ++entry) {
            const u32 feature = source.canonical_feature_ids[entry];
            if (feature >= source.feature_count) {
                return validation_error(validation_code::invalid_matrix_view, entry,
                    "plan application canonical feature id is outside the feature axis");
            }
            if (previous != invalid_id && feature <= previous) {
                return validation_error(validation_code::invalid_matrix_view, entry,
                    "plan application canonical feature ids must be strictly increasing per row");
            }
            previous = feature;
        }
    }
    if (source.row_offsets[source.row_count] != source.nnz_count) {
        return validation_error(validation_code::invalid_matrix_view, source.row_count,
            "plan application final row offset does not match nnz count");
    }
    return validation_ok();
}

validation_result apply_frozen_plan_host(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source,
    const plan_application_host_workspace_view &workspace,
    const plan_application_buffers &buffers,
    ordered_plan_partition_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "ordered plan partition output is null");
    }
    validation_result status = validate_plan_application_source_host(plan, context, source);
    if (!status) return status;
    status = validate_buffers(source, buffers);
    if (!status) return status;
    if (workspace.entry_capacity < source.nnz_count
        || (source.nnz_count != 0u && (workspace.keys == nullptr || workspace.source_order == nullptr))) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "plan application host workspace is insufficient");
    }

    std::memcpy(buffers.row_offsets, source.row_offsets,
        (static_cast<std::size_t>(source.row_count) + 1u) * sizeof(u32));
    for (u32 entry = 0u; entry < source.nnz_count; ++entry) {
        const u32 canonical = source.canonical_feature_ids[entry];
        const u32 block = plan.feature_to_block()[canonical];
        const u32 local = plan.feature_to_local()[canonical];
        workspace.keys[entry] = (static_cast<u64>(block) << 32u) | local;
        workspace.source_order[entry] = entry;
    }
    const auto key_less = [&workspace](u32 lhs, u32 rhs) {
        const u64 lhs_key = workspace.keys[lhs], rhs_key = workspace.keys[rhs];
        return lhs_key != rhs_key ? lhs_key < rhs_key : lhs < rhs;
    };
    auto *output_values = static_cast<unsigned char *>(buffers.values);
    const auto *source_values = static_cast<const unsigned char *>(source.values);
    for (u32 row = 0u; row < source.row_count; ++row) {
        const u32 begin = source.row_offsets[row], end = source.row_offsets[row + 1u];
        if (end - begin > 1u) {
            std::sort(workspace.source_order + begin, workspace.source_order + end, key_less);
        }
        for (u32 output_entry = begin; output_entry < end; ++output_entry) {
            const u32 source_entry = workspace.source_order[output_entry];
            const u64 key = workspace.keys[source_entry];
            buffers.block_ids[output_entry] = static_cast<u32>(key >> 32u);
            buffers.local_feature_ids[output_entry] = static_cast<u32>(key);
            buffers.canonical_feature_ids[output_entry] = source.canonical_feature_ids[source_entry];
            std::memcpy(output_values + static_cast<std::size_t>(output_entry) * source.value_size_bytes,
                source_values + static_cast<std::size_t>(source_entry) * source.value_size_bytes,
                source.value_size_bytes);
        }
    }
    set_result_view(context, source, buffers, out);
    return validation_ok();
}

} // namespace cellpack
