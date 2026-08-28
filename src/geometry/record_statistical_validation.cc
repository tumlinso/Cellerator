#include "Cellerator/geometry/record_statistical_validation.hh"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace cellpack {
namespace {

constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;
constexpr u64 plan_identity_domain = 0x435042503131504cull;
constexpr u64 held_out_identity_domain = 0x435042503131484full;

void hash_byte(u64 *hash, unsigned char value) noexcept {
    *hash ^= value;
    *hash *= fnv1a_prime;
}

void hash_u64(u64 *hash, u64 value) noexcept {
    for (u32 byte = 0u; byte < 8u; ++byte) {
        hash_byte(hash, static_cast<unsigned char>(value >> (byte * 8u)));
    }
}

u64 nonzero_hash(u64 hash) noexcept { return hash == 0u ? 1u : hash; }

bool add_overflows_u64(u64 lhs, u64 rhs, u64 *out) noexcept {
    if (lhs > std::numeric_limits<u64>::max() - rhs) return true;
    *out = lhs + rhs;
    return false;
}

bool multiply_overflows_u64(u64 lhs, u64 rhs, u64 *out) noexcept {
    if (lhs != 0u && rhs > std::numeric_limits<u64>::max() / lhs) return true;
    *out = lhs * rhs;
    return false;
}

u64 absolute_difference(u64 lhs, u64 rhs) noexcept {
    return lhs >= rhs ? lhs - rhs : rhs - lhs;
}

u64 frozen_plan_identity(const frozen_packing_plan &plan) noexcept {
    u64 hash = fnv1a_offset;
    hash_u64(&hash, plan_identity_domain);
    hash_u64(&hash, frozen_plan_validation_identity_version);
    hash_u64(&hash, plan.semantic_schema_version());
    hash_u64(&hash, plan.row_count());
    hash_u64(&hash, plan.feature_count());
    hash_u64(&hash, plan.feature_block_count());
    hash_u64(&hash, plan.row_group_count());
    hash_u64(&hash, plan.maximum_feature_block_width());
    hash_u64(&hash, plan.row_group_width());
    hash_u64(&hash, plan.feature_block_geometry_identity());
    hash_u64(&hash, plan.identity().feature_axis_fingerprint);
    hash_u64(&hash, plan.identity().feature_axis_fingerprint_version);
    hash_u64(&hash, static_cast<u64>(plan.identity().row_domain_kind));
    hash_u64(&hash, plan.identity().row_domain_identity);
    hash_u64(&hash, plan.identity().evaluation_source_identity);
    hash_u64(&hash, plan.identity().sampling_provenance_identity);
    hash_u64(&hash, static_cast<u64>(plan.objective_kind()));
    hash_u64(&hash, plan.cost_policy_identity());
    for (u32 feature = 0u; feature < plan.feature_count(); ++feature) {
        hash_u64(&hash, plan.feature_permutation()[feature]);
    }
    for (u32 block = 0u; block <= plan.feature_block_count(); ++block) {
        hash_u64(&hash, plan.feature_block_offsets()[block]);
    }
    for (u32 group = 0u; group <= plan.row_group_count(); ++group) {
        hash_u64(&hash, plan.row_group_offsets()[group]);
    }
    return nonzero_hash(hash);
}

validation_result validate_context(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const cell_block_record_view &records) {
    validation_result status = validate_cell_block_record_view_host(plan, records);
    if (!status) return status;
    if (context.feature_axis_identity == 0u
        || context.feature_axis_identity_version == 0u
        || context.row_domain_identity == 0u
        || context.plan_training_split_identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "record-validation context identities must be explicit");
    }
    if (context.feature_axis_identity != records.feature_axis_fingerprint
        || context.feature_axis_identity_version
            != records.feature_axis_fingerprint_version
        || context.row_domain_identity != records.row_domain_identity
        || context.identities.row_count != records.full_row_count) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "record-validation context disagrees with the frozen record domain");
    }
    if (context.row_partitions == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "record-validation row partitions are null");
    }
    status = validate_validation_split(context.identities, context.row_partitions,
        context.split_provenance);
    if (!status) return status;
    if (context.plan_training_split_identity
        != context.split_provenance.assignment_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "frozen plan training split identity disagrees with held-out assignment");
    }
    return validation_ok();
}

validation_result validate_source(
    const record_validation_source_view &source,
    const cell_block_record_view &records) {
    if (source.dataset_identity == 0u || source.value_size_bytes == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "record-validation source identity and value width must be nonzero");
    }
    validation_result status = validate_csr_support_view(source.support);
    if (!status) return status;
    if (source.global_row_begin != records.global_row_begin
        || source.full_row_count != records.full_row_count
        || source.support.row_count != records.row_count
        || source.support.feature_count != records.feature_count
        || source.support.nnz_count != records.nnz_count
        || source.value_size_bytes != records.value_size_bytes) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "canonical validation source and cell-block records disagree");
    }
    if (source.support.nnz_count != 0u && source.values == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "canonical validation source values are null");
    }
    return validation_ok();
}

struct decoded_entry {
    u32 canonical_feature = 0u;
    u32 record_value_index = 0u;
};

validation_result validate_selected_row_exactly(
    const frozen_packing_plan &plan,
    const record_validation_source_view &source,
    const cell_block_record_view &records,
    u32 row,
    std::vector<decoded_entry> *decoded) {
    decoded->clear();
    const u32 record_begin = records.row_record_offsets[row];
    const u32 record_end = records.row_record_offsets[row + 1u];
    for (u32 record = record_begin; record < record_end; ++record) {
        const u32 block = records.record_block_ids[record];
        const u32 execution_begin = plan.feature_block_offsets()[block];
        const u32 width = plan.feature_block_offsets()[block + 1u] - execution_begin;
        const u32 mask = records.record_gene_masks[record];
        u32 value_index = records.record_value_offsets[record];
        for (u32 local = 0u; local < width; ++local) {
            if ((mask & (1u << local)) != 0u) {
                decoded->push_back({plan.feature_permutation()[execution_begin + local],
                    value_index++});
            }
        }
        if (value_index != records.record_value_offsets[record + 1u]) {
            return validation_error(validation_code::invalid_offsets, record,
                "record-validation mask rank changed during decode");
        }
    }
    std::sort(decoded->begin(), decoded->end(),
        [](const decoded_entry &lhs, const decoded_entry &rhs) {
            return lhs.canonical_feature < rhs.canonical_feature;
        });
    const u32 source_begin = source.support.row_offsets[row];
    const u32 source_end = source.support.row_offsets[row + 1u];
    if (decoded->size() != static_cast<std::size_t>(source_end - source_begin)) {
        return validation_error(validation_code::invalid_matrix_view, row,
            "held-out record row does not reconstruct the canonical row degree");
    }
    const auto *source_bytes = static_cast<const unsigned char *>(source.values);
    const auto *record_bytes = static_cast<const unsigned char *>(records.values);
    for (u32 index = 0u; index < decoded->size(); ++index) {
        const u32 source_entry = source_begin + index;
        if ((*decoded)[index].canonical_feature != source.support.feature_ids[source_entry]) {
            return validation_error(validation_code::invalid_matrix_view, row,
                "held-out record row does not reconstruct canonical feature identity");
        }
        const std::size_t source_offset = static_cast<std::size_t>(source_entry)
            * source.value_size_bytes;
        const std::size_t record_offset =
            static_cast<std::size_t>((*decoded)[index].record_value_index)
            * source.value_size_bytes;
        if (std::memcmp(source_bytes + source_offset, record_bytes + record_offset,
                source.value_size_bytes) != 0) {
            return validation_error(validation_code::invalid_matrix_view, row,
                "held-out record row changed canonical value bytes");
        }
    }
    return validation_ok();
}

validation_result build_projection_bytes(
    u64 row_count,
    u64 nnz_count,
    u64 record_count,
    u32 value_size_bytes,
    u64 *metadata_bytes,
    u64 *encoded_bytes,
    u64 *baseline_bytes) noexcept {
    u64 row_identity_bytes = 0u, row_offset_bytes = 0u;
    u64 record_descriptor_bytes = 0u, record_value_offset_bytes = 0u;
    u64 value_bytes = 0u, baseline_feature_bytes = 0u, metadata = 0u;
    if (multiply_overflows_u64(row_count, sizeof(u64), &row_identity_bytes)
        || multiply_overflows_u64(row_count + 1u, sizeof(u32), &row_offset_bytes)
        || multiply_overflows_u64(record_count, 2u * sizeof(u32),
            &record_descriptor_bytes)
        || multiply_overflows_u64(record_count + 1u, sizeof(u32),
            &record_value_offset_bytes)
        || multiply_overflows_u64(nnz_count, value_size_bytes, &value_bytes)
        || multiply_overflows_u64(nnz_count, sizeof(u32), &baseline_feature_bytes)
        || add_overflows_u64(row_identity_bytes, row_offset_bytes, &metadata)
        || add_overflows_u64(metadata, record_descriptor_bytes, &metadata)
        || add_overflows_u64(metadata, record_value_offset_bytes, &metadata)
        || add_overflows_u64(metadata, value_bytes, encoded_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "held-out record projection byte count overflows u64");
    }
    u64 baseline = 0u;
    if (add_overflows_u64(row_identity_bytes, row_offset_bytes, &baseline)
        || add_overflows_u64(baseline, baseline_feature_bytes, &baseline)
        || add_overflows_u64(baseline, value_bytes, &baseline)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "held-out CSR baseline byte count overflows u64");
    }
    *metadata_bytes = metadata;
    *baseline_bytes = baseline;
    return validation_ok();
}

} // namespace

validation_result evaluate_held_out_cell_block_records(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &source,
    const cell_block_record_view &records,
    held_out_record_validation *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "held-out record validation output is null");
    }
    validation_result status = validate_context(plan, context, records);
    if (!status) return status;
    status = validate_source(source, records);
    if (!status) return status;

    try {
        u64 held_out_rows = 0u, held_out_nnz = 0u, held_out_records = 0u;
        u64 row_identity = fnv1a_offset;
        hash_u64(&row_identity, held_out_identity_domain);
        hash_u64(&row_identity, context.split_provenance.assignment_identity);
        hash_u64(&row_identity, records.global_row_begin);
        std::vector<decoded_entry> decoded;
        for (u32 row = 0u; row < records.row_count; ++row) {
            const u64 global_row = records.global_row_begin + row;
            if (context.row_partitions[global_row] != validation_partition::held_out) {
                continue;
            }
            status = validate_selected_row_exactly(plan, source, records, row, &decoded);
            if (!status) return status;
            const u32 row_nnz = source.support.row_offsets[row + 1u]
                - source.support.row_offsets[row];
            const u32 row_records = records.row_record_offsets[row + 1u]
                - records.row_record_offsets[row];
            ++held_out_rows;
            held_out_nnz += row_nnz;
            held_out_records += row_records;
            hash_u64(&row_identity, global_row);
            hash_u64(&row_identity, context.identities.row_identities[global_row]);
        }
        if (held_out_rows == 0u) {
            return validation_error(validation_code::invalid_matrix_view, invalid_id,
                "record partition contains no held-out rows");
        }

        packing_validation_metrics metrics;
        metrics.available = packing_validation_metric_records
            | packing_validation_metric_correctness;
        metrics.dataset_identity = source.dataset_identity;
        metrics.feature_axis_identity = context.feature_axis_identity;
        metrics.row_domain_identity = context.row_domain_identity;
        metrics.split_identity = context.split_provenance.assignment_identity;
        metrics.row_count = held_out_rows;
        metrics.feature_count = source.support.feature_count;
        metrics.nnz_count = held_out_nnz;
        metrics.active_block_references = held_out_records;
        metrics.correctness_items = held_out_rows + held_out_nnz;
        status = build_projection_bytes(held_out_rows, held_out_nnz,
            held_out_records, source.value_size_bytes, &metrics.metadata_bytes,
            &metrics.encoded_bytes, &metrics.baseline_bytes);
        if (!status) return status;
        if (held_out_nnz != 0u) {
            metrics.available |= packing_validation_metric_storage;
        }
        status = validate_packing_validation_metrics(metrics);
        if (!status) return status;

        held_out_record_validation result;
        result.frozen_plan_identity = frozen_plan_identity(plan);
        result.feature_block_geometry_identity = plan.feature_block_geometry_identity();
        result.held_out_row_identity = nonzero_hash(row_identity);
        result.plan_training_split_identity = context.plan_training_split_identity;
        result.unit_kind = context.split_provenance.unit_kind;
        result.claims_group_generalization =
            context.split_provenance.claims_group_generalization;
        result.metrics = metrics;
        *out = result;
        return validation_ok();
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "held-out record validation allocation failed");
    }
}

validation_result compare_held_out_cell_block_records_to_degree_null(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &real_source,
    const cell_block_record_view &real_records,
    const record_validation_source_view &null_source,
    const cell_block_record_view &null_records,
    const degree_preserving_null_provenance &null_provenance,
    held_out_record_null_comparison *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "held-out real/null comparison output is null");
    }
    if (real_source.dataset_identity != null_provenance.source_identity
        || null_source.dataset_identity != null_provenance.output_identity
        || real_source.value_size_bytes != null_source.value_size_bytes) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "real/null dataset identities or value widths disagree with null provenance");
    }
    validation_result status = validate_degree_preserving_null_provenance(
        real_source.support, null_source.support, null_provenance);
    if (!status) return status;

    held_out_record_null_comparison result;
    status = evaluate_held_out_cell_block_records(
        plan, context, real_source, real_records, &result.real);
    if (!status) return status;
    status = evaluate_held_out_cell_block_records(
        plan, context, null_source, null_records, &result.degree_preserving_null);
    if (!status) return status;
    if (result.real.held_out_row_identity
            != result.degree_preserving_null.held_out_row_identity
        || result.real.frozen_plan_identity
            != result.degree_preserving_null.frozen_plan_identity
        || result.real.metrics.row_count
            != result.degree_preserving_null.metrics.row_count
        || result.real.metrics.nnz_count
            != result.degree_preserving_null.metrics.nnz_count) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "real/null held-out evaluations do not share one frozen split and plan");
    }
    result.encoded_bytes_absolute_difference = absolute_difference(
        result.real.metrics.encoded_bytes,
        result.degree_preserving_null.metrics.encoded_bytes);
    result.metadata_bytes_absolute_difference = absolute_difference(
        result.real.metrics.metadata_bytes,
        result.degree_preserving_null.metrics.metadata_bytes);
    result.active_block_references_absolute_difference = absolute_difference(
        result.real.metrics.active_block_references,
        result.degree_preserving_null.metrics.active_block_references);
    result.real_encoded_bytes_no_greater = result.real.metrics.encoded_bytes
        <= result.degree_preserving_null.metrics.encoded_bytes;
    result.real_metadata_bytes_no_greater = result.real.metrics.metadata_bytes
        <= result.degree_preserving_null.metrics.metadata_bytes;
    result.real_active_blocks_no_greater = result.real.metrics.active_block_references
        <= result.degree_preserving_null.metrics.active_block_references;
    result.exact_degree_conservation = null_provenance.row_degrees_exact
        && null_provenance.feature_degrees_exact;
    *out = result;
    return validation_ok();
}

} // namespace cellpack
