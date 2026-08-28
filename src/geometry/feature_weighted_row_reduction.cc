#include "Cellerator/geometry/feature_weighted_row_reduction.hh"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cellpack {
namespace {

using storage_t = cellerator::real::storage_t;
using compute_t = cellerator::real::compute_t;
using accum_t = cellerator::real::accum_t;

u64 splitmix64(u64 value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

u64 compute_reduction_identity(
    const frozen_packing_plan &plan,
    const warp_tile_view &tiles,
    u64 feature_weight_identity) noexcept {
    u64 identity = splitmix64(tiles.tile_identity);
    identity = splitmix64(identity ^ plan.feature_block_geometry_identity());
    identity = splitmix64(identity ^ feature_weight_identity);
    identity = splitmix64(identity
        ^ (static_cast<u64>(feature_weighted_row_reduction_schema_version) << 32u)
        ^ static_cast<u32>(cellerator::real::code_of<storage_t>::code));
    identity = splitmix64(identity
        ^ (static_cast<u64>(cellerator::real::code_of<compute_t>::code) << 32u)
        ^ static_cast<u32>(cellerator::real::code_of<accum_t>::code));
    return identity == 0u ? 1u : identity;
}

compute_t load_storage_value(const void *values, u32 index) noexcept {
    storage_t stored{};
    const auto *bytes = static_cast<const unsigned char *>(values);
    std::memcpy(&stored, bytes + static_cast<std::size_t>(index) * sizeof(storage_t),
        sizeof(storage_t));
    return static_cast<compute_t>(stored);
}

validation_result validate_plan_view(
    const frozen_packing_plan &plan,
    const feature_weighted_row_reduction_plan_view &view) {
    validation_result status = plan.validate();
    if (!status) return status;
    if (view.semantic_plan_schema_version != plan.semantic_schema_version()
        || view.geometry_identity_version != feature_block_geometry_identity_version) {
        return validation_error(validation_code::unsupported_version,
            view.semantic_plan_schema_version,
            "weighted-row-reduction plan-view version is unsupported");
    }
    if (view.feature_count != plan.feature_count()
        || view.feature_block_count != plan.feature_block_count()
        || view.feature_block_geometry_identity != plan.feature_block_geometry_identity()
        || view.feature_block_offsets != plan.feature_block_offsets()
        || view.feature_permutation != plan.feature_permutation()) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "weighted-row-reduction plan view disagrees with the frozen plan");
    }
    return validation_ok();
}

validation_result validate_contract(
    const frozen_packing_plan &plan,
    const feature_weighted_row_reduction_view &input) {
    validation_result status = validate_plan_view(plan, input.plan);
    if (!status) return status;
    if (input.schema_version != feature_weighted_row_reduction_schema_version
        || input.tiles.tile_schema_version != warp_tile_schema_version) {
        return validation_error(validation_code::unsupported_version, input.schema_version,
            "weighted-row-reduction contract version is unsupported");
    }
    if (input.storage_type_code
            != static_cast<u32>(cellerator::real::code_of<storage_t>::code)
        || input.weight_type_code
            != static_cast<u32>(cellerator::real::code_of<compute_t>::code)
        || input.accumulation_type_code
            != static_cast<u32>(cellerator::real::code_of<accum_t>::code)
        || input.tiles.value_size_bytes != sizeof(storage_t)) {
        return validation_error(validation_code::invalid_matrix_view,
            input.tiles.value_size_bytes,
            "weighted-row-reduction numeric contract is incompatible with configured precision");
    }
    if (input.feature_weight_identity == 0u
        || input.tiles.tile_identity == 0u
        || input.tiles.ordering_identity == 0u
        || input.tiles.feature_axis_fingerprint == 0u
        || input.tiles.feature_axis_fingerprint_version == 0u
        || input.tiles.row_domain_identity == 0u
        || input.reduction_identity != compute_reduction_identity(
            plan, input.tiles, input.feature_weight_identity)) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "weighted-row-reduction identity is invalid");
    }
    if (input.tiles.semantic_plan_schema_version != plan.semantic_schema_version()
        || input.tiles.geometry_identity_version != feature_block_geometry_identity_version
        || input.tiles.feature_block_geometry_identity
            != plan.feature_block_geometry_identity()
        || input.tiles.feature_count != plan.feature_count()
        || input.tiles.feature_block_count != plan.feature_block_count()) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "weighted-row-reduction tile domain disagrees with the plan view");
    }
    if (input.feature_weight_capacity < input.tiles.feature_count) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "weighted-row-reduction feature-weight capacity is insufficient");
    }
    if (input.tiles.feature_count != 0u && input.feature_weights == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "weighted-row-reduction feature weights are null");
    }
    return validation_ok();
}

validation_result validate_record_domain(
    const cell_block_record_view &records,
    const feature_weighted_row_reduction_view &input) {
    const warp_tile_view &tiles = input.tiles;
    if (tiles.record_schema_version != records.record_schema_version
        || tiles.feature_block_geometry_identity != records.feature_block_geometry_identity
        || tiles.global_row_begin != records.global_row_begin
        || tiles.full_row_count != records.full_row_count
        || tiles.row_count != records.row_count
        || tiles.feature_count != records.feature_count
        || tiles.feature_block_count != records.feature_block_count
        || tiles.nnz_count != records.nnz_count
        || tiles.value_size_bytes != records.value_size_bytes
        || tiles.feature_axis_fingerprint != records.feature_axis_fingerprint
        || tiles.feature_axis_fingerprint_version != records.feature_axis_fingerprint_version
        || tiles.row_domain_identity != records.row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "weighted-row-reduction contract disagrees with the record domain");
    }
    return validation_ok();
}

validation_result validate_canonical_domain(
    const plan_application_context &context,
    const plan_application_source_view &source,
    const feature_weighted_row_reduction_view &input) {
    const warp_tile_view &tiles = input.tiles;
    if (tiles.global_row_begin != source.global_row_begin
        || tiles.full_row_count != context.full_row_count
        || tiles.row_count != source.row_count
        || tiles.feature_count != source.feature_count
        || tiles.feature_count != context.feature_count
        || tiles.nnz_count != source.nnz_count
        || tiles.value_size_bytes != source.value_size_bytes
        || tiles.feature_axis_fingerprint != context.feature_axis_fingerprint
        || tiles.feature_axis_fingerprint_version
            != context.feature_axis_fingerprint_version
        || tiles.row_domain_identity != context.row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "weighted-row-reduction contract disagrees with the canonical CSR domain");
    }
    return validation_ok();
}

validation_result validate_output(
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "weighted-row-reduction result output is null");
    }
    if (buffers.row_capacity < input.tiles.row_count) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "weighted-row-reduction output capacity is insufficient");
    }
    if (input.tiles.row_count != 0u && buffers.row_values == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "weighted-row-reduction output values are null");
    }
    if (buffers.row_values != nullptr
        && (static_cast<const void *>(buffers.row_values)
                == static_cast<const void *>(input.feature_weights)
            || static_cast<const void *>(buffers.row_values) == input.tiles.values)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "weighted-row-reduction output must not alias weights or tile values");
    }
    return validation_ok();
}

void initialize_output(
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers) {
    if (input.tiles.row_count != 0u) {
        std::fill(buffers.row_values, buffers.row_values + input.tiles.row_count, accum_t{});
    }
}

void set_result(
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out) {
    feature_weighted_row_reduction_result_view result;
    result.schema_version = input.schema_version;
    result.reduction_identity = input.reduction_identity;
    result.feature_weight_identity = input.feature_weight_identity;
    result.global_row_begin = input.tiles.global_row_begin;
    result.full_row_count = input.tiles.full_row_count;
    result.row_count = input.tiles.row_count;
    result.row_domain_identity = input.tiles.row_domain_identity;
    result.row_values = buffers.row_values;
    *out = result;
}

void accumulate(
    accum_t *output,
    u32 row,
    u32 canonical_feature,
    compute_t value,
    const compute_t *weights) noexcept {
    const compute_t product = value * weights[canonical_feature];
    output[row] += static_cast<accum_t>(product);
}

} // namespace

bool feature_weighted_row_reduction_within_tolerance(
    accum_t reference,
    accum_t candidate,
    double absolute_tolerance,
    double relative_tolerance) noexcept {
    if (absolute_tolerance < 0.0 || relative_tolerance < 0.0
        || std::isnan(static_cast<double>(reference))
        || std::isnan(static_cast<double>(candidate))) {
        return false;
    }
    if (reference == candidate) return true;
    const double reference_value = static_cast<double>(reference);
    const double candidate_value = static_cast<double>(candidate);
    if (!std::isfinite(reference_value) || !std::isfinite(candidate_value)) return false;
    return std::fabs(candidate_value - reference_value)
        <= absolute_tolerance + relative_tolerance * std::fabs(reference_value);
}

feature_weighted_row_reduction_plan_view
make_feature_weighted_row_reduction_plan_view(
    const frozen_packing_plan &plan) noexcept {
    feature_weighted_row_reduction_plan_view result;
    result.semantic_plan_schema_version = plan.semantic_schema_version();
    result.geometry_identity_version = feature_block_geometry_identity_version;
    result.feature_count = plan.feature_count();
    result.feature_block_count = plan.feature_block_count();
    result.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    result.feature_block_offsets = plan.feature_block_offsets();
    result.feature_permutation = plan.feature_permutation();
    return result;
}

feature_weighted_row_reduction_view make_feature_weighted_row_reduction_view(
    const frozen_packing_plan &plan,
    const warp_tile_view &tiles,
    u64 feature_weight_identity,
    std::size_t feature_weight_capacity,
    const compute_t *feature_weights) noexcept {
    feature_weighted_row_reduction_view result;
    result.schema_version = feature_weighted_row_reduction_schema_version;
    result.storage_type_code =
        static_cast<u32>(cellerator::real::code_of<storage_t>::code);
    result.weight_type_code =
        static_cast<u32>(cellerator::real::code_of<compute_t>::code);
    result.accumulation_type_code =
        static_cast<u32>(cellerator::real::code_of<accum_t>::code);
    result.feature_weight_identity = feature_weight_identity;
    result.reduction_identity = compute_reduction_identity(
        plan, tiles, feature_weight_identity);
    result.plan = make_feature_weighted_row_reduction_plan_view(plan);
    result.tiles = tiles;
    result.feature_weight_capacity = feature_weight_capacity;
    result.feature_weights = feature_weights;
    return result;
}

validation_result validate_feature_weighted_row_reduction_view_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const feature_weighted_row_reduction_view &input) {
    validation_result status = validate_contract(plan, input);
    if (!status) return status;
    status = validate_warp_tile_view_host(plan, records, order, input.tiles);
    if (!status) return status;
    return validate_record_domain(records, input);
}

validation_result evaluate_feature_weighted_row_reduction_canonical_host(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source,
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out) {
    validation_result status = validate_plan_application_source_host(plan, context, source);
    if (!status) return status;
    status = validate_contract(plan, input);
    if (!status) return status;
    status = validate_canonical_domain(context, source, input);
    if (!status) return status;
    status = validate_output(input, buffers, out);
    if (!status) return status;

    initialize_output(input, buffers);
    for (u32 row = 0u; row < source.row_count; ++row) {
        for (u32 entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u];
             ++entry) {
            accumulate(buffers.row_values, row, source.canonical_feature_ids[entry],
                load_storage_value(source.values, entry), input.feature_weights);
        }
    }
    set_result(input, buffers, out);
    return validation_ok();
}

validation_result evaluate_feature_weighted_row_reduction_records_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out) {
    validation_result status = validate_cell_block_record_view_host(plan, records);
    if (!status) return status;
    status = validate_contract(plan, input);
    if (!status) return status;
    status = validate_record_domain(records, input);
    if (!status) return status;
    status = validate_output(input, buffers, out);
    if (!status) return status;

    initialize_output(input, buffers);
    for (u32 row = 0u; row < records.row_count; ++row) {
        for (u32 record = records.row_record_offsets[row];
             record < records.row_record_offsets[row + 1u]; ++record) {
            const u32 block = records.record_block_ids[record];
            const u32 block_begin = plan.feature_block_offsets()[block];
            const u32 block_width = plan.feature_block_offsets()[block + 1u] - block_begin;
            const u32 gene_mask = records.record_gene_masks[record];
            u32 value = records.record_value_offsets[record];
            for (u32 local = 0u; local < block_width; ++local) {
                if ((gene_mask & (1u << local)) == 0u) continue;
                const u32 canonical_feature = plan.feature_permutation()[block_begin + local];
                accumulate(buffers.row_values, row, canonical_feature,
                    load_storage_value(records.values, value), input.feature_weights);
                ++value;
            }
        }
    }
    set_result(input, buffers, out);
    return validation_ok();
}

validation_result evaluate_feature_weighted_row_reduction_tiles_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out) {
    validation_result status = validate_feature_weighted_row_reduction_view_host(
        plan, records, order, input);
    if (!status) return status;
    status = validate_output(input, buffers, out);
    if (!status) return status;

    initialize_output(input, buffers);
    const warp_tile_view &tiles = input.tiles;
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 tile_begin = tile * tiles.tile_row_width;
        for (u32 descriptor = tiles.tile_block_offsets[tile];
             descriptor < tiles.tile_block_offsets[tile + 1u]; ++descriptor) {
            const u32 block = tiles.tile_block_ids[descriptor];
            const u32 block_begin = input.plan.feature_block_offsets[block];
            const u32 block_width = input.plan.feature_block_offsets[block + 1u]
                - block_begin;
            const u32 cell_mask = tiles.tile_block_cell_masks[descriptor];
            u32 entry = tiles.block_row_entry_offsets[descriptor];
            for (u32 lane = 0u; lane < tiles.tile_row_width; ++lane) {
                if ((cell_mask & (1u << lane)) == 0u) continue;
                const u32 row = order.row_permutation[tile_begin + lane];
                const u32 gene_mask = tiles.row_block_gene_masks[entry];
                u32 value = tiles.row_block_value_offsets[entry];
                for (u32 local = 0u; local < block_width; ++local) {
                    if ((gene_mask & (1u << local)) == 0u) continue;
                    const u32 canonical_feature =
                        input.plan.feature_permutation[block_begin + local];
                    accumulate(buffers.row_values, row, canonical_feature,
                        load_storage_value(tiles.values, value), input.feature_weights);
                    ++value;
                }
                ++entry;
            }
        }
    }
    set_result(input, buffers, out);
    return validation_ok();
}

} // namespace cellpack
