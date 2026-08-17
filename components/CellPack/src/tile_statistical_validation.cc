#include "CellPack/tile_statistical_validation.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

// Phase E intentionally keeps identity checks, direct reconstruction, metric
// projection, and deterministic bootstrap reduction in one translation unit.
// Splitting these coupled rules would risk accepting one provenance path while
// measuring another; the public pointer-first ABI remains compact and stable.
namespace cellpack {
namespace {

constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;
constexpr u64 plan_identity_domain = 0x435042503131504cull;
constexpr u64 held_out_identity_domain = 0x435042503131484full;
constexpr u64 bootstrap_realization_domain = 0x435031314254525aull;

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

u32 popcount_u32(u32 value) noexcept {
    u32 count = 0u;
    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

u32 lower_lane_mask(u32 lane) noexcept {
    return lane == 0u ? 0u : ((1u << lane) - 1u);
}

bool add_overflows(u64 lhs, u64 rhs, u64 *out) noexcept {
    if (lhs > std::numeric_limits<u64>::max() - rhs) return true;
    *out = lhs + rhs;
    return false;
}

bool multiply_overflows(u64 lhs, u64 rhs, u64 *out) noexcept {
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

validation_result validate_context_and_inputs(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &source,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles) {
    validation_result status = validate_warp_tile_view_host(plan, records, order, tiles);
    if (!status) return status;
    if (context.feature_axis_identity == 0u
        || context.feature_axis_identity_version == 0u
        || context.row_domain_identity == 0u
        || context.plan_training_split_identity == 0u
        || source.dataset_identity == 0u || source.value_size_bytes == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "tile-validation identities and value width must be explicit");
    }
    if (context.feature_axis_identity != records.feature_axis_fingerprint
        || context.feature_axis_identity_version
            != records.feature_axis_fingerprint_version
        || context.row_domain_identity != records.row_domain_identity
        || context.identities.row_count != records.full_row_count
        || context.plan_training_split_identity
            != context.split_provenance.assignment_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "tile-validation context disagrees with the frozen plan domain");
    }
    if (context.row_partitions == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "tile-validation row partitions are null");
    }
    status = validate_validation_split(
        context.identities, context.row_partitions, context.split_provenance);
    if (!status) return status;
    status = validate_csr_support_view(source.support);
    if (!status) return status;
    if (source.global_row_begin != records.global_row_begin
        || source.full_row_count != records.full_row_count
        || source.support.row_count != records.row_count
        || source.support.feature_count != records.feature_count
        || source.support.nnz_count != records.nnz_count
        || source.value_size_bytes != records.value_size_bytes) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "canonical tile-validation source disagrees with the tile domain");
    }
    if (source.support.nnz_count != 0u && source.values == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "canonical tile-validation source values are null");
    }
    return validation_ok();
}

bool find_source_feature(
    const record_validation_source_view &source,
    u32 row,
    u32 canonical_feature,
    u32 *entry) noexcept {
    u32 begin = source.support.row_offsets[row];
    u32 end = source.support.row_offsets[row + 1u];
    while (begin < end) {
        const u32 middle = begin + (end - begin) / 2u;
        if (source.support.feature_ids[middle] < canonical_feature) begin = middle + 1u;
        else end = middle;
    }
    if (begin == source.support.row_offsets[row + 1u]
        || source.support.feature_ids[begin] != canonical_feature) return false;
    *entry = begin;
    return true;
}

validation_result validate_row_exactly(
    const frozen_packing_plan &plan,
    const record_validation_source_view &source,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    u32 row,
    u64 *nnz,
    u64 *active_blocks) {
    const u32 execution = order.inverse_row_permutation[row];
    const u32 tile = execution / tiles.tile_row_width;
    const u32 lane = execution % tiles.tile_row_width;
    const u32 lane_bit = 1u << lane;
    const auto *source_bytes = static_cast<const unsigned char *>(source.values);
    const auto *tile_bytes = static_cast<const unsigned char *>(tiles.values);
    u32 observed_nnz = 0u;
    u32 observed_blocks = 0u;
    for (u32 descriptor = tiles.tile_block_offsets[tile];
         descriptor < tiles.tile_block_offsets[tile + 1u]; ++descriptor) {
        const u32 cell_mask = tiles.tile_block_cell_masks[descriptor];
        if ((cell_mask & lane_bit) == 0u) continue;
        ++observed_blocks;
        const u32 block = tiles.tile_block_ids[descriptor];
        const u32 block_begin = plan.feature_block_offsets()[block];
        const u32 block_width = plan.feature_block_offsets()[block + 1u] - block_begin;
        const u32 entry = tiles.block_row_entry_offsets[descriptor]
            + popcount_u32(cell_mask & lower_lane_mask(lane));
        const u32 gene_mask = tiles.row_block_gene_masks[entry];
        u32 tile_value = tiles.row_block_value_offsets[entry];
        for (u32 local = 0u; local < block_width; ++local) {
            if ((gene_mask & (1u << local)) == 0u) continue;
            const u32 canonical_feature = plan.feature_permutation()[block_begin + local];
            u32 source_entry = 0u;
            if (!find_source_feature(source, row, canonical_feature, &source_entry)) {
                return validation_error(validation_code::invalid_matrix_view, row,
                    "warp tile does not reconstruct a canonical source feature");
            }
            const std::size_t source_offset = static_cast<std::size_t>(source_entry)
                * source.value_size_bytes;
            const std::size_t tile_offset = static_cast<std::size_t>(tile_value)
                * source.value_size_bytes;
            if (std::memcmp(source_bytes + source_offset, tile_bytes + tile_offset,
                    source.value_size_bytes) != 0) {
                return validation_error(validation_code::invalid_matrix_view, row,
                    "warp tile changed canonical source value bytes");
            }
            ++tile_value;
            ++observed_nnz;
        }
        if (tile_value != tiles.row_block_value_offsets[entry + 1u]) {
            return validation_error(validation_code::invalid_offsets, entry,
                "warp-tile mask rank changed during validation");
        }
    }
    const u32 source_nnz = source.support.row_offsets[row + 1u]
        - source.support.row_offsets[row];
    if (observed_nnz != source_nnz) {
        return validation_error(validation_code::invalid_matrix_view, row,
            "warp tile does not reconstruct the canonical row degree");
    }
    *nnz = observed_nnz;
    *active_blocks = observed_blocks;
    return validation_ok();
}

validation_result build_projection_bytes(
    u64 row_count,
    u64 nnz_count,
    u64 tile_count,
    u64 tile_block_union,
    u64 active_blocks,
    u32 value_size_bytes,
    u64 *metadata_bytes,
    u64 *encoded_bytes,
    u64 *baseline_bytes) noexcept {
    u64 row_identity_bytes = 0u, tile_offset_bytes = 0u;
    u64 tile_descriptor_bytes = 0u, block_entry_offset_bytes = 0u;
    u64 row_entry_bytes = 0u, row_value_offset_bytes = 0u;
    u64 value_bytes = 0u, baseline_row_offset_bytes = 0u;
    u64 baseline_feature_bytes = 0u, metadata = 0u;
    if (multiply_overflows(row_count, sizeof(u64), &row_identity_bytes)
        || multiply_overflows(tile_count + 1u, sizeof(u32), &tile_offset_bytes)
        || multiply_overflows(tile_block_union, 2u * sizeof(u32),
            &tile_descriptor_bytes)
        || multiply_overflows(tile_block_union + 1u, sizeof(u32),
            &block_entry_offset_bytes)
        || multiply_overflows(active_blocks, sizeof(u32), &row_entry_bytes)
        || multiply_overflows(active_blocks + 1u, sizeof(u32),
            &row_value_offset_bytes)
        || multiply_overflows(nnz_count, value_size_bytes, &value_bytes)
        || multiply_overflows(row_count + 1u, sizeof(u32),
            &baseline_row_offset_bytes)
        || multiply_overflows(nnz_count, sizeof(u32), &baseline_feature_bytes)
        || add_overflows(row_identity_bytes, tile_offset_bytes, &metadata)
        || add_overflows(metadata, tile_descriptor_bytes, &metadata)
        || add_overflows(metadata, block_entry_offset_bytes, &metadata)
        || add_overflows(metadata, row_entry_bytes, &metadata)
        || add_overflows(metadata, row_value_offset_bytes, &metadata)
        || add_overflows(metadata, value_bytes, encoded_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "tile projection byte count overflows u64");
    }
    u64 baseline = 0u;
    if (add_overflows(row_identity_bytes, baseline_row_offset_bytes, &baseline)
        || add_overflows(baseline, baseline_feature_bytes, &baseline)
        || add_overflows(baseline, value_bytes, &baseline)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "tile CSR baseline byte count overflows u64");
    }
    *metadata_bytes = metadata;
    *baseline_bytes = baseline;
    return validation_ok();
}

validation_result finalize_metrics(
    u64 dataset_identity,
    u64 feature_axis_identity,
    u64 row_domain_identity,
    u64 split_identity,
    u64 row_count,
    u64 feature_count,
    u64 nnz_count,
    u64 tile_count,
    u64 tile_block_union,
    u64 active_blocks,
    u64 padding_slots,
    u32 value_size_bytes,
    packing_validation_metrics *out) {
    packing_validation_metrics result;
    result.available = packing_validation_metric_records
        | packing_validation_metric_tiles | packing_validation_metric_correctness;
    result.dataset_identity = dataset_identity;
    result.feature_axis_identity = feature_axis_identity;
    result.row_domain_identity = row_domain_identity;
    result.split_identity = split_identity;
    result.row_count = row_count;
    result.feature_count = feature_count;
    result.nnz_count = nnz_count;
    result.active_block_references = active_blocks;
    result.tile_count = tile_count;
    result.tile_block_union_references = tile_block_union;
    result.padding_slots = padding_slots;
    if (add_overflows(row_count, nnz_count, &result.correctness_items)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "tile correctness denominator overflows u64");
    }
    validation_result status = build_projection_bytes(row_count, nnz_count,
        tile_count, tile_block_union, active_blocks, value_size_bytes,
        &result.metadata_bytes, &result.encoded_bytes, &result.baseline_bytes);
    if (!status) return status;
    if (nnz_count != 0u) result.available |= packing_validation_metric_storage;
    status = validate_packing_validation_metrics(result);
    if (!status) return status;
    *out = result;
    return validation_ok();
}

struct scalar_accumulator {
    u32 count = 0u;
    double minimum = 0.0;
    double mean = 0.0;
    double maximum = 0.0;
    double sum_squared_delta = 0.0;

    void add(double value) noexcept {
        if (count == 0u) {
            count = 1u;
            minimum = mean = maximum = value;
            return;
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        sum_squared_delta += delta * (value - mean);
    }

    bootstrap_scalar_summary finish() const noexcept {
        bootstrap_scalar_summary result;
        result.observation_count = count;
        if (count == 0u) return result;
        result.minimum = minimum;
        result.mean = mean;
        result.maximum = maximum;
        result.sample_standard_deviation = count > 1u
            ? std::sqrt(sum_squared_delta / static_cast<double>(count - 1u)) : 0.0;
        return result;
    }
};

struct summary_accumulators {
    scalar_accumulator encoded_bytes, metadata_bytes, nnz_count, row_count;
    scalar_accumulator tile_count, tile_union, active_blocks, padding;
    scalar_accumulator encoded_per_nnz, metadata_per_nnz, active_per_row;
    scalar_accumulator union_per_tile, padding_per_nnz;

    void add(const packing_validation_metrics &metrics) noexcept {
        encoded_bytes.add(static_cast<double>(metrics.encoded_bytes));
        metadata_bytes.add(static_cast<double>(metrics.metadata_bytes));
        nnz_count.add(static_cast<double>(metrics.nnz_count));
        row_count.add(static_cast<double>(metrics.row_count));
        tile_count.add(static_cast<double>(metrics.tile_count));
        tile_union.add(static_cast<double>(metrics.tile_block_union_references));
        active_blocks.add(static_cast<double>(metrics.active_block_references));
        padding.add(static_cast<double>(metrics.padding_slots));
        if (metrics.nnz_count != 0u) {
            encoded_per_nnz.add(static_cast<double>(metrics.encoded_bytes)
                / static_cast<double>(metrics.nnz_count));
            metadata_per_nnz.add(static_cast<double>(metrics.metadata_bytes)
                / static_cast<double>(metrics.nnz_count));
            padding_per_nnz.add(static_cast<double>(metrics.padding_slots)
                / static_cast<double>(metrics.nnz_count));
        }
        if (metrics.row_count != 0u) {
            active_per_row.add(static_cast<double>(metrics.active_block_references)
                / static_cast<double>(metrics.row_count));
        }
        if (metrics.tile_count != 0u) {
            union_per_tile.add(static_cast<double>(metrics.tile_block_union_references)
                / static_cast<double>(metrics.tile_count));
        }
    }
};

validation_result validate_realization(
    const record_validation_context &context,
    const validation_bootstrap_provenance &provenance,
    const u32 *multiplicities,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const bootstrap_tile_realization_view &realization) {
    validation_result status = validate_validation_bootstrap(
        context.identities, multiplicities, provenance);
    if (!status) return status;
    if (provenance.unit_kind != context.split_provenance.unit_kind) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "bootstrap and frozen training split use different unit kinds");
    }
    if (records.global_row_begin != 0u || records.row_count != records.full_row_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "bootstrap tile realization requires the complete frozen row domain");
    }
    if (realization.schema_version != bootstrap_tile_realization_schema_version) {
        return validation_error(validation_code::unsupported_version,
            realization.schema_version, "bootstrap tile realization version is unsupported");
    }
    if (realization.bootstrap_identity != provenance.bootstrap_identity
        || realization.materialized_row_count != provenance.materialized_row_count
        || realization.realization_identity == 0u
        || realization.realization_identity != bootstrap_tile_realization_identity(
            provenance, realization.global_row_indices,
            realization.materialized_row_count)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "bootstrap tile realization identity disagrees with its provenance");
    }
    if (realization.materialized_row_count != 0u
        && realization.global_row_indices == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "bootstrap tile realization row sequence is null");
    }
    for (u32 row = 0u; row < context.identities.row_count; ++row) {
        u64 observed = 0u;
        for (u64 index = 0u; index < realization.materialized_row_count; ++index) {
            const u32 global_row = realization.global_row_indices[index];
            if (global_row >= context.identities.row_count) {
                return validation_error(validation_code::invalid_permutation,
                    static_cast<u32>(index),
                    "bootstrap tile realization row is outside the frozen domain");
            }
            if (global_row == row) ++observed;
        }
        if (observed != multiplicities[row]) {
            return validation_error(validation_code::invalid_permutation, row,
                "bootstrap tile realization does not match row multiplicities");
        }
    }
    u32 maximum_multiplicity = 0u;
    for (u32 row = 0u; row < context.identities.row_count; ++row) {
        maximum_multiplicity = std::max(maximum_multiplicity, multiplicities[row]);
    }
    u64 materialized = 0u;
    for (u32 layer = 0u; layer < maximum_multiplicity; ++layer) {
        for (u32 execution = 0u; execution < order.row_count; ++execution) {
            const u32 row = order.row_permutation[execution];
            if (multiplicities[row] <= layer) continue;
            if (materialized >= realization.materialized_row_count
                || realization.global_row_indices[materialized] != row) {
                return validation_error(validation_code::invalid_permutation,
                    static_cast<u32>(materialized),
                    "bootstrap materialization changed the frozen local order");
            }
            ++materialized;
        }
    }
    if (materialized != realization.materialized_row_count) {
        return validation_error(validation_code::invalid_offsets, invalid_id,
            "bootstrap materialization does not exhaust its frozen-order rows");
    }
    return validation_ok();
}

validation_result evaluate_bootstrap_replicate(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const bootstrap_tile_replicate_input &input,
    bootstrap_tile_replicate_validation *out) {
    if (input.bootstrap_provenance == nullptr || input.row_multiplicities == nullptr
        || input.source == nullptr || input.records == nullptr || input.order == nullptr
        || input.tiles == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "bootstrap tile replicate input pointer is null");
    }
    validation_result status = validate_context_and_inputs(plan, context, *input.source,
        *input.records, *input.order, *input.tiles);
    if (!status) return status;
    status = validate_realization(context, *input.bootstrap_provenance,
        input.row_multiplicities, *input.records, *input.order, input.realization);
    if (!status) return status;

    u64 nnz_count = 0u, active_blocks = 0u, tile_union = 0u, padding = 0u;
    const u64 row_count = input.realization.materialized_row_count;
    if (row_count > std::numeric_limits<u32>::max()) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "bootstrap tile realization exceeds the v1 u32 row domain");
    }
    const u32 width = input.tiles->tile_row_width;
    const u64 tile_count = row_count / width + (row_count % width != 0u ? 1u : 0u);
    for (u64 materialized_tile = 0u; materialized_tile < tile_count;
         ++materialized_tile) {
        const u64 begin = materialized_tile * width;
        const u32 lane_count = static_cast<u32>(
            std::min<u64>(width, row_count - begin));
        u64 tile_active = 0u, tile_union_count = 0u;
        for (u32 lane = 0u; lane < lane_count; ++lane) {
            const u32 row = input.realization.global_row_indices[begin + lane];
            u64 row_nnz = 0u, row_blocks = 0u;
            status = validate_row_exactly(plan, *input.source, *input.order,
                *input.tiles, row, &row_nnz, &row_blocks);
            if (!status) return status;
            if (add_overflows(nnz_count, row_nnz, &nnz_count)
                || add_overflows(active_blocks, row_blocks, &active_blocks)
                || add_overflows(tile_active, row_blocks, &tile_active)) {
                return validation_error(validation_code::integer_overflow, row,
                    "bootstrap tile row counters overflow u64");
            }
        }
        for (u32 block = 0u; block < plan.feature_block_count(); ++block) {
            bool present = false;
            for (u32 lane = 0u; lane < lane_count && !present; ++lane) {
                const u32 row = input.realization.global_row_indices[begin + lane];
                const u32 execution = input.order->inverse_row_permutation[row];
                const u32 source_tile = execution / input.tiles->tile_row_width;
                const u32 source_lane = execution % input.tiles->tile_row_width;
                const u32 lane_bit = 1u << source_lane;
                for (u32 descriptor = input.tiles->tile_block_offsets[source_tile];
                     descriptor < input.tiles->tile_block_offsets[source_tile + 1u];
                     ++descriptor) {
                    if (input.tiles->tile_block_ids[descriptor] == block
                        && (input.tiles->tile_block_cell_masks[descriptor]
                            & lane_bit) != 0u) {
                        present = true;
                        break;
                    }
                }
            }
            if (present) ++tile_union_count;
        }
        if (add_overflows(tile_union, tile_union_count, &tile_union)) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "bootstrap tile-union counter overflows u64");
        }
        u64 slots = 0u;
        if (multiply_overflows(tile_union_count, lane_count, &slots)
            || slots < tile_active) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "bootstrap tile padding count overflows");
        }
        if (add_overflows(padding, slots - tile_active, &padding)) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "bootstrap tile padding counter overflows u64");
        }
    }
    if (nnz_count > std::numeric_limits<u32>::max()
        || active_blocks > std::numeric_limits<u32>::max()
        || tile_union > std::numeric_limits<u32>::max()) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "bootstrap realization exceeds CP-BP-08 v1 offset capacity");
    }
    packing_validation_metrics metrics;
    status = finalize_metrics(input.source->dataset_identity,
        context.feature_axis_identity, context.row_domain_identity,
        input.bootstrap_provenance->bootstrap_identity, row_count,
        input.source->support.feature_count, nnz_count, tile_count, tile_union,
        active_blocks, padding, input.source->value_size_bytes, &metrics);
    if (!status) return status;
    bootstrap_tile_replicate_validation result;
    result.bootstrap_identity = input.bootstrap_provenance->bootstrap_identity;
    result.realization_identity = input.realization.realization_identity;
    result.frozen_plan_identity = frozen_plan_identity(plan);
    result.tile_identity = input.tiles->tile_identity;
    result.ordering_identity = input.order->ordering_identity;
    result.metrics = metrics;
    *out = result;
    return validation_ok();
}

} // namespace

u64 bootstrap_tile_realization_identity(
    const validation_bootstrap_provenance &provenance,
    const u32 *global_row_indices,
    u64 materialized_row_count) noexcept {
    if (provenance.bootstrap_identity == 0u
        || provenance.materialized_row_count != materialized_row_count
        || (materialized_row_count != 0u && global_row_indices == nullptr)) return 0u;
    u64 hash = fnv1a_offset;
    hash_u64(&hash, bootstrap_realization_domain);
    hash_u64(&hash, bootstrap_tile_realization_schema_version);
    hash_u64(&hash, provenance.bootstrap_identity);
    hash_u64(&hash, materialized_row_count);
    for (u64 index = 0u; index < materialized_row_count; ++index) {
        hash_u64(&hash, global_row_indices[index]);
    }
    return nonzero_hash(hash);
}

validation_result evaluate_held_out_warp_tiles(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &source,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    held_out_tile_validation *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "held-out tile validation output is null");
    }
    validation_result status = validate_context_and_inputs(
        plan, context, source, records, order, tiles);
    if (!status) return status;
    u64 held_out_rows = 0u, held_out_nnz = 0u, active_blocks = 0u;
    u64 tile_count = 0u, tile_union = 0u, padding = 0u;
    u64 row_identity = fnv1a_offset;
    hash_u64(&row_identity, held_out_identity_domain);
    hash_u64(&row_identity, context.split_provenance.assignment_identity);
    hash_u64(&row_identity, records.global_row_begin);

    for (u32 row = 0u; row < records.row_count; ++row) {
        const u64 global_row = records.global_row_begin + row;
        if (context.row_partitions[global_row] != validation_partition::held_out) continue;
        u64 row_nnz = 0u, row_blocks = 0u;
        status = validate_row_exactly(
            plan, source, order, tiles, row, &row_nnz, &row_blocks);
        if (!status) return status;
        ++held_out_rows;
        held_out_nnz += row_nnz;
        active_blocks += row_blocks;
        hash_u64(&row_identity, global_row);
        hash_u64(&row_identity, context.identities.row_identities[global_row]);
    }
    if (held_out_rows == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "tile partition contains no held-out rows");
    }
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 execution_begin = tile * tiles.tile_row_width;
        const u32 lane_count = std::min(tiles.tile_row_width,
            tiles.row_count - execution_begin);
        u32 selected_mask = 0u;
        for (u32 lane = 0u; lane < lane_count; ++lane) {
            const u32 row = order.row_permutation[execution_begin + lane];
            const u64 global_row = records.global_row_begin + row;
            if (context.row_partitions[global_row] == validation_partition::held_out) {
                selected_mask |= 1u << lane;
            }
        }
        const u32 selected_rows = popcount_u32(selected_mask);
        if (selected_rows == 0u) continue;
        ++tile_count;
        u64 selected_active = 0u, selected_union = 0u;
        for (u32 descriptor = tiles.tile_block_offsets[tile];
             descriptor < tiles.tile_block_offsets[tile + 1u]; ++descriptor) {
            const u32 selected_cells = popcount_u32(
                tiles.tile_block_cell_masks[descriptor] & selected_mask);
            if (selected_cells == 0u) continue;
            ++selected_union;
            selected_active += selected_cells;
        }
        tile_union += selected_union;
        u64 slots = 0u;
        if (multiply_overflows(selected_union, selected_rows, &slots)
            || slots < selected_active) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "held-out tile padding count overflows");
        }
        padding += slots - selected_active;
    }
    packing_validation_metrics metrics;
    status = finalize_metrics(source.dataset_identity, context.feature_axis_identity,
        context.row_domain_identity, context.split_provenance.assignment_identity,
        held_out_rows, source.support.feature_count, held_out_nnz, tile_count,
        tile_union, active_blocks, padding, source.value_size_bytes, &metrics);
    if (!status) return status;
    held_out_tile_validation result;
    result.frozen_plan_identity = frozen_plan_identity(plan);
    result.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    result.tile_identity = tiles.tile_identity;
    result.ordering_identity = order.ordering_identity;
    result.held_out_row_identity = nonzero_hash(row_identity);
    result.plan_training_split_identity = context.plan_training_split_identity;
    result.unit_kind = context.split_provenance.unit_kind;
    result.claims_group_generalization =
        context.split_provenance.claims_group_generalization;
    result.metrics = metrics;
    *out = result;
    return validation_ok();
}

validation_result compare_held_out_warp_tiles_to_degree_null(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &real_source,
    const cell_block_record_view &real_records,
    const local_cell_order_view &real_order,
    const warp_tile_view &real_tiles,
    const record_validation_source_view &null_source,
    const cell_block_record_view &null_records,
    const local_cell_order_view &null_order,
    const warp_tile_view &null_tiles,
    const degree_preserving_null_provenance &null_provenance,
    held_out_tile_null_comparison *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "held-out tile null-comparison output is null");
    }
    if (real_source.dataset_identity != null_provenance.source_identity
        || null_source.dataset_identity != null_provenance.output_identity
        || real_source.value_size_bytes != null_source.value_size_bytes) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "real/null tile sources disagree with null provenance");
    }
    validation_result status = validate_degree_preserving_null_provenance(
        real_source.support, null_source.support, null_provenance);
    if (!status) return status;
    held_out_tile_null_comparison result;
    status = evaluate_held_out_warp_tiles(plan, context, real_source, real_records,
        real_order, real_tiles, &result.real);
    if (!status) return status;
    status = evaluate_held_out_warp_tiles(plan, context, null_source, null_records,
        null_order, null_tiles, &result.degree_preserving_null);
    if (!status) return status;
    for (u32 execution = 0u; execution < real_order.row_count; ++execution) {
        if (real_order.row_permutation[execution]
            != null_order.row_permutation[execution]) {
            return validation_error(validation_code::invalid_permutation, execution,
                "degree-null evaluation relearned the frozen local row order");
        }
    }
    if (result.real.frozen_plan_identity
            != result.degree_preserving_null.frozen_plan_identity
        || result.real.held_out_row_identity
            != result.degree_preserving_null.held_out_row_identity
        || result.real.ordering_identity
            != result.degree_preserving_null.ordering_identity
        || result.real.metrics.row_count
            != result.degree_preserving_null.metrics.row_count
        || result.real.metrics.nnz_count
            != result.degree_preserving_null.metrics.nnz_count) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "real/null tiles do not share one frozen plan, order, and held-out split");
    }
    result.encoded_bytes_absolute_difference = absolute_difference(
        result.real.metrics.encoded_bytes,
        result.degree_preserving_null.metrics.encoded_bytes);
    result.metadata_bytes_absolute_difference = absolute_difference(
        result.real.metrics.metadata_bytes,
        result.degree_preserving_null.metrics.metadata_bytes);
    result.tile_block_union_absolute_difference = absolute_difference(
        result.real.metrics.tile_block_union_references,
        result.degree_preserving_null.metrics.tile_block_union_references);
    result.active_block_references_absolute_difference = absolute_difference(
        result.real.metrics.active_block_references,
        result.degree_preserving_null.metrics.active_block_references);
    result.padding_slots_absolute_difference = absolute_difference(
        result.real.metrics.padding_slots,
        result.degree_preserving_null.metrics.padding_slots);
    result.real_encoded_bytes_no_greater = result.real.metrics.encoded_bytes
        <= result.degree_preserving_null.metrics.encoded_bytes;
    result.real_metadata_bytes_no_greater = result.real.metrics.metadata_bytes
        <= result.degree_preserving_null.metrics.metadata_bytes;
    result.real_tile_union_no_greater = result.real.metrics.tile_block_union_references
        <= result.degree_preserving_null.metrics.tile_block_union_references;
    result.real_active_blocks_no_greater = result.real.metrics.active_block_references
        <= result.degree_preserving_null.metrics.active_block_references;
    result.real_padding_no_greater = result.real.metrics.padding_slots
        <= result.degree_preserving_null.metrics.padding_slots;
    result.exact_degree_conservation = null_provenance.row_degrees_exact
        && null_provenance.feature_degrees_exact;
    *out = result;
    return validation_ok();
}

validation_result evaluate_bootstrap_warp_tile_stability(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const bootstrap_tile_replicate_input *inputs,
    u32 input_count,
    const bootstrap_tile_validation_buffers &buffers,
    bootstrap_tile_stability_summary *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "bootstrap tile stability output is null");
    }
    if (input_count == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "bootstrap tile stability requires at least one replicate");
    }
    if (inputs == nullptr || buffers.replicates == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "bootstrap tile input or output array is null");
    }
    if (buffers.replicate_capacity < input_count) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "bootstrap tile replicate output capacity is insufficient");
    }
    summary_accumulators accumulators;
    u64 expected_tile_identity = 0u, expected_ordering_identity = 0u;
    u64 expected_dataset_identity = 0u;
    for (u32 index = 0u; index < input_count; ++index) {
        for (u32 prior = 0u; prior < index; ++prior) {
            if (inputs[index].bootstrap_provenance != nullptr
                && inputs[prior].bootstrap_provenance != nullptr
                && inputs[index].bootstrap_provenance->bootstrap_identity
                    == inputs[prior].bootstrap_provenance->bootstrap_identity) {
                return validation_error(validation_code::duplicate_id, index,
                    "bootstrap tile replicate identity is duplicated");
            }
        }
        validation_result status = evaluate_bootstrap_replicate(
            plan, context, inputs[index], &buffers.replicates[index]);
        if (!status) return status;
        if (index == 0u) {
            expected_tile_identity = buffers.replicates[index].tile_identity;
            expected_ordering_identity = buffers.replicates[index].ordering_identity;
            expected_dataset_identity = buffers.replicates[index].metrics.dataset_identity;
        } else if (buffers.replicates[index].tile_identity != expected_tile_identity
            || buffers.replicates[index].ordering_identity != expected_ordering_identity
            || buffers.replicates[index].metrics.dataset_identity
                != expected_dataset_identity) {
            return validation_error(validation_code::invalid_plan_geometry, index,
                "bootstrap replicates changed source tiles, order, or dataset");
        }
        accumulators.add(buffers.replicates[index].metrics);
    }
    bootstrap_tile_stability_summary result;
    result.repeat_count = input_count;
    result.frozen_plan_identity = frozen_plan_identity(plan);
    result.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    result.dataset_identity = expected_dataset_identity;
    result.feature_axis_identity = context.feature_axis_identity;
    result.row_domain_identity = context.row_domain_identity;
    result.tile_identity = expected_tile_identity;
    result.ordering_identity = expected_ordering_identity;
    result.plan_training_split_identity = context.plan_training_split_identity;
    result.unit_kind = context.split_provenance.unit_kind;
    result.claims_group_generalization =
        context.split_provenance.claims_group_generalization;
    result.encoded_bytes = accumulators.encoded_bytes.finish();
    result.metadata_bytes = accumulators.metadata_bytes.finish();
    result.nnz_count = accumulators.nnz_count.finish();
    result.row_count = accumulators.row_count.finish();
    result.tile_count = accumulators.tile_count.finish();
    result.tile_block_union_references = accumulators.tile_union.finish();
    result.active_block_references = accumulators.active_blocks.finish();
    result.padding_slots = accumulators.padding.finish();
    result.encoded_bytes_per_nnz = accumulators.encoded_per_nnz.finish();
    result.metadata_bytes_per_nnz = accumulators.metadata_per_nnz.finish();
    result.active_blocks_per_row = accumulators.active_per_row.finish();
    result.tile_block_union_per_tile = accumulators.union_per_tile.finish();
    result.padding_slots_per_nnz = accumulators.padding_per_nnz.finish();
    *out = result;
    return validation_ok();
}

} // namespace cellpack
