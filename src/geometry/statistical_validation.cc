#include "Cellerator/geometry/statistical_validation.hh"

#include <Cellerator/compute/sampling.hh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace cellpack {
namespace {

namespace sampling = ::cellerator::compute::sampling;

constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;
constexpr u64 split_domain = 0x4350425031315350ull;
constexpr u64 bootstrap_domain = 0x4350425031314254ull;
constexpr u64 null_domain = 0x4350425031314e55ull;

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

u64 random_word(u64 seed, u64 domain, u64 counter) noexcept {
    const u64 keyed = sampling::splitmix64_hash(seed ^ domain);
    return sampling::splitmix64_hash(keyed + counter);
}

u64 bounded_random(u64 word, u64 bound) noexcept {
    return static_cast<u64>((static_cast<unsigned __int128>(word) * bound) >> 64u);
}

struct validation_units {
    validation_unit_kind kind = validation_unit_kind::row_identity;
    u32 row_count = 0u;
    u32 unit_count = 0u;
    std::unique_ptr<u64[]> identities;
    std::unique_ptr<u32[]> row_to_unit;
    std::unique_ptr<u32[]> unit_offsets;
    std::unique_ptr<u32[]> unit_rows;
};

struct group_row_pair { u64 identity = 0u; u32 row = 0u; };

validation_result collect_validation_units(
    const validation_identity_view &source,
    validation_units *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "validation unit output is null");
    }
    if (source.row_count == 0u || source.row_identities == nullptr) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "validation row identities must describe a nonempty row axis");
    }
    std::unique_ptr<u64[]> unique_rows(new (std::nothrow) u64[source.row_count]);
    if (unique_rows == nullptr) return validation_error(validation_code::integer_overflow,
        invalid_id, "validation identity scratch allocation failed");
    for (u32 row = 0u; row < source.row_count; ++row) {
        unique_rows[row] = source.row_identities[row];
    }
    std::sort(unique_rows.get(), unique_rows.get() + source.row_count);
    for (u32 row = 1u; row < source.row_count; ++row) {
        if (unique_rows[row] == unique_rows[row - 1u]) {
            return validation_error(validation_code::duplicate_id, row,
                "validation row identities must be unique");
        }
    }

    validation_units result;
    result.kind = source.group_identities == nullptr
        ? validation_unit_kind::row_identity
        : validation_unit_kind::caller_group_identity;
    result.row_count = source.row_count;
    result.row_to_unit.reset(new (std::nothrow) u32[source.row_count]);
    result.unit_rows.reset(new (std::nothrow) u32[source.row_count]);
    if (result.row_to_unit == nullptr || result.unit_rows == nullptr) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "validation relation allocation failed");
    }
    if (source.group_identities == nullptr) {
        result.unit_count = source.row_count;
        result.identities.reset(new (std::nothrow) u64[source.row_count]);
        result.unit_offsets.reset(new (std::nothrow) u32[static_cast<std::size_t>(source.row_count) + 1u]);
        if (result.identities == nullptr || result.unit_offsets == nullptr) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "validation row-unit allocation failed");
        }
        for (u32 row = 0u; row < source.row_count; ++row) {
            result.identities[row] = source.row_identities[row];
            result.row_to_unit[row] = row;
            result.unit_offsets[row] = row;
            result.unit_rows[row] = row;
        }
        result.unit_offsets[source.row_count] = source.row_count;
    } else {
        std::unique_ptr<group_row_pair[]> pairs(
            new (std::nothrow) group_row_pair[source.row_count]);
        if (pairs == nullptr) return validation_error(validation_code::integer_overflow,
            invalid_id, "validation group-row scratch allocation failed");
        for (u32 row = 0u; row < source.row_count; ++row) {
            pairs[row] = {source.group_identities[row], row};
        }
        std::sort(pairs.get(), pairs.get() + source.row_count,
            [](const group_row_pair &lhs, const group_row_pair &rhs) {
                return lhs.identity != rhs.identity ? lhs.identity < rhs.identity : lhs.row < rhs.row;
            });
        result.unit_count = 1u;
        for (u32 i = 1u; i < source.row_count; ++i) {
            result.unit_count += pairs[i].identity != pairs[i - 1u].identity ? 1u : 0u;
        }
        result.identities.reset(new (std::nothrow) u64[result.unit_count]);
        result.unit_offsets.reset(new (std::nothrow) u32[static_cast<std::size_t>(result.unit_count) + 1u]);
        if (result.identities == nullptr || result.unit_offsets == nullptr) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "validation grouped relation allocation failed");
        }
        u32 unit = 0u;
        result.unit_offsets[0] = 0u;
        for (u32 i = 0u; i < source.row_count; ++i) {
            if (i != 0u && pairs[i].identity != pairs[i - 1u].identity) {
                result.unit_offsets[++unit] = i;
            }
            result.identities[unit] = pairs[i].identity;
            result.unit_rows[i] = pairs[i].row;
            result.row_to_unit[pairs[i].row] = unit;
        }
        result.unit_offsets[result.unit_count] = source.row_count;
    }
    *out = std::move(result);
    return validation_ok();
}

u64 split_identity(
    const validation_identity_view &identities,
    const validation_partition *partitions,
    const validation_split_provenance &provenance) noexcept {
    u64 hash = fnv1a_offset;
    hash_u64(&hash, split_domain);
    hash_u64(&hash, provenance.schema_version);
    hash_u64(&hash, provenance.algorithm_version);
    hash_u64(&hash, provenance.seed);
    hash_u64(&hash, static_cast<u64>(provenance.unit_kind));
    hash_u64(&hash, identities.row_count);
    for (u32 row = 0u; row < identities.row_count; ++row) {
        hash_u64(&hash, identities.row_identities[row]);
        hash_u64(&hash, identities.group_identities == nullptr
            ? identities.row_identities[row]
            : identities.group_identities[row]);
        hash_u64(&hash, static_cast<u64>(partitions[row]));
    }
    return nonzero_hash(hash);
}

u64 bootstrap_identity(
    const validation_identity_view &identities,
    const u32 *multiplicities,
    const validation_bootstrap_provenance &provenance) noexcept {
    u64 hash = fnv1a_offset;
    hash_u64(&hash, bootstrap_domain);
    hash_u64(&hash, provenance.schema_version);
    hash_u64(&hash, provenance.algorithm_version);
    hash_u64(&hash, provenance.seed);
    hash_u64(&hash, static_cast<u64>(provenance.unit_kind));
    hash_u64(&hash, provenance.unit_draw_count);
    for (u32 row = 0u; row < identities.row_count; ++row) {
        hash_u64(&hash, identities.row_identities[row]);
        hash_u64(&hash, identities.group_identities == nullptr
            ? identities.row_identities[row]
            : identities.group_identities[row]);
        hash_u64(&hash, multiplicities[row]);
    }
    return nonzero_hash(hash);
}

u64 matrix_identity(const csr_support_view &source) noexcept {
    u64 hash = fnv1a_offset;
    hash_u64(&hash, null_domain);
    hash_u64(&hash, source.row_count);
    hash_u64(&hash, source.feature_count);
    hash_u64(&hash, source.nnz_count);
    if (source.row_count != 0u) {
        for (u32 row = 0u; row <= source.row_count; ++row) {
            hash_u64(&hash, source.row_offsets[row]);
        }
    }
    for (u32 entry = 0u; entry < source.nnz_count; ++entry) {
        hash_u64(&hash, source.feature_ids[entry]);
    }
    return nonzero_hash(hash);
}

u64 edge_key(u32 row, u32 feature) noexcept {
    return (static_cast<u64>(row) << 32u) | feature;
}

struct edge_membership_table {
    std::unique_ptr<u64[]> keys;
    std::unique_ptr<unsigned char[]> states; // 0 empty, 1 occupied, 2 tombstone
    std::size_t capacity = 0u;

    bool initialize(std::size_t edge_count) noexcept {
        capacity = 8u;
        if (edge_count > SIZE_MAX / 4u) return false;
        while (capacity < edge_count * 4u) {
            if (capacity > SIZE_MAX / 2u) return false;
            capacity *= 2u;
        }
        keys.reset(new (std::nothrow) u64[capacity]);
        states.reset(new (std::nothrow) unsigned char[capacity]());
        return keys != nullptr && states != nullptr;
    }

    std::size_t slot(u64 key) const noexcept {
        return static_cast<std::size_t>(sampling::splitmix64_hash(key)) & (capacity - 1u);
    }

    bool contains(u64 key) const noexcept {
        std::size_t index = slot(key);
        for (std::size_t probe = 0u; probe < capacity; ++probe) {
            if (states[index] == 0u) return false;
            if (states[index] == 1u && keys[index] == key) return true;
            index = (index + 1u) & (capacity - 1u);
        }
        return false;
    }

    bool insert(u64 key) noexcept {
        std::size_t index = slot(key), tombstone = capacity;
        for (std::size_t probe = 0u; probe < capacity; ++probe) {
            if (states[index] == 1u && keys[index] == key) return false;
            if (states[index] == 2u && tombstone == capacity) tombstone = index;
            if (states[index] == 0u) {
                const std::size_t target = tombstone == capacity ? index : tombstone;
                keys[target] = key;
                states[target] = 1u;
                return true;
            }
            index = (index + 1u) & (capacity - 1u);
        }
        if (tombstone != capacity) {
            keys[tombstone] = key;
            states[tombstone] = 1u;
            return true;
        }
        return false;
    }

    bool erase(u64 key) noexcept {
        std::size_t index = slot(key);
        for (std::size_t probe = 0u; probe < capacity; ++probe) {
            if (states[index] == 0u) return false;
            if (states[index] == 1u && keys[index] == key) {
                states[index] = 2u;
                return true;
            }
            index = (index + 1u) & (capacity - 1u);
        }
        return false;
    }
};

} // namespace

validation_result validate_packing_validation_metrics(
    const packing_validation_metrics &metrics) {
    if (metrics.schema_version != packing_validation_schema_version) {
        return validation_error(validation_code::unsupported_version,
            metrics.schema_version, "unsupported packing-validation metric schema");
    }
    constexpr u32 known_metrics = packing_validation_metric_storage
        | packing_validation_metric_records | packing_validation_metric_tiles
        | packing_validation_metric_preprocessing | packing_validation_metric_runtime
        | packing_validation_metric_correctness
        | packing_validation_metric_workload_profile;
    if ((metrics.available & ~known_metrics) != 0u) {
        return validation_error(validation_code::unsupported_version,
            metrics.available, "packing-validation metrics contain unknown availability flags");
    }
    if (metrics.dataset_identity == 0u || metrics.feature_axis_identity == 0u
        || metrics.row_domain_identity == 0u || metrics.split_identity == 0u
        || metrics.row_count == 0u || metrics.feature_count == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "packing-validation metrics lack dataset, axis, row, split, or dimension context");
    }
    if ((metrics.available & packing_validation_metric_storage) != 0u
        && (metrics.nnz_count == 0u || metrics.encoded_bytes == 0u
            || metrics.baseline_bytes == 0u || metrics.metadata_bytes > metrics.encoded_bytes)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "storage metrics lack a valid NNZ/byte denominator");
    }
    if ((metrics.available & packing_validation_metric_tiles) != 0u
        && metrics.tile_count == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "tile metrics lack a tile-count denominator");
    }
    if ((metrics.available & packing_validation_metric_preprocessing) != 0u
        && (metrics.preprocessing_input_nnz == 0u
            || metrics.preprocessing_elapsed_nanoseconds == 0u
            || metrics.preprocessing_repeat_count == 0u)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "preprocessing metrics lack work, time, or repeat denominators");
    }
    if ((metrics.available & packing_validation_metric_runtime) != 0u
        && (metrics.runtime_input_nnz == 0u || metrics.runtime_bytes == 0u
            || metrics.runtime_elapsed_nanoseconds == 0u
            || metrics.runtime_repeat_count == 0u)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "runtime metrics lack work, bytes, time, or repeat denominators");
    }
    if ((metrics.available & packing_validation_metric_correctness) != 0u
        && (metrics.correctness_items == 0u
            || metrics.correctness_mismatches > metrics.correctness_items)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "correctness metrics lack a valid comparison denominator");
    }
    if ((metrics.available & packing_validation_metric_workload_profile) != 0u
        && (metrics.workload_profile_identity == 0u
            || metrics.workload_evidence_revision == 0u
            || metrics.forward_elapsed_nanoseconds == 0u
            || metrics.transpose_elapsed_nanoseconds == 0u
            || metrics.forward_repeat_count == 0u
            || metrics.transpose_repeat_count == 0u
            || metrics.bootstrap_median_total_nanoseconds == 0u
            || metrics.bootstrap_sample_count == 0u
            || metrics.bootstrap_mad_nanoseconds
                > metrics.bootstrap_median_total_nanoseconds)) {
        return validation_error(validation_code::invalid_plan_geometry,
            invalid_id,
            "workload profile lacks identities, repeats, or bootstrap evidence");
    }
    return validation_ok();
}

validation_result derive_packing_validation_metric_rates(
    const packing_validation_metrics &metrics,
    packing_validation_metric_rates *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "packing-validation derived metric output is null");
    }
    const validation_result status = validate_packing_validation_metrics(metrics);
    if (!status) return status;
    packing_validation_metric_rates result;
    if ((metrics.available & packing_validation_metric_storage) != 0u) {
        result.encoded_bytes_per_nnz = static_cast<double>(metrics.encoded_bytes)
            / static_cast<double>(metrics.nnz_count);
        result.metadata_bytes_per_nnz = static_cast<double>(metrics.metadata_bytes)
            / static_cast<double>(metrics.nnz_count);
        result.compression_ratio = static_cast<double>(metrics.baseline_bytes)
            / static_cast<double>(metrics.encoded_bytes);
        result.padding_slots_per_nnz = static_cast<double>(metrics.padding_slots)
            / static_cast<double>(metrics.nnz_count);
    }
    if ((metrics.available & packing_validation_metric_records) != 0u) {
        result.active_blocks_per_row = static_cast<double>(metrics.active_block_references)
            / static_cast<double>(metrics.row_count);
    }
    if ((metrics.available & packing_validation_metric_tiles) != 0u) {
        result.tile_block_union_per_tile = static_cast<double>(metrics.tile_block_union_references)
            / static_cast<double>(metrics.tile_count);
    }
    if ((metrics.available & packing_validation_metric_preprocessing) != 0u) {
        result.preprocessing_nnz_per_second = static_cast<double>(metrics.preprocessing_input_nnz)
            * 1.0e9 / static_cast<double>(metrics.preprocessing_elapsed_nanoseconds);
    }
    if ((metrics.available & packing_validation_metric_runtime) != 0u) {
        result.runtime_nnz_per_second = static_cast<double>(metrics.runtime_input_nnz)
            * 1.0e9 / static_cast<double>(metrics.runtime_elapsed_nanoseconds);
        result.runtime_gigabytes_per_second = static_cast<double>(metrics.runtime_bytes)
            / static_cast<double>(metrics.runtime_elapsed_nanoseconds);
    }
    if ((metrics.available & packing_validation_metric_correctness) != 0u) {
        result.exact_correctness = metrics.correctness_mismatches == 0u;
    }
    *out = result;
    return validation_ok();
}

validation_result build_validation_split(
    const validation_identity_view &identities,
    const validation_split_config &config,
    const validation_split_buffers &buffers,
    validation_split_provenance *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "validation split provenance output is null");
    }
    if (buffers.row_capacity < identities.row_count
        || (identities.row_count != 0u && buffers.row_partitions == nullptr)) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "validation split row buffer is insufficient");
    }
    try {
        validation_units units;
        validation_result status = collect_validation_units(identities, &units);
        if (!status) return status;
        if (config.held_out_unit_count == 0u
            || config.held_out_unit_count >= units.unit_count) {
            return validation_error(validation_code::invalid_plan_geometry,
                config.held_out_unit_count,
                "held-out unit count must leave nonempty training and held-out sets");
        }
        std::unique_ptr<u32[]> order(new (std::nothrow) u32[units.unit_count]);
        std::unique_ptr<unsigned char[]> held_out(new (std::nothrow) unsigned char[units.unit_count]());
        if (order == nullptr || held_out == nullptr) return validation_error(
            validation_code::integer_overflow, invalid_id, "validation split scratch allocation failed");
        for (u32 unit = 0u; unit < units.unit_count; ++unit) order[unit] = unit;
        std::sort(order.get(), order.get() + units.unit_count, [&](u32 lhs, u32 rhs) {
            const u64 lhs_hash = random_word(config.seed, split_domain, units.identities[lhs]);
            const u64 rhs_hash = random_word(config.seed, split_domain, units.identities[rhs]);
            return lhs_hash != rhs_hash
                ? lhs_hash < rhs_hash
                : units.identities[lhs] < units.identities[rhs];
        });
        for (u32 index = 0u; index < config.held_out_unit_count; ++index) {
            held_out[order[index]] = 1u;
        }
        u32 held_out_rows = 0u;
        for (u32 row = 0u; row < identities.row_count; ++row) {
            const bool selected = held_out[units.row_to_unit[row]] != 0u;
            buffers.row_partitions[row] = selected
                ? validation_partition::held_out
                : validation_partition::training;
            held_out_rows += selected ? 1u : 0u;
        }
        validation_split_provenance result;
        result.seed = config.seed;
        result.unit_kind = units.kind;
        result.row_count = identities.row_count;
        result.unit_count = units.unit_count;
        result.held_out_unit_count = config.held_out_unit_count;
        result.training_unit_count = result.unit_count - result.held_out_unit_count;
        result.held_out_row_count = held_out_rows;
        result.training_row_count = result.row_count - result.held_out_row_count;
        result.claims_group_generalization = identities.group_identities != nullptr;
        result.assignment_identity = split_identity(
            identities, buffers.row_partitions, result);
        *out = result;
        return validation_ok();
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "validation split allocation failed");
    }
}

validation_result validate_validation_split(
    const validation_identity_view &identities,
    const validation_partition *row_partitions,
    const validation_split_provenance &provenance) {
    if (row_partitions == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "validation split assignment is null");
    }
    if (provenance.schema_version != packing_validation_schema_version
        || provenance.algorithm_version != validation_split_algorithm_version) {
        return validation_error(validation_code::unsupported_version,
            provenance.algorithm_version, "unsupported validation split provenance");
    }
    try {
        validation_units units;
        validation_result status = collect_validation_units(identities, &units);
        if (!status) return status;
        if (provenance.row_count != identities.row_count
            || provenance.unit_count != units.unit_count
            || provenance.unit_kind != units.kind
            || provenance.claims_group_generalization != (identities.group_identities != nullptr)) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "validation split provenance dimensions or unit semantics disagree");
        }
        std::unique_ptr<validation_partition[]> unit_partition(
            new (std::nothrow) validation_partition[units.unit_count]());
        if (unit_partition == nullptr) return validation_error(validation_code::integer_overflow,
            invalid_id, "validation split verification scratch allocation failed");
        u32 training_rows = 0u, held_out_rows = 0u;
        for (u32 row = 0u; row < identities.row_count; ++row) {
            const validation_partition partition = row_partitions[row];
            if (partition != validation_partition::training
                && partition != validation_partition::held_out) {
                return validation_error(validation_code::invalid_permutation, row,
                    "validation split contains an unsupported assignment");
            }
            const u32 unit = units.row_to_unit[row];
            if (static_cast<u32>(unit_partition[unit]) == 0u) {
                unit_partition[unit] = partition;
            } else if (unit_partition[unit] != partition) {
                return validation_error(validation_code::invalid_permutation, row,
                    "one validation group crosses training and held-out partitions");
            }
            training_rows += partition == validation_partition::training ? 1u : 0u;
            held_out_rows += partition == validation_partition::held_out ? 1u : 0u;
        }
        u32 training_units = 0u, held_out_units = 0u;
        for (u32 unit = 0u; unit < units.unit_count; ++unit) {
            const validation_partition partition = unit_partition[unit];
            training_units += partition == validation_partition::training ? 1u : 0u;
            held_out_units += partition == validation_partition::held_out ? 1u : 0u;
        }
        if (training_rows != provenance.training_row_count
            || held_out_rows != provenance.held_out_row_count
            || training_units != provenance.training_unit_count
            || held_out_units != provenance.held_out_unit_count
            || split_identity(identities, row_partitions, provenance)
                != provenance.assignment_identity) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "validation split counts or immutable identity disagree");
        }
        return validation_ok();
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "validation split verification allocation failed");
    }
}

validation_result build_validation_bootstrap(
    const validation_identity_view &identities,
    const validation_bootstrap_config &config,
    const validation_bootstrap_buffers &buffers,
    validation_bootstrap_provenance *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "validation bootstrap provenance output is null");
    }
    if (buffers.row_capacity < identities.row_count
        || (identities.row_count != 0u && buffers.row_multiplicities == nullptr)) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "validation bootstrap row buffer is insufficient");
    }
    if (config.unit_draw_count == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "validation bootstrap draw count must be nonzero");
    }
    try {
        validation_units units;
        validation_result status = collect_validation_units(identities, &units);
        if (!status) return status;
        std::fill(buffers.row_multiplicities,
            buffers.row_multiplicities + identities.row_count, 0u);
        for (u32 draw = 0u; draw < config.unit_draw_count; ++draw) {
            const u32 unit = static_cast<u32>(bounded_random(
                random_word(config.seed, bootstrap_domain, draw),
                units.unit_count));
            for (u32 offset = units.unit_offsets[unit]; offset < units.unit_offsets[unit + 1u]; ++offset) {
                const u32 row = units.unit_rows[offset];
                ++buffers.row_multiplicities[row];
            }
        }
        u64 materialized_rows = 0u;
        for (u32 row = 0u; row < identities.row_count; ++row) {
            materialized_rows += buffers.row_multiplicities[row];
        }
        validation_bootstrap_provenance result;
        result.seed = config.seed;
        result.unit_kind = units.kind;
        result.row_count = identities.row_count;
        result.unit_count = units.unit_count;
        result.unit_draw_count = config.unit_draw_count;
        result.materialized_row_count = materialized_rows;
        result.bootstrap_identity = bootstrap_identity(
            identities, buffers.row_multiplicities, result);
        *out = result;
        return validation_ok();
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "validation bootstrap allocation failed");
    }
}

validation_result validate_validation_bootstrap(
    const validation_identity_view &identities,
    const u32 *row_multiplicities,
    const validation_bootstrap_provenance &provenance) {
    if (row_multiplicities == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "validation bootstrap multiplicities are null");
    }
    if (provenance.schema_version != packing_validation_schema_version
        || provenance.algorithm_version != validation_bootstrap_algorithm_version) {
        return validation_error(validation_code::unsupported_version,
            provenance.algorithm_version, "unsupported validation bootstrap provenance");
    }
    try {
        validation_units units;
        validation_result status = collect_validation_units(identities, &units);
        if (!status) return status;
        if (provenance.row_count != identities.row_count
            || provenance.unit_count != units.unit_count
            || provenance.unit_kind != units.kind
            || provenance.unit_draw_count == 0u) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "validation bootstrap provenance dimensions or unit semantics disagree");
        }
        u64 materialized_rows = 0u;
        u64 observed_unit_draws = 0u;
        for (u32 unit = 0u; unit < units.unit_count; ++unit) {
            const u32 begin = units.unit_offsets[unit], end = units.unit_offsets[unit + 1u];
            const u32 unit_multiplicity = row_multiplicities[units.unit_rows[begin]];
            observed_unit_draws += unit_multiplicity;
            for (u32 offset = begin; offset < end; ++offset) {
                const u32 row = units.unit_rows[offset];
                if (row_multiplicities[row] != unit_multiplicity) {
                    return validation_error(validation_code::invalid_permutation, row,
                        "one validation bootstrap group has inconsistent multiplicity");
                }
                materialized_rows += row_multiplicities[row];
            }
        }
        if (observed_unit_draws != provenance.unit_draw_count
            || materialized_rows != provenance.materialized_row_count
            || bootstrap_identity(identities, row_multiplicities, provenance)
                != provenance.bootstrap_identity) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                "validation bootstrap count or immutable identity disagrees");
        }
        return validation_ok();
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "validation bootstrap verification allocation failed");
    }
}

validation_result query_degree_preserving_null_requirements(
    const csr_support_view &source,
    degree_preserving_null_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "degree-preserving null requirements output is null");
    }
    const validation_result status = validate_csr_support_view(source);
    if (!status) return status;
    degree_preserving_null_requirements result;
    result.row_offset_capacity = source.row_count == 0u
        ? 0u
        : static_cast<std::size_t>(source.row_count) + 1u;
    result.feature_capacity = source.nnz_count;
    *out = result;
    return validation_ok();
}

validation_result build_degree_preserving_null_reference(
    const csr_support_view &source,
    const degree_preserving_null_config &config,
    const degree_preserving_null_buffers &buffers,
    csr_support_view *out,
    degree_preserving_null_provenance *provenance) {
    if (out == nullptr || provenance == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "degree-preserving null output or provenance is null");
    }
    const validation_result source_status = validate_csr_support_view(source);
    if (!source_status) return source_status;
    if (config.source_identity == 0u || config.requested_swaps == 0u
        || config.maximum_attempts == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "degree-preserving null config identities and attempt limits must be nonzero");
    }
    degree_preserving_null_requirements required;
    validation_result status = query_degree_preserving_null_requirements(source, &required);
    if (!status) return status;
    if (buffers.row_offset_capacity < required.row_offset_capacity
        || buffers.feature_capacity < required.feature_capacity
        || (required.row_offset_capacity != 0u && buffers.row_offsets == nullptr)
        || (required.feature_capacity != 0u && buffers.feature_ids == nullptr)) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "degree-preserving null output buffers are insufficient");
    }
    if ((required.row_offset_capacity != 0u && buffers.row_offsets == source.row_offsets)
        || (required.feature_capacity != 0u && buffers.feature_ids == source.feature_ids)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "degree-preserving null construction is out-of-place");
    }
    try {
        if (required.row_offset_capacity != 0u) {
            std::copy(source.row_offsets, source.row_offsets + required.row_offset_capacity,
                buffers.row_offsets);
        }
        if (required.feature_capacity != 0u) {
            std::copy(source.feature_ids, source.feature_ids + source.nnz_count,
                buffers.feature_ids);
        }
        std::unique_ptr<u32[]> edge_rows(new (std::nothrow) u32[source.nnz_count]);
        edge_membership_table edges;
        if ((source.nnz_count != 0u && edge_rows == nullptr)
            || !edges.initialize(source.nnz_count)) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "degree-preserving null membership allocation failed");
        }
        for (u32 row = 0u; row < source.row_count; ++row) {
            for (u32 entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u]; ++entry) {
                edge_rows[entry] = row;
                if (!edges.insert(edge_key(row, source.feature_ids[entry]))) {
                    return validation_error(validation_code::duplicate_id, entry,
                        "degree-preserving null source contains a duplicate edge");
                }
            }
        }

        u64 attempts = 0u, accepted = 0u;
        while (attempts < config.maximum_attempts
            && accepted < config.requested_swaps && source.nnz_count >= 2u) {
            const u32 lhs = static_cast<u32>(bounded_random(
                random_word(config.seed, null_domain, attempts * 2u), source.nnz_count));
            const u32 rhs = static_cast<u32>(bounded_random(
                random_word(config.seed, null_domain, attempts * 2u + 1u), source.nnz_count));
            ++attempts;
            if (lhs == rhs) continue;
            const u32 lhs_row = edge_rows[lhs], rhs_row = edge_rows[rhs];
            const u32 lhs_feature = buffers.feature_ids[lhs];
            const u32 rhs_feature = buffers.feature_ids[rhs];
            if (lhs_row == rhs_row || lhs_feature == rhs_feature) continue;
            const u64 lhs_new = edge_key(lhs_row, rhs_feature);
            const u64 rhs_new = edge_key(rhs_row, lhs_feature);
            if (edges.contains(lhs_new) || edges.contains(rhs_new)) continue;
            if (!edges.erase(edge_key(lhs_row, lhs_feature))
                || !edges.erase(edge_key(rhs_row, rhs_feature))
                || !edges.insert(lhs_new) || !edges.insert(rhs_new)) {
                return validation_error(validation_code::invalid_plan_geometry,
                    invalid_id, "degree-preserving null membership state diverged");
            }
            std::swap(buffers.feature_ids[lhs], buffers.feature_ids[rhs]);
            ++accepted;
        }
        if (source.nnz_count != 0u) {
            for (u32 row = 0u; row < source.row_count; ++row) {
                std::sort(buffers.feature_ids + buffers.row_offsets[row],
                    buffers.feature_ids + buffers.row_offsets[row + 1u]);
            }
        }
        csr_support_view result;
        result.row_count = source.row_count;
        result.feature_count = source.feature_count;
        result.nnz_count = source.nnz_count;
        result.row_offsets = buffers.row_offsets;
        result.feature_ids = buffers.feature_ids;
        status = validate_csr_support_view(result);
        if (!status) return status;
        degree_conservation_report conservation;
        status = validate_degree_conservation(source, result, &conservation);
        if (!status) return status;
        degree_preserving_null_provenance result_provenance;
        result_provenance.seed = config.seed;
        result_provenance.source_identity = config.source_identity;
        result_provenance.source_support_identity = matrix_identity(source);
        result_provenance.output_identity = matrix_identity(result);
        result_provenance.row_count = source.row_count;
        result_provenance.feature_count = source.feature_count;
        result_provenance.nnz_count = source.nnz_count;
        result_provenance.requested_swaps = config.requested_swaps;
        result_provenance.maximum_attempts = config.maximum_attempts;
        result_provenance.attempted_swaps = attempts;
        result_provenance.accepted_swaps = accepted;
        result_provenance.target_reached = accepted == config.requested_swaps;
        result_provenance.row_degrees_exact = conservation.row_degree_mismatches == 0u;
        result_provenance.feature_degrees_exact = conservation.feature_degree_mismatches == 0u;
        *out = result;
        *provenance = result_provenance;
        return validation_ok();
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "degree-preserving null allocation failed");
    }
}

validation_result validate_degree_conservation(
    const csr_support_view &source,
    const csr_support_view &candidate,
    degree_conservation_report *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "degree conservation report output is null");
    }
    validation_result status = validate_csr_support_view(source);
    if (!status) return status;
    status = validate_csr_support_view(candidate);
    if (!status) return status;
    if (source.row_count != candidate.row_count
        || source.feature_count != candidate.feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "degree conservation matrices have different dimensions");
    }
    try {
        degree_conservation_report result;
        result.source_nnz = source.nnz_count;
        result.candidate_nnz = candidate.nnz_count;
        for (u32 row = 0u; row < source.row_count; ++row) {
            const u32 source_degree = source.row_offsets[row + 1u] - source.row_offsets[row];
            const u32 candidate_degree = candidate.row_offsets[row + 1u] - candidate.row_offsets[row];
            result.row_degree_mismatches += source_degree == candidate_degree ? 0u : 1u;
        }
        std::unique_ptr<u64[]> source_degrees(
            new (std::nothrow) u64[source.feature_count]());
        std::unique_ptr<u64[]> candidate_degrees(
            new (std::nothrow) u64[candidate.feature_count]());
        if ((source.feature_count != 0u && source_degrees == nullptr)
            || (candidate.feature_count != 0u && candidate_degrees == nullptr)) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "degree-conservation scratch allocation failed");
        }
        for (u32 entry = 0u; entry < source.nnz_count; ++entry) {
            ++source_degrees[source.feature_ids[entry]];
        }
        for (u32 entry = 0u; entry < candidate.nnz_count; ++entry) {
            ++candidate_degrees[candidate.feature_ids[entry]];
        }
        for (u32 feature = 0u; feature < source.feature_count; ++feature) {
            result.feature_degree_mismatches +=
                source_degrees[feature] == candidate_degrees[feature] ? 0u : 1u;
        }
        result.exact = result.source_nnz == result.candidate_nnz
            && result.row_degree_mismatches == 0u
            && result.feature_degree_mismatches == 0u;
        *out = result;
        return validation_ok();
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "degree conservation allocation failed");
    }
}

validation_result validate_degree_preserving_null_provenance(
    const csr_support_view &source,
    const csr_support_view &candidate,
    const degree_preserving_null_provenance &provenance) {
    if (provenance.schema_version != packing_validation_schema_version
        || provenance.algorithm_version != degree_preserving_null_algorithm_version) {
        return validation_error(validation_code::unsupported_version,
            provenance.algorithm_version, "unsupported degree-preserving null provenance");
    }
    degree_conservation_report conservation;
    validation_result status = validate_degree_conservation(source, candidate, &conservation);
    if (!status) return status;
    if (provenance.source_identity == 0u
        || provenance.source_support_identity != matrix_identity(source)
        || provenance.output_identity != matrix_identity(candidate)
        || provenance.row_count != source.row_count
        || provenance.feature_count != source.feature_count
        || provenance.nnz_count != source.nnz_count
        || provenance.attempted_swaps > provenance.maximum_attempts
        || provenance.accepted_swaps > provenance.attempted_swaps
        || provenance.accepted_swaps > provenance.requested_swaps
        || provenance.target_reached != (provenance.accepted_swaps == provenance.requested_swaps)
        || provenance.row_degrees_exact != (conservation.row_degree_mismatches == 0u)
        || provenance.feature_degrees_exact != (conservation.feature_degree_mismatches == 0u)
        || !conservation.exact) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "degree-preserving null provenance, identity, or conservation disagrees");
    }
    return validation_ok();
}

} // namespace cellpack
