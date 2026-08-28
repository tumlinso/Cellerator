#include "Cellerator/geometry/persistent_packing_payload.hh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

// This file deliberately keeps the versioned image layout, identity hash,
// semantic validation, and pointer relocation together. Splitting those
// tightly coupled rules would make it easier for a writer and reader to drift
// while the v1 persistence ABI is still being established.
namespace cellpack {
namespace {

constexpr unsigned char image_magic[8] = {'C','E','L','L','P','K','0','1'};
constexpr u32 image_endian_marker = 0x01020304u;
constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;

enum section_index : u32 {
    feature_permutation_section,
    inverse_feature_permutation_section,
    feature_block_offsets_section,
    feature_to_block_section,
    feature_to_local_section,
    row_group_offsets_section,
    row_permutation_section,
    inverse_row_permutation_section,
    tile_block_offsets_section,
    tile_block_ids_section,
    tile_block_cell_masks_section,
    block_row_entry_offsets_section,
    row_block_gene_masks_section,
    row_block_value_offsets_section,
    values_section,
    section_count
};

struct persistent_image_header {
    unsigned char magic[8];
    u32 schema_version;
    u32 header_bytes;
    u32 endian;
    u32 alignment;
    u64 image_bytes;
    u64 payload_identity;

    u32 semantic_plan_schema_version;
    u32 geometry_identity_version;
    u32 order_schema_version;
    u32 signature_algorithm_version;
    u32 tile_schema_version;
    u32 record_schema_version;
    u32 objective_kind;
    u32 row_domain_kind;

    u32 full_row_count;
    u32 row_count;
    u32 feature_count;
    u32 feature_block_count;
    u32 row_group_count;
    u32 maximum_feature_block_width;
    u32 row_group_width;
    u32 feature_axis_fingerprint_version;

    u64 feature_block_geometry_identity;
    u64 feature_axis_fingerprint;
    u64 row_domain_identity;
    u64 evaluation_source_identity;
    u64 sampling_provenance_identity;
    u64 cost_policy_identity;

    u32 order_kind;
    u32 order_window_size;
    u32 order_group_width;
    u32 reserved0;
    u64 order_seed;
    u64 ordering_identity;

    u64 tile_identity;
    u64 global_row_begin;
    u32 tile_row_width;
    u32 tile_count;
    u32 nnz_count;
    u32 tile_block_count;
    u32 row_block_entry_count;
    u32 value_size_bytes;
    u32 reserved1;
    u32 reserved2;

    u64 offsets[section_count];
};

static_assert(std::is_trivially_copyable<persistent_image_header>::value,
    "persistent image header must remain trivially copyable");

struct image_layout {
    u64 offsets[section_count]{};
    std::size_t total_bytes = 0u;
};

validation_result invalid(const char *message) {
    return validation_error(validation_code::invalid_matrix_view, invalid_id, message);
}

bool checked_add(std::size_t left, std::size_t right, std::size_t *out) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    *out = left + right;
    return true;
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t *out) noexcept {
    if (left != 0u && right > std::numeric_limits<std::size_t>::max() / left) return false;
    *out = left * right;
    return true;
}

bool align_cursor(std::size_t cursor, std::size_t *out) noexcept {
    const std::size_t mask = persistent_packing_payload_alignment - 1u;
    if (cursor > std::numeric_limits<std::size_t>::max() - mask) return false;
    *out = (cursor + mask) & ~mask;
    return true;
}

bool add_section(image_layout *layout, section_index section,
    std::size_t count, std::size_t element_bytes, std::size_t *cursor) noexcept {
    std::size_t aligned = 0u, bytes = 0u;
    if (!align_cursor(*cursor, &aligned)
        || !checked_multiply(count, element_bytes, &bytes)
        || !checked_add(aligned, bytes, cursor)) return false;
    layout->offsets[section] = aligned;
    return true;
}

bool make_layout(u32 rows, u32 features, u32 blocks, u32 groups, u32 tiles,
    u32 tile_blocks, u32 entries, u32 nnz, u32 value_bytes, image_layout *out) {
    if (out == nullptr) return false;
    image_layout layout;
    std::size_t cursor = sizeof(persistent_image_header);
    const bool ok = add_section(&layout, feature_permutation_section, features,
            sizeof(u32), &cursor)
        && add_section(&layout, inverse_feature_permutation_section, features,
            sizeof(u32), &cursor)
        && add_section(&layout, feature_block_offsets_section,
            static_cast<std::size_t>(blocks) + 1u, sizeof(u32), &cursor)
        && add_section(&layout, feature_to_block_section, features, sizeof(u32), &cursor)
        && add_section(&layout, feature_to_local_section, features, sizeof(u32), &cursor)
        && add_section(&layout, row_group_offsets_section,
            static_cast<std::size_t>(groups) + 1u, sizeof(u32), &cursor)
        && add_section(&layout, row_permutation_section, rows, sizeof(u32), &cursor)
        && add_section(&layout, inverse_row_permutation_section, rows, sizeof(u32), &cursor)
        && add_section(&layout, tile_block_offsets_section,
            static_cast<std::size_t>(tiles) + 1u, sizeof(u32), &cursor)
        && add_section(&layout, tile_block_ids_section, tile_blocks, sizeof(u32), &cursor)
        && add_section(&layout, tile_block_cell_masks_section, tile_blocks,
            sizeof(u32), &cursor)
        && add_section(&layout, block_row_entry_offsets_section,
            static_cast<std::size_t>(tile_blocks) + 1u, sizeof(u32), &cursor)
        && add_section(&layout, row_block_gene_masks_section, entries, sizeof(u32), &cursor)
        && add_section(&layout, row_block_value_offsets_section,
            static_cast<std::size_t>(entries) + 1u, sizeof(u32), &cursor)
        && add_section(&layout, values_section, nnz, value_bytes, &cursor);
    if (!ok || !align_cursor(cursor, &layout.total_bytes)) return false;
    *out = layout;
    return true;
}

u64 hash_bytes(u64 hash, const void *data, std::size_t bytes) noexcept {
    const auto *cursor = static_cast<const unsigned char *>(data);
    for (std::size_t index = 0u; index < bytes; ++index) {
        hash ^= cursor[index];
        hash *= fnv1a_prime;
    }
    return hash;
}

void hash_u32(u64 *hash, u32 value) noexcept {
    for (u32 byte = 0u; byte < 4u; ++byte) {
        const unsigned char part = static_cast<unsigned char>(value >> (byte * 8u));
        *hash ^= part;
        *hash *= fnv1a_prime;
    }
}

void hash_literal(u64 *hash, const char *value) noexcept {
    while (*value != '\0') {
        *hash ^= static_cast<unsigned char>(*value++);
        *hash *= fnv1a_prime;
    }
    *hash ^= 0u;
    *hash *= fnv1a_prime;
}

u64 geometry_identity(const feature_weighted_row_reduction_plan_view &plan) noexcept {
    u64 hash = fnv1a_offset;
    hash_literal(&hash, "cellerator_feature_block_geometry_identity_v1");
    hash_u32(&hash, feature_block_geometry_identity_version);
    hash_u32(&hash, packing_plan_semantic_schema_version);
    hash_u32(&hash, plan.feature_count);
    hash_u32(&hash, plan.feature_block_count);
    for (u32 block = 0u; block <= plan.feature_block_count; ++block)
        hash_u32(&hash, plan.feature_block_offsets[block]);
    for (u32 execution = 0u; execution < plan.feature_count; ++execution)
        hash_u32(&hash, plan.feature_permutation[execution]);
    return hash == 0u ? 1u : hash;
}

u64 image_identity(const void *image, std::size_t bytes) noexcept {
    persistent_image_header header;
    std::memcpy(&header, image, sizeof(header));
    header.payload_identity = 0u;
    u64 hash = hash_bytes(fnv1a_offset, &header, sizeof(header));
    const auto *base = static_cast<const unsigned char *>(image);
    hash = hash_bytes(hash, base + sizeof(header), bytes - sizeof(header));
    return hash == 0u ? 1u : hash;
}

template<typename T>
const T *pointer_at(const void *base, u64 offset) noexcept {
    return reinterpret_cast<const T *>(
        static_cast<const unsigned char *>(base) + offset);
}

void populate_view(const persistent_image_header &header, const void *base,
    persistent_packing_payload_view *out) {
    persistent_packing_payload_view result;
    result.payload_schema_version = header.schema_version;
    result.payload_kind = persistent_packing_payload_kind;
    result.payload_identity = header.payload_identity;
    result.image_base = base;
    result.image_bytes = static_cast<std::size_t>(header.image_bytes);
    result.plan_identity.feature_axis_fingerprint = header.feature_axis_fingerprint;
    result.plan_identity.feature_axis_fingerprint_version =
        header.feature_axis_fingerprint_version;
    result.plan_identity.row_domain_kind =
        static_cast<packing_row_domain_kind>(header.row_domain_kind);
    result.plan_identity.row_domain_identity = header.row_domain_identity;
    result.plan_identity.evaluation_source_identity = header.evaluation_source_identity;
    result.plan_identity.sampling_provenance_identity =
        header.sampling_provenance_identity;
    result.objective_kind = static_cast<packing_exact_objective_kind>(
        header.objective_kind);
    result.cost_policy_identity = header.cost_policy_identity;
    result.maximum_feature_block_width = header.maximum_feature_block_width;
    result.row_group_width = header.row_group_width;
    result.inverse_feature_permutation = pointer_at<u32>(base,
        header.offsets[inverse_feature_permutation_section]);
    result.feature_to_block = pointer_at<u32>(base,
        header.offsets[feature_to_block_section]);
    result.feature_to_local = pointer_at<u32>(base,
        header.offsets[feature_to_local_section]);
    result.row_group_count = header.row_group_count;
    result.row_group_offsets = pointer_at<u32>(base,
        header.offsets[row_group_offsets_section]);

    result.plan.semantic_plan_schema_version = header.semantic_plan_schema_version;
    result.plan.geometry_identity_version = header.geometry_identity_version;
    result.plan.feature_count = header.feature_count;
    result.plan.feature_block_count = header.feature_block_count;
    result.plan.feature_block_geometry_identity = header.feature_block_geometry_identity;
    result.plan.feature_block_offsets = pointer_at<u32>(base,
        header.offsets[feature_block_offsets_section]);
    result.plan.feature_permutation = pointer_at<u32>(base,
        header.offsets[feature_permutation_section]);

    result.order.order_schema_version = header.order_schema_version;
    result.order.signature_algorithm_version = header.signature_algorithm_version;
    result.order.kind = static_cast<local_cell_order_kind>(header.order_kind);
    result.order.window_size = header.order_window_size;
    result.order.group_width = header.order_group_width;
    result.order.seed = header.order_seed;
    result.order.ordering_identity = header.ordering_identity;
    result.order.global_row_begin = header.global_row_begin;
    result.order.full_row_count = header.full_row_count;
    result.order.row_count = header.row_count;
    result.order.feature_block_count = header.feature_block_count;
    result.order.feature_block_geometry_identity = header.feature_block_geometry_identity;
    result.order.row_domain_identity = header.row_domain_identity;
    result.order.row_permutation = pointer_at<u32>(base,
        header.offsets[row_permutation_section]);
    result.order.inverse_row_permutation = pointer_at<u32>(base,
        header.offsets[inverse_row_permutation_section]);

    result.tiles.tile_schema_version = header.tile_schema_version;
    result.tiles.record_schema_version = header.record_schema_version;
    result.tiles.semantic_plan_schema_version = header.semantic_plan_schema_version;
    result.tiles.geometry_identity_version = header.geometry_identity_version;
    result.tiles.order_schema_version = header.order_schema_version;
    result.tiles.tile_identity = header.tile_identity;
    result.tiles.feature_block_geometry_identity = header.feature_block_geometry_identity;
    result.tiles.ordering_identity = header.ordering_identity;
    result.tiles.global_row_begin = header.global_row_begin;
    result.tiles.full_row_count = header.full_row_count;
    result.tiles.row_count = header.row_count;
    result.tiles.feature_count = header.feature_count;
    result.tiles.feature_block_count = header.feature_block_count;
    result.tiles.tile_row_width = header.tile_row_width;
    result.tiles.tile_count = header.tile_count;
    result.tiles.nnz_count = header.nnz_count;
    result.tiles.tile_block_count = header.tile_block_count;
    result.tiles.row_block_entry_count = header.row_block_entry_count;
    result.tiles.value_size_bytes = header.value_size_bytes;
    result.tiles.feature_axis_fingerprint = header.feature_axis_fingerprint;
    result.tiles.feature_axis_fingerprint_version =
        header.feature_axis_fingerprint_version;
    result.tiles.row_domain_identity = header.row_domain_identity;
    result.tiles.tile_block_offsets = pointer_at<u32>(base,
        header.offsets[tile_block_offsets_section]);
    result.tiles.tile_block_ids = pointer_at<u32>(base,
        header.offsets[tile_block_ids_section]);
    result.tiles.tile_block_cell_masks = pointer_at<u32>(base,
        header.offsets[tile_block_cell_masks_section]);
    result.tiles.block_row_entry_offsets = pointer_at<u32>(base,
        header.offsets[block_row_entry_offsets_section]);
    result.tiles.row_block_gene_masks = pointer_at<u32>(base,
        header.offsets[row_block_gene_masks_section]);
    result.tiles.row_block_value_offsets = pointer_at<u32>(base,
        header.offsets[row_block_value_offsets_section]);
    result.tiles.values = pointer_at<unsigned char>(base,
        header.offsets[values_section]);
    *out = result;
}

validation_result validate_plan_arrays(const persistent_packing_payload_view &view) {
    const u32 features = view.plan.feature_count;
    const u32 blocks = view.plan.feature_block_count;
    if (view.plan.feature_block_offsets[0] != 0u
        || view.plan.feature_block_offsets[blocks] != features
        || view.row_group_offsets[0] != 0u
        || view.row_group_offsets[view.row_group_count] != view.tiles.full_row_count
        || geometry_identity(view.plan) != view.plan.feature_block_geometry_identity) {
        return invalid("persistent plan boundaries are invalid");
    }
    for (u32 execution = 0u; execution < features; ++execution) {
        const u32 canonical = view.plan.feature_permutation[execution];
        if (canonical >= features || view.inverse_feature_permutation[canonical] != execution) {
            return invalid("persistent feature permutations do not round trip");
        }
    }
    for (u32 block = 0u; block < blocks; ++block) {
        const u32 begin = view.plan.feature_block_offsets[block];
        const u32 end = view.plan.feature_block_offsets[block + 1u];
        if (end <= begin || end - begin > view.maximum_feature_block_width) {
            return invalid("persistent feature-block width is invalid");
        }
        for (u32 execution = begin; execution < end; ++execution) {
            const u32 canonical = view.plan.feature_permutation[execution];
            if (view.feature_to_block[canonical] != block
                || view.feature_to_local[canonical] != execution - begin) {
                return invalid("persistent canonical feature maps disagree");
            }
        }
    }
    for (u32 group = 0u; group < view.row_group_count; ++group) {
        const u32 begin = view.row_group_offsets[group];
        const u32 end = view.row_group_offsets[group + 1u];
        if (end <= begin || end - begin > view.row_group_width
            || (group + 1u < view.row_group_count
                && end - begin != view.row_group_width)) {
            return invalid("persistent row-group geometry is invalid");
        }
    }
    for (u32 execution = 0u; execution < view.order.row_count; ++execution) {
        const u32 canonical = view.order.row_permutation[execution];
        if (canonical >= view.order.row_count
            || view.order.inverse_row_permutation[canonical] != execution) {
            return invalid("persistent row permutations do not round trip");
        }
    }
    return validation_ok();
}

u32 valid_lane_mask(u32 lanes) noexcept {
    return lanes >= 32u ? 0xffffffffu : ((u32{1u} << lanes) - 1u);
}

validation_result validate_tile_arrays(const persistent_packing_payload_view &view) {
    const warp_tile_view &tiles = view.tiles;
    if (tiles.tile_row_width == 0u || tiles.tile_row_width > 32u
        || tiles.tile_count != tiles.row_count / tiles.tile_row_width
            + (tiles.row_count % tiles.tile_row_width != 0u ? 1u : 0u)
        || tiles.tile_block_offsets[0] != 0u
        || tiles.tile_block_offsets[tiles.tile_count] != tiles.tile_block_count
        || tiles.block_row_entry_offsets[0] != 0u
        || tiles.block_row_entry_offsets[tiles.tile_block_count]
            != tiles.row_block_entry_count
        || tiles.row_block_value_offsets[0] != 0u
        || tiles.row_block_value_offsets[tiles.row_block_entry_count]
            != tiles.nnz_count) {
        return invalid("persistent tile terminal geometry is invalid");
    }
    for (u32 tile = 0u; tile < tiles.tile_count; ++tile) {
        const u32 begin = tiles.tile_block_offsets[tile];
        const u32 end = tiles.tile_block_offsets[tile + 1u];
        const u32 rows_remaining = tiles.row_count - tile * tiles.tile_row_width;
        const u32 lanes = std::min(tiles.tile_row_width, rows_remaining);
        u32 previous_block = 0u;
        for (u32 descriptor = begin; descriptor < end; ++descriptor) {
            const u32 block = tiles.tile_block_ids[descriptor];
            const u32 cell_mask = tiles.tile_block_cell_masks[descriptor];
            const u32 entry_begin = tiles.block_row_entry_offsets[descriptor];
            const u32 entry_end = tiles.block_row_entry_offsets[descriptor + 1u];
            if (block >= tiles.feature_block_count
                || (descriptor != begin && block <= previous_block)
                || cell_mask == 0u || (cell_mask & ~valid_lane_mask(lanes)) != 0u
                || entry_end < entry_begin
                || entry_end - entry_begin != static_cast<u32>(__builtin_popcount(cell_mask))) {
                return invalid("persistent tile descriptor is invalid");
            }
            const u32 block_width = view.plan.feature_block_offsets[block + 1u]
                - view.plan.feature_block_offsets[block];
            for (u32 entry = entry_begin; entry < entry_end; ++entry) {
                const u32 gene_mask = tiles.row_block_gene_masks[entry];
                const u32 value_begin = tiles.row_block_value_offsets[entry];
                const u32 value_end = tiles.row_block_value_offsets[entry + 1u];
                if (gene_mask == 0u
                    || (gene_mask & ~valid_lane_mask(block_width)) != 0u
                    || value_end < value_begin
                    || value_end - value_begin
                        != static_cast<u32>(__builtin_popcount(gene_mask))) {
                    return invalid("persistent tile row-block entry is invalid");
                }
            }
            previous_block = block;
        }
    }
    return validation_ok();
}

u64 splitmix64(u64 value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

u64 reduction_identity(const feature_weighted_row_reduction_view &input) noexcept {
    u64 identity = splitmix64(input.tiles.tile_identity);
    identity = splitmix64(identity ^ input.plan.feature_block_geometry_identity);
    identity = splitmix64(identity ^ input.feature_weight_identity);
    identity = splitmix64(identity
        ^ (static_cast<u64>(feature_weighted_row_reduction_schema_version) << 32u)
        ^ static_cast<u32>(cellerator::real::code_of<cellerator::real::storage_t>::code));
    identity = splitmix64(identity
        ^ (static_cast<u64>(cellerator::real::code_of<cellerator::real::compute_t>::code) << 32u)
        ^ static_cast<u32>(cellerator::real::code_of<cellerator::real::accum_t>::code));
    return identity == 0u ? 1u : identity;
}

} // namespace

validation_result query_persistent_packing_payload_requirements_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    persistent_packing_payload_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "persistent payload requirements output is null");
    }
    validation_result status = plan.validate();
    if (!status) return status;
    status = validate_cell_block_record_view_host(plan, records);
    if (!status) return status;
    status = validate_local_cell_order_view_host(records, order);
    if (!status) return status;
    status = validate_warp_tile_view_host(plan, records, order, tiles);
    if (!status) return status;
    image_layout layout;
    if (!make_layout(tiles.row_count, tiles.feature_count, tiles.feature_block_count,
        plan.row_group_count(), tiles.tile_count, tiles.tile_block_count,
        tiles.row_block_entry_count, tiles.nnz_count, tiles.value_size_bytes, &layout)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "persistent payload byte requirements overflow");
    }
    out->image_bytes = layout.total_bytes;
    return validation_ok();
}

validation_result build_persistent_packing_payload_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    const persistent_packing_payload_buffers &buffers,
    persistent_packing_payload_view *out) {
    if (out == nullptr || buffers.image == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "persistent payload output or image is null");
    }
    persistent_packing_payload_requirements required;
    validation_result status = query_persistent_packing_payload_requirements_host(
        plan, records, order, tiles, &required);
    if (!status) return status;
    if (buffers.image_capacity_bytes < required.image_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "persistent payload image buffer is too small");
    }
    image_layout layout;
    if (!make_layout(tiles.row_count, tiles.feature_count, tiles.feature_block_count,
        plan.row_group_count(), tiles.tile_count, tiles.tile_block_count,
        tiles.row_block_entry_count, tiles.nnz_count, tiles.value_size_bytes, &layout)) {
        return invalid("persistent payload layout failed after preflight");
    }
    std::memset(buffers.image, 0, required.image_bytes);
    persistent_image_header header{};
    std::memcpy(header.magic, image_magic, sizeof(image_magic));
    header.schema_version = persistent_packing_payload_schema_version;
    header.header_bytes = sizeof(header);
    header.endian = image_endian_marker;
    header.alignment = persistent_packing_payload_alignment;
    header.image_bytes = required.image_bytes;
    header.semantic_plan_schema_version = plan.semantic_schema_version();
    header.geometry_identity_version = feature_block_geometry_identity_version;
    header.order_schema_version = order.order_schema_version;
    header.signature_algorithm_version = order.signature_algorithm_version;
    header.tile_schema_version = tiles.tile_schema_version;
    header.record_schema_version = tiles.record_schema_version;
    header.objective_kind = static_cast<u32>(plan.objective_kind());
    header.row_domain_kind = static_cast<u32>(plan.identity().row_domain_kind);
    header.full_row_count = tiles.full_row_count;
    header.row_count = tiles.row_count;
    header.feature_count = tiles.feature_count;
    header.feature_block_count = tiles.feature_block_count;
    header.row_group_count = plan.row_group_count();
    header.maximum_feature_block_width = plan.maximum_feature_block_width();
    header.row_group_width = plan.row_group_width();
    header.feature_axis_fingerprint_version = tiles.feature_axis_fingerprint_version;
    header.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    header.feature_axis_fingerprint = tiles.feature_axis_fingerprint;
    header.row_domain_identity = tiles.row_domain_identity;
    header.evaluation_source_identity = plan.identity().evaluation_source_identity;
    header.sampling_provenance_identity = plan.identity().sampling_provenance_identity;
    header.cost_policy_identity = plan.cost_policy_identity();
    header.order_kind = static_cast<u32>(order.kind);
    header.order_window_size = order.window_size;
    header.order_group_width = order.group_width;
    header.order_seed = order.seed;
    header.ordering_identity = order.ordering_identity;
    header.tile_identity = tiles.tile_identity;
    header.global_row_begin = tiles.global_row_begin;
    header.tile_row_width = tiles.tile_row_width;
    header.tile_count = tiles.tile_count;
    header.nnz_count = tiles.nnz_count;
    header.tile_block_count = tiles.tile_block_count;
    header.row_block_entry_count = tiles.row_block_entry_count;
    header.value_size_bytes = tiles.value_size_bytes;
    std::copy(layout.offsets, layout.offsets + section_count, header.offsets);
    std::memcpy(buffers.image, &header, sizeof(header));

    auto *base = static_cast<unsigned char *>(buffers.image);
    auto copy = [&](section_index section, const void *source, std::size_t bytes) {
        if (bytes != 0u) std::memcpy(base + layout.offsets[section], source, bytes);
    };
    copy(feature_permutation_section, plan.feature_permutation(),
        static_cast<std::size_t>(tiles.feature_count) * sizeof(u32));
    copy(inverse_feature_permutation_section, plan.inverse_feature_permutation(),
        static_cast<std::size_t>(tiles.feature_count) * sizeof(u32));
    copy(feature_block_offsets_section, plan.feature_block_offsets(),
        (static_cast<std::size_t>(tiles.feature_block_count) + 1u) * sizeof(u32));
    copy(feature_to_block_section, plan.feature_to_block(),
        static_cast<std::size_t>(tiles.feature_count) * sizeof(u32));
    copy(feature_to_local_section, plan.feature_to_local(),
        static_cast<std::size_t>(tiles.feature_count) * sizeof(u32));
    copy(row_group_offsets_section, plan.row_group_offsets(),
        (static_cast<std::size_t>(plan.row_group_count()) + 1u) * sizeof(u32));
    copy(row_permutation_section, order.row_permutation,
        static_cast<std::size_t>(tiles.row_count) * sizeof(u32));
    copy(inverse_row_permutation_section, order.inverse_row_permutation,
        static_cast<std::size_t>(tiles.row_count) * sizeof(u32));
    copy(tile_block_offsets_section, tiles.tile_block_offsets,
        (static_cast<std::size_t>(tiles.tile_count) + 1u) * sizeof(u32));
    copy(tile_block_ids_section, tiles.tile_block_ids,
        static_cast<std::size_t>(tiles.tile_block_count) * sizeof(u32));
    copy(tile_block_cell_masks_section, tiles.tile_block_cell_masks,
        static_cast<std::size_t>(tiles.tile_block_count) * sizeof(u32));
    copy(block_row_entry_offsets_section, tiles.block_row_entry_offsets,
        (static_cast<std::size_t>(tiles.tile_block_count) + 1u) * sizeof(u32));
    copy(row_block_gene_masks_section, tiles.row_block_gene_masks,
        static_cast<std::size_t>(tiles.row_block_entry_count) * sizeof(u32));
    copy(row_block_value_offsets_section, tiles.row_block_value_offsets,
        (static_cast<std::size_t>(tiles.row_block_entry_count) + 1u) * sizeof(u32));
    copy(values_section, tiles.values,
        static_cast<std::size_t>(tiles.nnz_count) * tiles.value_size_bytes);
    header.payload_identity = image_identity(buffers.image, required.image_bytes);
    std::memcpy(buffers.image, &header, sizeof(header));
    const persistent_packing_payload_compatibility expected{
        tiles.global_row_begin, tiles.row_count, tiles.feature_count,
        tiles.feature_axis_fingerprint, tiles.feature_axis_fingerprint_version,
        tiles.row_domain_identity, header.payload_identity};
    return validate_persistent_packing_payload_host(
        buffers.image, required.image_bytes, expected, out);
}

validation_result validate_persistent_packing_payload_host(
    const void *image,
    std::size_t image_bytes,
    const persistent_packing_payload_compatibility &expected,
    persistent_packing_payload_view *out) {
    if (image == nullptr || out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "persistent payload image or output is null");
    }
    if (image_bytes < sizeof(persistent_image_header)) return invalid("persistent image is truncated");
    persistent_image_header header;
    std::memcpy(&header, image, sizeof(header));
    if (std::memcmp(header.magic, image_magic, sizeof(image_magic)) != 0
        || header.schema_version != persistent_packing_payload_schema_version
        || header.header_bytes != sizeof(header) || header.endian != image_endian_marker
        || header.alignment != persistent_packing_payload_alignment
        || header.image_bytes != image_bytes || header.payload_identity == 0u
        || header.payload_identity != image_identity(image, image_bytes)) {
        return invalid("persistent image header or identity is invalid");
    }
    if (header.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || header.geometry_identity_version != feature_block_geometry_identity_version
        || header.order_schema_version != local_cell_order_schema_version
        || header.signature_algorithm_version != local_cell_signature_algorithm_version
        || header.tile_schema_version != warp_tile_schema_version
        || header.record_schema_version != cell_block_record_schema_version
        || header.feature_block_geometry_identity == 0u
        || header.ordering_identity == 0u || header.tile_identity == 0u
        || header.cost_policy_identity == 0u || header.full_row_count == 0u
        || header.row_count == 0u
        || header.feature_count == 0u || header.feature_block_count == 0u
        || header.row_group_count == 0u || header.maximum_feature_block_width == 0u
        || header.row_group_width == 0u || header.feature_axis_fingerprint == 0u
        || header.feature_axis_fingerprint_version == 0u
        || header.row_domain_identity == 0u
        || header.value_size_bytes != sizeof(cellerator::real::storage_t)
        || header.global_row_begin > header.full_row_count
        || header.row_count > header.full_row_count - header.global_row_begin
        || header.objective_kind < static_cast<u32>(packing_exact_objective_kind::total_bytes)
        || header.objective_kind > static_cast<u32>(packing_exact_objective_kind::weighted_score)
        || (header.row_domain_kind
                != static_cast<u32>(packing_row_domain_kind::full_dataset_identity)
            && header.row_domain_kind
                != static_cast<u32>(packing_row_domain_kind::sampled_rows_identity))
        || header.order_kind < static_cast<u32>(local_cell_order_kind::inferred_minhash)
        || header.order_kind > static_cast<u32>(local_cell_order_kind::row_nnz_descending)
        || header.reserved0 != 0u || header.reserved1 != 0u || header.reserved2 != 0u) {
        return invalid("persistent image semantic metadata is invalid");
    }
    image_layout layout;
    if (!make_layout(header.row_count, header.feature_count,
        header.feature_block_count, header.row_group_count, header.tile_count,
        header.tile_block_count, header.row_block_entry_count, header.nnz_count,
        header.value_size_bytes, &layout) || layout.total_bytes != image_bytes
        || !std::equal(layout.offsets, layout.offsets + section_count, header.offsets)) {
        return invalid("persistent image section offsets are invalid");
    }
    if (expected.global_row_begin != header.global_row_begin
        || expected.row_count != header.row_count
        || expected.feature_count != header.feature_count
        || expected.feature_axis_fingerprint != header.feature_axis_fingerprint
        || expected.feature_axis_fingerprint_version
            != header.feature_axis_fingerprint_version
        || expected.row_domain_identity != header.row_domain_identity
        || expected.payload_identity != header.payload_identity) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "persistent image compatibility identity mismatches");
    }
    persistent_packing_payload_view result;
    populate_view(header, image, &result);
    validation_result status = validate_plan_arrays(result);
    if (!status) return status;
    status = validate_tile_arrays(result);
    if (!status) return status;
    if (result.plan.feature_count != result.tiles.feature_count
        || result.plan.feature_block_count != result.tiles.feature_block_count
        || result.plan.feature_block_geometry_identity
            != result.tiles.feature_block_geometry_identity
        || result.order.ordering_identity != result.tiles.ordering_identity
        || result.order.group_width != result.tiles.tile_row_width
        || result.order.row_count != result.tiles.row_count
        || result.order.global_row_begin != result.tiles.global_row_begin
        || result.order.row_domain_identity != result.tiles.row_domain_identity) {
        return invalid("persistent plan/order/tile domains disagree");
    }
    *out = result;
    return validation_ok();
}

validation_result rebind_persistent_packing_payload(
    const persistent_packing_payload_view &validated_host_view,
    const void *new_image_base,
    std::size_t new_image_bytes,
    persistent_packing_payload_view *out) {
    if (validated_host_view.image_base == nullptr || new_image_base == nullptr || out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "persistent payload rebind pointer is null");
    }
    if (new_image_bytes != validated_host_view.image_bytes) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "persistent payload rebind size differs");
    }
    const auto *old_base = static_cast<const unsigned char *>(validated_host_view.image_base);
    const auto *new_base = static_cast<const unsigned char *>(new_image_base);
    auto rebase = [&](const void *pointer) -> const void * {
        if (pointer == nullptr) return nullptr;
        const auto offset = static_cast<const unsigned char *>(pointer) - old_base;
        return new_base + offset;
    };
    persistent_packing_payload_view result = validated_host_view;
    result.image_base = new_image_base;
    result.inverse_feature_permutation = static_cast<const u32 *>(
        rebase(result.inverse_feature_permutation));
    result.feature_to_block = static_cast<const u32 *>(rebase(result.feature_to_block));
    result.feature_to_local = static_cast<const u32 *>(rebase(result.feature_to_local));
    result.row_group_offsets = static_cast<const u32 *>(rebase(result.row_group_offsets));
    result.plan.feature_block_offsets = static_cast<const u32 *>(
        rebase(result.plan.feature_block_offsets));
    result.plan.feature_permutation = static_cast<const u32 *>(
        rebase(result.plan.feature_permutation));
    result.order.row_permutation = static_cast<const u32 *>(rebase(result.order.row_permutation));
    result.order.inverse_row_permutation = static_cast<const u32 *>(
        rebase(result.order.inverse_row_permutation));
    result.tiles.tile_block_offsets = static_cast<const u32 *>(
        rebase(result.tiles.tile_block_offsets));
    result.tiles.tile_block_ids = static_cast<const u32 *>(rebase(result.tiles.tile_block_ids));
    result.tiles.tile_block_cell_masks = static_cast<const u32 *>(
        rebase(result.tiles.tile_block_cell_masks));
    result.tiles.block_row_entry_offsets = static_cast<const u32 *>(
        rebase(result.tiles.block_row_entry_offsets));
    result.tiles.row_block_gene_masks = static_cast<const u32 *>(
        rebase(result.tiles.row_block_gene_masks));
    result.tiles.row_block_value_offsets = static_cast<const u32 *>(
        rebase(result.tiles.row_block_value_offsets));
    result.tiles.values = rebase(result.tiles.values);
    *out = result;
    return validation_ok();
}

feature_weighted_row_reduction_view
make_persistent_feature_weighted_row_reduction_view(
    const persistent_packing_payload_view &payload,
    u64 feature_weight_identity,
    std::size_t feature_weight_capacity,
    const cellerator::real::compute_t *feature_weights) noexcept {
    feature_weighted_row_reduction_view result;
    result.schema_version = feature_weighted_row_reduction_schema_version;
    result.storage_type_code = static_cast<u32>(
        cellerator::real::code_of<cellerator::real::storage_t>::code);
    result.weight_type_code = static_cast<u32>(
        cellerator::real::code_of<cellerator::real::compute_t>::code);
    result.accumulation_type_code = static_cast<u32>(
        cellerator::real::code_of<cellerator::real::accum_t>::code);
    result.feature_weight_identity = feature_weight_identity;
    result.plan = payload.plan;
    result.tiles = payload.tiles;
    result.feature_weight_capacity = feature_weight_capacity;
    result.feature_weights = feature_weights;
    result.reduction_identity = reduction_identity(result);
    return result;
}

} // namespace cellpack
