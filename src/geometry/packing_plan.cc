#include "Cellerator/geometry/packing_plan.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

namespace cellpack {
namespace {

constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;

void hash_byte(u64 *hash, unsigned char value) noexcept {
    *hash ^= value;
    *hash *= fnv1a_prime;
}

void hash_u32(u64 *hash, u32 value) noexcept {
    for (u32 byte = 0u; byte < 4u; ++byte) {
        hash_byte(hash, static_cast<unsigned char>(value >> (byte * 8u)));
    }
}

void hash_literal(u64 *hash, const char *value) noexcept {
    while (*value != '\0') hash_byte(hash, static_cast<unsigned char>(*value++));
    hash_byte(hash, 0u);
}

u64 compute_feature_block_geometry_identity(
    u32 feature_count,
    const u32 *feature_permutation,
    u32 feature_block_count,
    const u32 *feature_block_offsets) noexcept {
    u64 hash = fnv1a_offset;
    hash_literal(&hash, "cellerator_feature_block_geometry_identity_v1");
    hash_u32(&hash, feature_block_geometry_identity_version);
    hash_u32(&hash, packing_plan_semantic_schema_version);
    hash_u32(&hash, feature_count);
    hash_u32(&hash, feature_block_count);
    for (u32 block = 0u; block <= feature_block_count; ++block) {
        hash_u32(&hash, feature_block_offsets[block]);
    }
    for (u32 execution = 0u; execution < feature_count; ++execution) {
        hash_u32(&hash, feature_permutation[execution]);
    }
    return hash == 0u ? 1u : hash;
}

bool checked_u32_bytes(std::size_t count, std::size_t *out) noexcept {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(u32)) return false;
    *out = count * sizeof(u32);
    return true;
}

bool append_aligned_section(
    std::size_t bytes,
    std::size_t *cursor,
    cellerator::memory::rel32 *offset) noexcept {
    const std::size_t mask = packing_plan_image_alignment - 1u;
    if (*cursor > std::numeric_limits<std::size_t>::max() - mask) return false;
    const std::size_t aligned = (*cursor + mask) & ~mask;
    if (aligned > std::numeric_limits<u32>::max()
        || bytes > std::numeric_limits<std::size_t>::max() - aligned) return false;
    offset->byte_offset = static_cast<u32>(aligned);
    *cursor = aligned + bytes;
    return true;
}

validation_result bind_image_view(
    const packing_plan_image_header &header,
    const void *base,
    std::size_t bytes,
    packing_plan_image_view *out) noexcept {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "packing plan image view output is null");
    }
    auto resolve = [&](cellerator::memory::rel32 offset, std::size_t count, const u32 **target) {
        std::size_t span_bytes = 0u;
        if (!checked_u32_bytes(count, &span_bytes)
            || offset.byte_offset > bytes || span_bytes > bytes - offset.byte_offset
            || (offset.byte_offset % alignof(u32)) != 0u) return false;
        *target = reinterpret_cast<const u32 *>(
            static_cast<const unsigned char *>(base) + offset.byte_offset);
        return true;
    };
    packing_plan_image_view result;
    result.header = header;
    result.image_base = base;
    result.image_bytes = bytes;
    if (!resolve(header.feature_permutation, header.feature_count, &result.feature_permutation)
        || !resolve(header.inverse_feature_permutation, header.feature_count, &result.inverse_feature_permutation)
        || !resolve(header.feature_block_offsets,
            static_cast<std::size_t>(header.feature_block_count) + 1u, &result.feature_block_offsets)
        || !resolve(header.feature_to_block, header.feature_count, &result.feature_to_block)
        || !resolve(header.feature_to_local, header.feature_count, &result.feature_to_local)
        || !resolve(header.row_group_offsets,
            static_cast<std::size_t>(header.row_group_count) + 1u, &result.row_group_offsets)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "packing plan image section is out of bounds or misaligned");
    }
    *out = result;
    return validation_ok();
}

validation_result validate_build_view(const frozen_packing_plan_build_view &source) {
    if (source.row_count == 0u || source.feature_count == 0u
        || source.maximum_feature_block_width == 0u || source.row_group_width == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "frozen plan dimensions and configured widths must be nonzero");
    }
    if (source.feature_permutation == nullptr || source.inverse_feature_permutation == nullptr
        || source.feature_block_offsets == nullptr || source.feature_to_block == nullptr
        || source.feature_to_local == nullptr || source.row_group_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "frozen plan build arrays are null");
    }
    packing_plan_view plan;
    plan.row_count = source.row_count;
    plan.feature_count = source.feature_count;
    plan.feature_permutation = source.feature_permutation;
    plan.inverse_feature_permutation = source.inverse_feature_permutation;
    plan.feature_block_count = source.feature_block_count;
    plan.feature_block_offsets = source.feature_block_offsets;
    plan.row_group_count = source.row_group_count;
    plan.row_group_offsets = source.row_group_offsets;
    const validation_result plan_status = validate_packing_plan_view(plan);
    if (!plan_status) return plan_status;
    for (u32 block = 0u; block < source.feature_block_count; ++block) {
        const u32 width = source.feature_block_offsets[block + 1u] - source.feature_block_offsets[block];
        if (width > source.maximum_feature_block_width) {
            return validation_error(validation_code::invalid_plan_geometry, block, "frozen feature block exceeds configured maximum width");
        }
        for (u32 local = 0u; local < width; ++local) {
            const u32 canonical = source.feature_permutation[source.feature_block_offsets[block] + local];
            if (source.feature_to_block[canonical] != block || source.feature_to_local[canonical] != local) {
                return validation_error(validation_code::invalid_plan_geometry, canonical, "frozen feature block/local lookup disagrees with authoritative geometry");
            }
        }
    }
    if (source.identity.row_domain_kind == packing_row_domain_kind::unknown) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "frozen plan row domain kind is unknown");
    }
    if (source.identity.feature_axis_fingerprint == 0u
        || source.identity.feature_axis_fingerprint_version == 0u
        || source.identity.row_domain_identity == 0u
        || source.identity.evaluation_source_identity == 0u
        || (source.identity.row_domain_kind == packing_row_domain_kind::sampled_rows_identity
            && source.identity.sampling_provenance_identity == 0u)
        || source.cost_policy_identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "frozen plan compatibility/provenance identities must be explicit");
    }
    if (source.objective_kind != packing_exact_objective_kind::total_bytes
        && source.objective_kind != packing_exact_objective_kind::row_active_block_references
        && source.objective_kind != packing_exact_objective_kind::weighted_score) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "frozen plan exact objective kind is unsupported");
    }
    for (u32 group = 0u; group < source.row_group_count; ++group) {
        const u32 width = source.row_group_offsets[group + 1u] - source.row_group_offsets[group];
        if (width == 0u || width > source.row_group_width
            || (group + 1u < source.row_group_count && width != source.row_group_width)) {
            return validation_error(validation_code::invalid_plan_geometry, group, "frozen row groups do not match configured fixed width");
        }
    }
    if (!std::isfinite(source.baseline.objective) || !std::isfinite(source.final.objective)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "frozen plan exact objective is not finite");
    }
    return validation_ok();
}

} // namespace

void packing_plan_image_deleter::operator()(unsigned char *pointer) const noexcept {
    if (pointer != nullptr) {
        ::operator delete(pointer, std::align_val_t{packing_plan_image_alignment});
    }
}

validation_result validate_packing_plan_image_host(
    const void *image,
    std::size_t image_bytes,
    packing_plan_image_view *out) noexcept {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "packing plan image output is null");
    }
    *out = packing_plan_image_view{};
    if (image == nullptr || image_bytes < sizeof(packing_plan_image_header)
        || (reinterpret_cast<std::uintptr_t>(image) & (packing_plan_image_alignment - 1u)) != 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "packing plan image base is null, short, or misaligned");
    }
    const auto *header = static_cast<const packing_plan_image_header *>(image);
    if (header->common.magic != packing_plan_image_magic
        || header->common.schema_version != packing_plan_semantic_schema_version
        || header->common.required_alignment != packing_plan_image_alignment
        || header->common.section_count != 6u
        || header->common.total_bytes != image_bytes
        || header->common.identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "packing plan image header is incompatible");
    }
    packing_plan_image_view result;
    validation_result status = bind_image_view(*header, image, image_bytes, &result);
    if (!status) return status;
    frozen_packing_plan_build_view source;
    source.row_count = header->row_count;
    source.feature_count = header->feature_count;
    source.feature_permutation = result.feature_permutation;
    source.inverse_feature_permutation = result.inverse_feature_permutation;
    source.feature_block_count = header->feature_block_count;
    source.feature_block_offsets = result.feature_block_offsets;
    source.feature_to_block = result.feature_to_block;
    source.feature_to_local = result.feature_to_local;
    source.row_group_count = header->row_group_count;
    source.row_group_offsets = result.row_group_offsets;
    source.maximum_feature_block_width = header->maximum_feature_block_width;
    source.row_group_width = header->row_group_width;
    source.identity = header->identity;
    source.objective_kind = header->objective_kind;
    source.cost_policy_identity = header->cost_policy_identity;
    source.baseline = header->baseline;
    source.final = header->final;
    status = validate_build_view(source);
    if (!status) return status;
    const u64 geometry_identity = compute_feature_block_geometry_identity(
        source.feature_count, source.feature_permutation,
        source.feature_block_count, source.feature_block_offsets);
    if (geometry_identity != header->common.identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "packing plan image identity is inconsistent with canonical geometry");
    }
    *out = result;
    return validation_ok();
}

validation_result rebind_packing_plan_image(
    const packing_plan_image_view &validated_host_view,
    const void *new_image_base,
    std::size_t new_image_bytes,
    packing_plan_image_view *out) noexcept {
    if (new_image_base == nullptr || out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "packing plan rebind base or output is null");
    }
    if (validated_host_view.image_base == nullptr
        || validated_host_view.header.common.magic != packing_plan_image_magic
        || validated_host_view.header.common.identity == 0u
        || new_image_bytes != validated_host_view.image_bytes) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "packing plan rebind source or destination size is invalid");
    }
    return bind_image_view(
        validated_host_view.header, new_image_base, new_image_bytes, out);
}

packing_plan_view frozen_packing_plan::view() const noexcept {
    packing_plan_view result;
    result.row_count = row_count_;
    result.feature_count = feature_count_;
    result.feature_permutation = feature_permutation();
    result.inverse_feature_permutation = inverse_feature_permutation();
    result.row_group_count = row_group_count_;
    result.row_group_offsets = row_group_offsets();
    result.feature_block_count = feature_block_count_;
    result.feature_block_offsets = feature_block_offsets();
    return result;
}

validation_result frozen_packing_plan::validate() const {
    packing_plan_image_view rebound;
    const validation_result image_status = validate_packing_plan_image_host(
        image_storage_.get(), image_allocation_.bytes, &rebound);
    if (!image_status) return image_status;
    if (rebound.header.common.identity != feature_block_geometry_identity_
        || rebound.header.row_count != row_count_
        || rebound.header.feature_count != feature_count_) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "frozen plan metadata disagrees with its image header");
    }
    frozen_packing_plan_build_view source;
    source.row_count = row_count_;
    source.feature_count = feature_count_;
    source.feature_permutation = feature_permutation();
    source.inverse_feature_permutation = inverse_feature_permutation();
    source.feature_block_count = feature_block_count_;
    source.feature_block_offsets = feature_block_offsets();
    source.feature_to_block = feature_to_block();
    source.feature_to_local = feature_to_local();
    source.row_group_count = row_group_count_;
    source.row_group_offsets = row_group_offsets();
    source.maximum_feature_block_width = maximum_feature_block_width_;
    source.row_group_width = row_group_width_;
    source.identity = identity_;
    source.objective_kind = objective_kind_;
    source.cost_policy_identity = cost_policy_identity_;
    source.baseline = baseline_;
    source.final = final_;
    const validation_result status = validate_build_view(source);
    if (!status) return status;
    const u64 expected_geometry_identity = compute_feature_block_geometry_identity(
        feature_count_, feature_permutation(), feature_block_count_,
        feature_block_offsets());
    if (feature_block_geometry_identity_ != expected_geometry_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "frozen plan feature-block geometry identity is inconsistent");
    }
    return validation_ok();
}

validation_result frozen_packing_plan::validate_compatibility(
    const packing_plan_compatibility &expected) const {
    if (expected.feature_count != feature_count_ || expected.row_count != row_count_) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "PackingPlan dimensions are incompatible");
    }
    if (expected.feature_axis_fingerprint != identity_.feature_axis_fingerprint
        || expected.feature_axis_fingerprint_version != identity_.feature_axis_fingerprint_version) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "PackingPlan feature axis fingerprint is incompatible");
    }
    if (expected.row_domain_kind != identity_.row_domain_kind
        || expected.row_domain_identity != identity_.row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id, "PackingPlan row domain is incompatible");
    }
    return validation_ok();
}

validation_result freeze_packing_plan(
    const frozen_packing_plan_build_view &source,
    frozen_packing_plan *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "frozen PackingPlan output is null");
    }
    const validation_result status = validate_build_view(source);
    if (!status) return status;
    try {
        packing_plan_image_header header;
        std::size_t cursor = sizeof(packing_plan_image_header);
        std::size_t feature_bytes = 0u;
        std::size_t feature_block_offset_bytes = 0u;
        std::size_t row_group_offset_bytes = 0u;
        if (!checked_u32_bytes(source.feature_count, &feature_bytes)
            || !checked_u32_bytes(static_cast<std::size_t>(source.feature_block_count) + 1u,
                &feature_block_offset_bytes)
            || !checked_u32_bytes(static_cast<std::size_t>(source.row_group_count) + 1u,
                &row_group_offset_bytes)
            || !append_aligned_section(feature_bytes, &cursor, &header.feature_permutation)
            || !append_aligned_section(feature_bytes, &cursor, &header.inverse_feature_permutation)
            || !append_aligned_section(feature_block_offset_bytes, &cursor, &header.feature_block_offsets)
            || !append_aligned_section(feature_bytes, &cursor, &header.feature_to_block)
            || !append_aligned_section(feature_bytes, &cursor, &header.feature_to_local)
            || !append_aligned_section(row_group_offset_bytes, &cursor, &header.row_group_offsets)) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "packing plan image size overflows relative-offset schema");
        }
        const u64 geometry_identity = compute_feature_block_geometry_identity(
            source.feature_count, source.feature_permutation,
            source.feature_block_count, source.feature_block_offsets);
        header.common.magic = packing_plan_image_magic;
        header.common.schema_version = static_cast<std::uint16_t>(packing_plan_semantic_schema_version);
        header.common.total_bytes = cursor;
        header.common.required_alignment = packing_plan_image_alignment;
        header.common.section_count = 6u;
        header.common.identity = geometry_identity;
        header.row_count = source.row_count;
        header.feature_count = source.feature_count;
        header.feature_block_count = source.feature_block_count;
        header.row_group_count = source.row_group_count;
        header.maximum_feature_block_width = source.maximum_feature_block_width;
        header.row_group_width = source.row_group_width;
        header.objective_kind = source.objective_kind;
        header.cost_policy_identity = source.cost_policy_identity;
        header.identity = source.identity;
        header.baseline = source.baseline;
        header.final = source.final;

        auto *storage = static_cast<unsigned char *>(
            ::operator new(cursor, std::align_val_t{packing_plan_image_alignment}));
        std::memset(storage, 0, cursor);
        std::memcpy(storage, &header, sizeof(header));
        auto copy_section = [&](cellerator::memory::rel32 offset, const u32 *data, std::size_t bytes) {
            std::memcpy(storage + offset.byte_offset, data, bytes);
        };
        copy_section(header.feature_permutation, source.feature_permutation, feature_bytes);
        copy_section(header.inverse_feature_permutation, source.inverse_feature_permutation, feature_bytes);
        copy_section(header.feature_block_offsets, source.feature_block_offsets, feature_block_offset_bytes);
        copy_section(header.feature_to_block, source.feature_to_block, feature_bytes);
        copy_section(header.feature_to_local, source.feature_to_local, feature_bytes);
        copy_section(header.row_group_offsets, source.row_group_offsets, row_group_offset_bytes);

        frozen_packing_plan result;
        result.image_storage_.reset(storage);
        result.image_allocation_ = cellerator::memory::allocation{
            storage,
            cursor,
            packing_plan_image_alignment,
            cellerator::memory::placement{cellerator::memory::domain::host, -1, -1, 0u},
            1u
        };
        validation_result image_status = validate_packing_plan_image_host(
            storage, cursor, &result.image_view_);
        if (!image_status) return image_status;
        result.row_count_ = source.row_count;
        result.feature_count_ = source.feature_count;
        result.feature_block_count_ = source.feature_block_count;
        result.row_group_count_ = source.row_group_count;
        result.maximum_feature_block_width_ = source.maximum_feature_block_width;
        result.row_group_width_ = source.row_group_width;
        result.feature_block_geometry_identity_ = geometry_identity;
        result.identity_ = source.identity;
        result.objective_kind_ = source.objective_kind;
        result.cost_policy_identity_ = source.cost_policy_identity;
        result.baseline_ = source.baseline;
        result.final_ = source.final;
        *out = std::move(result);
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::integer_overflow, invalid_id, "frozen PackingPlan allocation failed");
    }
    return validation_ok();
}

} // namespace cellpack
