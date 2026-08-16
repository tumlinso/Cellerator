#include "CellPack/packing_plan.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
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

std::unique_ptr<u32[]> copy_u32(const u32 *source, std::size_t count) {
    if (count == 0u) return {};
    std::unique_ptr<u32[]> result(new u32[count]);
    std::copy(source, source + count, result.get());
    return result;
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

packing_plan_view frozen_packing_plan::view() const noexcept {
    packing_plan_view result;
    result.row_count = row_count_;
    result.feature_count = feature_count_;
    result.feature_permutation = feature_permutation_.get();
    result.inverse_feature_permutation = inverse_feature_permutation_.get();
    result.row_group_count = row_group_count_;
    result.row_group_offsets = row_group_offsets_.get();
    result.feature_block_count = feature_block_count_;
    result.feature_block_offsets = feature_block_offsets_.get();
    return result;
}

validation_result frozen_packing_plan::validate() const {
    frozen_packing_plan_build_view source;
    source.row_count = row_count_;
    source.feature_count = feature_count_;
    source.feature_permutation = feature_permutation_.get();
    source.inverse_feature_permutation = inverse_feature_permutation_.get();
    source.feature_block_count = feature_block_count_;
    source.feature_block_offsets = feature_block_offsets_.get();
    source.feature_to_block = feature_to_block_.get();
    source.feature_to_local = feature_to_local_.get();
    source.row_group_count = row_group_count_;
    source.row_group_offsets = row_group_offsets_.get();
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
        feature_count_, feature_permutation_.get(), feature_block_count_,
        feature_block_offsets_.get());
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
        frozen_packing_plan result;
        result.row_count_ = source.row_count;
        result.feature_count_ = source.feature_count;
        result.feature_block_count_ = source.feature_block_count;
        result.row_group_count_ = source.row_group_count;
        result.maximum_feature_block_width_ = source.maximum_feature_block_width;
        result.row_group_width_ = source.row_group_width;
        result.feature_block_geometry_identity_ = compute_feature_block_geometry_identity(
            source.feature_count, source.feature_permutation,
            source.feature_block_count, source.feature_block_offsets);
        result.feature_permutation_ = copy_u32(source.feature_permutation, source.feature_count);
        result.inverse_feature_permutation_ = copy_u32(source.inverse_feature_permutation, source.feature_count);
        result.feature_block_offsets_ = copy_u32(source.feature_block_offsets,
            static_cast<std::size_t>(source.feature_block_count) + 1u);
        result.feature_to_block_ = copy_u32(source.feature_to_block, source.feature_count);
        result.feature_to_local_ = copy_u32(source.feature_to_local, source.feature_count);
        result.row_group_offsets_ = copy_u32(source.row_group_offsets,
            static_cast<std::size_t>(source.row_group_count) + 1u);
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
