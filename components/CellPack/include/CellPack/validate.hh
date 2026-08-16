#pragma once

#include "CellPack/format.hh"
#include "CellPack/permutation.hh"

namespace cellpack {

enum class validation_code : u32 {
    ok = 0u,
    null_pointer = 1u,
    unsupported_version = 2u,
    invalid_layout = 3u,
    invalid_region_role = 4u,
    invalid_region_bounds = 5u,
    invalid_offsets = 6u,
    duplicate_id = 7u,
    invalid_permutation = 8u,
    unknown_module = 9u,
    invalid_signature = 10u,
    invalid_matrix_view = 11u,
    missing_region = 12u,
    invalid_plan_geometry = 13u,
    insufficient_capacity = 14u,
    integer_overflow = 15u
};

struct validation_result {
    validation_code code;
    u32 index;
    const char *message;

    constexpr explicit operator bool() const {
        return code == validation_code::ok;
    }
};

inline constexpr validation_result validation_ok() {
    return { validation_code::ok, invalid_id, "ok" };
}

inline constexpr validation_result validation_error(validation_code code, u32 index, const char *message) {
    return { code, index, message };
}

inline bool add_overflows_u32(u32 begin, u32 count) {
    return begin + count < begin;
}

inline validation_result validate_plan_desc(const plan_desc &desc) {
    if (desc.version != abi_version) {
        return validation_error(validation_code::unsupported_version, desc.version, "unsupported CellPack ABI version");
    }
    if (desc.residual_region_count > desc.region_count) {
        return validation_error(validation_code::invalid_offsets, desc.residual_region_count, "residual region count exceeds total region count");
    }
    return validation_ok();
}

inline validation_result validate_region_desc(
    const packed_region_desc &region,
    u32 total_rows,
    u32 total_features) {
    const layout_kind layout = static_cast<layout_kind>(region.layout);
    const region_role role = static_cast<region_role>(region.role);
    if (!is_valid_layout(layout)) {
        return validation_error(validation_code::invalid_layout, region.region_id, "invalid packed-region layout");
    }
    if (!is_valid_region_role(role)) {
        return validation_error(validation_code::invalid_region_role, region.region_id, "invalid packed-region role");
    }
    if (add_overflows_u32(region.row_begin, region.row_count)
        || add_overflows_u32(region.feature_begin, region.feature_count)
        || region.row_begin + region.row_count > total_rows
        || region.feature_begin + region.feature_count > total_features) {
        return validation_error(validation_code::invalid_region_bounds, region.region_id, "packed-region rectangle is outside plan bounds");
    }
    if (role == region_role::residual && layout != layout_kind::residual_csr) {
        return validation_error(validation_code::invalid_layout, region.region_id, "residual region must use residual CSR layout in M0/M1");
    }
    if ((region.flags & region_flag_residual) != 0u && role != region_role::residual) {
        return validation_error(validation_code::invalid_region_role, region.region_id, "residual flag requires residual role");
    }
    if ((region.flags & region_flag_conditional) != 0u && role != region_role::conditional) {
        return validation_error(validation_code::invalid_region_role, region.region_id, "conditional flag requires conditional role");
    }
    return validation_ok();
}

inline bool rectangles_overlap(const packed_region_desc &lhs, const packed_region_desc &rhs) {
    const bool rows_disjoint = lhs.row_begin + lhs.row_count <= rhs.row_begin
        || rhs.row_begin + rhs.row_count <= lhs.row_begin;
    const bool features_disjoint = lhs.feature_begin + lhs.feature_count <= rhs.feature_begin
        || rhs.feature_begin + rhs.feature_count <= lhs.feature_begin;
    return !rows_disjoint && !features_disjoint;
}

inline validation_result validate_region_sequence(
    const packed_region_desc *regions,
    u32 region_count,
    u32 total_rows,
    u32 total_features) {
    if (region_count != 0u && regions == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "region array is null");
    }
    for (u32 i = 0; i < region_count; ++i) {
        validation_result region_result = validate_region_desc(regions[i], total_rows, total_features);
        if (!region_result) return region_result;
        for (u32 j = i + 1u; j < region_count; ++j) {
            if (regions[i].region_id == regions[j].region_id) {
                return validation_error(validation_code::duplicate_id, regions[i].region_id, "duplicate packed-region id");
            }
            const region_role lhs_role = static_cast<region_role>(regions[i].role);
            const region_role rhs_role = static_cast<region_role>(regions[j].role);
            if (lhs_role != region_role::discarded
                && rhs_role != region_role::discarded
                && rectangles_overlap(regions[i], regions[j])) {
                return validation_error(validation_code::invalid_region_bounds, regions[j].region_id, "packed regions overlap");
            }
        }
    }
    return validation_ok();
}

inline validation_result validate_permutation_desc(
    const permutation_desc &desc,
    const u32 *permutation,
    const u32 *inverse) {
    if (!validate_permutation(permutation, desc.count)) {
        return validation_error(validation_code::invalid_permutation, desc.count, "invalid permutation");
    }
    if (!validate_inverse_permutation(permutation, inverse, desc.count)) {
        return validation_error(validation_code::invalid_permutation, desc.count, "invalid inverse permutation");
    }
    return validation_ok();
}

} // namespace cellpack
