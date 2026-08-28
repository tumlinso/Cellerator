#include "Cellerator/geometry/pack.hh"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace cellpack {
namespace {

validation_result validate_plan_for_coordinates(const static_plan &plan, u32 row_count, u32 feature_count) {
    validation_result desc_result = validate_plan_desc(plan.desc);
    if (!desc_result) return desc_result;
    if (plan.desc.row_count != row_count || plan.desc.feature_count != feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "source matrix dimensions do not match CellPack plan");
    }
    validation_result row_perm_result = validate_permutation_desc(
        plan.desc.row_permutation,
        plan.row_permutation.data(),
        plan.inverse_row_permutation.data());
    if (!row_perm_result) return row_perm_result;
    validation_result feature_perm_result = validate_permutation_desc(
        plan.desc.feature_permutation,
        plan.feature_permutation.data(),
        plan.inverse_feature_permutation.data());
    if (!feature_perm_result) return feature_perm_result;
    return validate_region_sequence(
        plan.regions.data(),
        static_cast<u32>(plan.regions.size()),
        row_count,
        feature_count);
}

bool contains_coordinate(const packed_region_desc &region, u32 permuted_row, u32 permuted_feature) {
    return permuted_row >= region.row_begin
        && permuted_row < region.row_begin + region.row_count
        && permuted_feature >= region.feature_begin
        && permuted_feature < region.feature_begin + region.feature_count;
}

u32 find_region_id(const static_plan &plan, u32 permuted_row, u32 permuted_feature) {
    for (const packed_region_desc &region : plan.regions) {
        if (static_cast<region_role>(region.role) == region_role::discarded) continue;
        if (contains_coordinate(region, permuted_row, permuted_feature)) return region.region_id;
    }
    return invalid_id;
}

validation_result append_coordinate(
    const static_plan &plan,
    u32 original_row,
    u32 original_feature,
    float value,
    packed_coordinate_plan *out) {
    const u32 permuted_row = plan.inverse_row_permutation[original_row];
    const u32 permuted_feature = plan.inverse_feature_permutation[original_feature];
    const u32 region_id = find_region_id(plan, permuted_row, permuted_feature);
    if (region_id == invalid_id) {
        return validation_error(validation_code::missing_region, original_row, "source entry does not map to a precompiled CellPack region");
    }
    packed_coordinate coordinate;
    coordinate.original_row = original_row;
    coordinate.original_feature = original_feature;
    coordinate.permuted_row = permuted_row;
    coordinate.permuted_feature = permuted_feature;
    coordinate.region_id = region_id;
    coordinate.value = value;
    out->coordinates.push_back(coordinate);
    return validation_ok();
}

bool coordinate_less(const packed_coordinate &lhs, const packed_coordinate &rhs) {
    if (lhs.original_row != rhs.original_row) return lhs.original_row < rhs.original_row;
    return lhs.original_feature < rhs.original_feature;
}

} // namespace

validation_result validate_csr_view(const csr_view &view) {
    if (view.row_count != 0u && view.row_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "CSR row offsets are null");
    }
    if (view.nnz_count != 0u && (view.feature_ids == nullptr || view.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id, "CSR feature ids or values are null");
    }
    if (view.row_count == 0u) {
        return view.nnz_count == 0u
            ? validation_ok()
            : validation_error(validation_code::invalid_matrix_view, invalid_id, "empty CSR row axis cannot contain nonzeros");
    }
    if (view.row_offsets[0] != 0u) {
        return validation_error(validation_code::invalid_matrix_view, 0u, "CSR row offsets must start at zero");
    }
    for (u32 row = 0; row < view.row_count; ++row) {
        const u32 begin = view.row_offsets[row];
        const u32 end = view.row_offsets[row + 1u];
        if (end < begin || end > view.nnz_count) {
            return validation_error(validation_code::invalid_matrix_view, row, "CSR row offsets are not monotonic");
        }
        u32 previous_feature = invalid_id;
        for (u32 entry = begin; entry < end; ++entry) {
            const u32 feature = view.feature_ids[entry];
            if (feature >= view.feature_count) {
                return validation_error(validation_code::invalid_matrix_view, entry, "CSR feature id is outside matrix bounds");
            }
            if (previous_feature != invalid_id && feature <= previous_feature) {
                return validation_error(validation_code::invalid_matrix_view, entry, "CSR feature ids must be strictly increasing within each row");
            }
            previous_feature = feature;
        }
    }
    if (view.row_offsets[view.row_count] != view.nnz_count) {
        return validation_error(validation_code::invalid_matrix_view, view.row_count, "CSR final row offset does not match nnz count");
    }
    return validation_ok();
}

validation_result validate_coo_view(const coo_view &view) {
    if (view.nnz_count != 0u && (view.row_ids == nullptr || view.feature_ids == nullptr || view.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id, "COO rows, features, or values are null");
    }
    u32 previous_row = 0u;
    u32 previous_feature = 0u;
    bool have_previous = false;
    for (u32 entry = 0; entry < view.nnz_count; ++entry) {
        const u32 row = view.row_ids[entry];
        const u32 feature = view.feature_ids[entry];
        if (row >= view.row_count || feature >= view.feature_count) {
            return validation_error(validation_code::invalid_matrix_view, entry, "COO coordinate is outside matrix bounds");
        }
        if (have_previous
            && (row < previous_row || (row == previous_row && feature <= previous_feature))) {
            return validation_error(validation_code::invalid_matrix_view, entry, "COO coordinates must be strictly increasing in row-major order");
        }
        previous_row = row;
        previous_feature = feature;
        have_previous = true;
    }
    return validation_ok();
}

validation_result build_packed_coordinate_plan(
    const csr_view &source,
    const static_plan &plan,
    packed_coordinate_plan *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "packed coordinate output is null");
    }
    validation_result source_result = validate_csr_view(source);
    if (!source_result) return source_result;
    validation_result plan_result = validate_plan_for_coordinates(plan, source.row_count, source.feature_count);
    if (!plan_result) return plan_result;

    packed_coordinate_plan packed;
    packed.row_count = source.row_count;
    packed.feature_count = source.feature_count;
    packed.coordinates.reserve(source.nnz_count);
    for (u32 row = 0; row < source.row_count; ++row) {
        for (u32 entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u]; ++entry) {
            validation_result append_result = append_coordinate(
                plan,
                row,
                source.feature_ids[entry],
                source.values[entry],
                &packed);
            if (!append_result) return append_result;
        }
    }
    *out = std::move(packed);
    return validation_ok();
}

validation_result build_packed_coordinate_plan(
    const coo_view &source,
    const static_plan &plan,
    packed_coordinate_plan *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "packed coordinate output is null");
    }
    validation_result source_result = validate_coo_view(source);
    if (!source_result) return source_result;
    validation_result plan_result = validate_plan_for_coordinates(plan, source.row_count, source.feature_count);
    if (!plan_result) return plan_result;

    packed_coordinate_plan packed;
    packed.row_count = source.row_count;
    packed.feature_count = source.feature_count;
    packed.coordinates.reserve(source.nnz_count);
    for (u32 entry = 0; entry < source.nnz_count; ++entry) {
        validation_result append_result = append_coordinate(
            plan,
            source.row_ids[entry],
            source.feature_ids[entry],
            source.values[entry],
            &packed);
        if (!append_result) return append_result;
    }
    *out = std::move(packed);
    return validation_ok();
}

validation_result reconstruct_csr_from_coordinate_plan(
    u32 row_count,
    u32 feature_count,
    const static_plan &plan,
    const packed_coordinate_plan &packed,
    reconstructed_csr *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "reconstructed CSR output is null");
    }
    if (packed.row_count != row_count || packed.feature_count != feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "packed coordinate dimensions do not match requested reconstruction shape");
    }
    validation_result plan_result = validate_plan_for_coordinates(plan, row_count, feature_count);
    if (!plan_result) return plan_result;

    std::vector<packed_coordinate> sorted = packed.coordinates;
    std::sort(sorted.begin(), sorted.end(), coordinate_less);
    for (u32 i = 0; i < static_cast<u32>(sorted.size()); ++i) {
        const packed_coordinate &coordinate = sorted[i];
        if (coordinate.original_row >= row_count || coordinate.original_feature >= feature_count) {
            return validation_error(validation_code::invalid_matrix_view, i, "packed coordinate original index is outside reconstruction bounds");
        }
        if (coordinate.permuted_row != plan.inverse_row_permutation[coordinate.original_row]
            || coordinate.permuted_feature != plan.inverse_feature_permutation[coordinate.original_feature]) {
            return validation_error(validation_code::invalid_permutation, i, "packed coordinate permutation fields do not match plan inverse maps");
        }
        if (find_region_id(plan, coordinate.permuted_row, coordinate.permuted_feature) != coordinate.region_id) {
            return validation_error(validation_code::missing_region, i, "packed coordinate region does not match plan region lookup");
        }
        if (i != 0u
            && sorted[i - 1u].original_row == coordinate.original_row
            && sorted[i - 1u].original_feature == coordinate.original_feature) {
            return validation_error(validation_code::invalid_matrix_view, i, "duplicate packed coordinate in reconstruction");
        }
    }

    reconstructed_csr csr;
    csr.row_count = row_count;
    csr.feature_count = feature_count;
    csr.row_offsets.assign(static_cast<std::size_t>(row_count) + 1u, 0u);
    csr.feature_ids.reserve(sorted.size());
    csr.values.reserve(sorted.size());
    for (const packed_coordinate &coordinate : sorted) {
        ++csr.row_offsets[coordinate.original_row + 1u];
        csr.feature_ids.push_back(coordinate.original_feature);
        csr.values.push_back(coordinate.value);
    }
    for (u32 row = 0; row < row_count; ++row) {
        csr.row_offsets[row + 1u] += csr.row_offsets[row];
    }
    *out = std::move(csr);
    return validation_ok();
}

} // namespace cellpack
