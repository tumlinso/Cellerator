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

validation_result make_coordinate(
    const static_plan &plan,
    u32 original_row,
    u32 original_feature,
    float value,
    packed_coordinate *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "packed coordinate storage is null");
    }
    const u32 permuted_row = plan.inverse_row_permutation[original_row];
    const u32 permuted_feature = plan.inverse_feature_permutation[original_feature];
    const u32 region_id = find_region_id(plan, permuted_row, permuted_feature);
    if (region_id == invalid_id) {
        return validation_error(validation_code::missing_region, original_row,
                                "source entry does not map to a precompiled CellPack region");
    }
    *out = {original_row, original_feature, permuted_row, permuted_feature, region_id, value};
    return validation_ok();
}

} // namespace

packed_coordinate_plan_view view_packed_coordinates(const packed_coordinate_plan &plan) {
    return {plan.row_count, plan.feature_count,
            {plan.coordinates.data(), plan.coordinates.size(),
             {::cellerator::memory::domain::host, -1, -1, 0u}}};
}

validation_result build_packed_coordinate_plan_into(
    const csr_view &source,
    const static_plan &plan,
    packed_coordinate_plan_storage storage,
    packed_coordinate_plan_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "packed coordinate view output is null");
    }
    validation_result source_result = validate_csr_view(source);
    if (!source_result) return source_result;
    validation_result plan_result = validate_plan_for_coordinates(
        plan, source.row_count, source.feature_count);
    if (!plan_result) return plan_result;
    if (storage.coordinates.count < source.nnz_count
        || (source.nnz_count != 0u && storage.coordinates.data == nullptr)) {
        return validation_error(validation_code::insufficient_capacity, source.nnz_count,
                                "packed coordinate storage capacity is insufficient");
    }
    u32 cursor = 0u;
    for (u32 row = 0; row < source.row_count; ++row) {
        for (u32 entry = source.row_offsets[row]; entry < source.row_offsets[row + 1u]; ++entry) {
            validation_result result = make_coordinate(
                plan, row, source.feature_ids[entry], source.values[entry],
                storage.coordinates.data + cursor);
            if (!result) return result;
            ++cursor;
        }
    }
    *out = {source.row_count, source.feature_count,
            {storage.coordinates.data, cursor, storage.coordinates.where}};
    return validation_ok();
}

validation_result build_packed_coordinate_plan_into(
    const coo_view &source,
    const static_plan &plan,
    packed_coordinate_plan_storage storage,
    packed_coordinate_plan_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "packed coordinate view output is null");
    }
    validation_result source_result = validate_coo_view(source);
    if (!source_result) return source_result;
    validation_result plan_result = validate_plan_for_coordinates(
        plan, source.row_count, source.feature_count);
    if (!plan_result) return plan_result;
    if (storage.coordinates.count < source.nnz_count
        || (source.nnz_count != 0u && storage.coordinates.data == nullptr)) {
        return validation_error(validation_code::insufficient_capacity, source.nnz_count,
                                "packed coordinate storage capacity is insufficient");
    }
    for (u32 entry = 0; entry < source.nnz_count; ++entry) {
        validation_result result = make_coordinate(
            plan, source.row_ids[entry], source.feature_ids[entry], source.values[entry],
            storage.coordinates.data + entry);
        if (!result) return result;
    }
    *out = {source.row_count, source.feature_count,
            {storage.coordinates.data, source.nnz_count, storage.coordinates.where}};
    return validation_ok();
}

validation_result reconstruct_csr_from_coordinate_plan_into(
    u32 row_count,
    u32 feature_count,
    const static_plan &plan,
    packed_coordinate_plan_view packed,
    reconstructed_csr_view output) {
    if (packed.row_count != row_count || packed.feature_count != feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
                                "packed coordinate dimensions do not match reconstruction shape");
    }
    validation_result plan_result = validate_plan_for_coordinates(plan, row_count, feature_count);
    if (!plan_result) return plan_result;
    const std::size_t nnz = packed.coordinates.count;
    if ((nnz != 0u && packed.coordinates.data == nullptr)
        || output.row_offsets.count < static_cast<std::size_t>(row_count) + 1u
        || output.feature_ids.count < nnz || output.values.count < nnz
        || output.row_offsets.data == nullptr
        || (nnz != 0u && (output.feature_ids.data == nullptr || output.values.data == nullptr))) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
                                "direct CSR reconstruction capacity is insufficient");
    }
    std::fill_n(output.row_offsets.data, static_cast<std::size_t>(row_count) + 1u, 0u);
    u32 previous_row = 0u, previous_feature = 0u;
    bool have_previous = false;
    for (std::size_t index = 0u; index < nnz; ++index) {
        const packed_coordinate &coordinate = packed.coordinates.data[index];
        if (coordinate.original_row >= row_count || coordinate.original_feature >= feature_count) {
            return validation_error(validation_code::invalid_matrix_view,
                index > invalid_id ? invalid_id : static_cast<u32>(index),
                "packed coordinate original index is outside reconstruction bounds");
        }
        if (have_previous && (coordinate.original_row < previous_row
            || (coordinate.original_row == previous_row
                && coordinate.original_feature <= previous_feature))) {
            return validation_error(validation_code::invalid_matrix_view,
                index > invalid_id ? invalid_id : static_cast<u32>(index),
                "direct CSR reconstruction requires canonical row-major coordinates");
        }
        if (coordinate.permuted_row != plan.inverse_row_permutation[coordinate.original_row]
            || coordinate.permuted_feature != plan.inverse_feature_permutation[coordinate.original_feature]
            || find_region_id(plan, coordinate.permuted_row,
                              coordinate.permuted_feature) != coordinate.region_id) {
            return validation_error(validation_code::invalid_permutation,
                index > invalid_id ? invalid_id : static_cast<u32>(index),
                "packed coordinate identity does not match the static plan");
        }
        ++output.row_offsets.data[coordinate.original_row + 1u];
        output.feature_ids.data[index] = coordinate.original_feature;
        output.values.data[index] = coordinate.value;
        previous_row = coordinate.original_row;
        previous_feature = coordinate.original_feature;
        have_previous = true;
    }
    for (u32 row = 0u; row < row_count; ++row) {
        output.row_offsets.data[row + 1u] += output.row_offsets.data[row];
    }
    output.row_count = row_count;
    output.feature_count = feature_count;
    output.row_offsets.count = static_cast<std::size_t>(row_count) + 1u;
    output.feature_ids.count = nnz;
    output.values.count = nnz;
    return validation_ok();
}

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
    packed_coordinate_plan packed;
    packed.coordinates.resize(source.nnz_count);
    packed_coordinate_plan_view view;
    validation_result result = build_packed_coordinate_plan_into(
        source, plan, {{packed.coordinates.data(), packed.coordinates.size(), {}}}, &view);
    if (!result) return result;
    packed.row_count = view.row_count;
    packed.feature_count = view.feature_count;
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
    packed_coordinate_plan packed;
    packed.coordinates.resize(source.nnz_count);
    packed_coordinate_plan_view view;
    validation_result result = build_packed_coordinate_plan_into(
        source, plan, {{packed.coordinates.data(), packed.coordinates.size(), {}}}, &view);
    if (!result) return result;
    packed.row_count = view.row_count;
    packed.feature_count = view.feature_count;
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
    reconstructed_csr csr;
    csr.row_count = row_count;
    csr.feature_count = feature_count;
    csr.row_offsets.resize(static_cast<std::size_t>(row_count) + 1u);
    csr.feature_ids.resize(packed.coordinates.size());
    csr.values.resize(packed.coordinates.size());
    validation_result result = reconstruct_csr_from_coordinate_plan_into(
        row_count, feature_count, plan, view_packed_coordinates(packed),
        {row_count, feature_count,
         {csr.row_offsets.data(), csr.row_offsets.size(), {}},
         {csr.feature_ids.data(), csr.feature_ids.size(), {}},
         {csr.values.data(), csr.values.size(), {}}});
    if (!result) return result;
    *out = std::move(csr);
    return validation_ok();
}

} // namespace cellpack
