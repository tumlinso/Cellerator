#include "Cellerator/execution/atom_plane/partial_result_atom_v1.hh"

#include <limits>

namespace cellerator::execution::atom_plane {
namespace {

partial_result_atom_status_v1 failure(
    partial_result_atom_code_v1 code,
    u32 extent_index = 0u,
    u64 subject = 0u,
    mutable_state_atom_plane_code_v1 state_code =
        mutable_state_atom_plane_code_v1::success) noexcept {
    return {code, state_code, 0u, extent_index, subject};
}

partial_result_atom_status_v1 validate_coverage(
    const state_dirty_extent_v1 *extents,
    u32 extent_count,
    u64 total_element_count,
    u64 declared_covered_count) noexcept {
    if (extent_count == 0u || extents == nullptr) {
        return failure(partial_result_atom_code_v1::missing_covered_extents);
    }
    u64 previous_end = 0u;
    u64 observed_count = 0u;
    for (u32 index = 0u; index < extent_count; ++index) {
        const state_dirty_extent_v1 extent = extents[index];
        if (extent.element_count == 0u) {
            return failure(
                partial_result_atom_code_v1::empty_covered_extent, index);
        }
        if (extent.element_offset > std::numeric_limits<u64>::max()
                - extent.element_count
            || extent.element_offset + extent.element_count
                > total_element_count) {
            return failure(
                partial_result_atom_code_v1::covered_extent_out_of_range,
                index, extent.element_offset);
        }
        if (index != 0u && extent.element_offset < previous_end) {
            return failure(partial_result_atom_code_v1::
                    overlapping_or_unsorted_covered_extent,
                index, extent.element_offset);
        }
        if (observed_count > std::numeric_limits<u64>::max()
                - extent.element_count) {
            return failure(
                partial_result_atom_code_v1::covered_element_count_mismatch,
                index, observed_count);
        }
        observed_count += extent.element_count;
        previous_end = extent.element_offset + extent.element_count;
    }
    if (observed_count != declared_covered_count) {
        return failure(
            partial_result_atom_code_v1::covered_element_count_mismatch,
            extent_count, observed_count);
    }
    if (observed_count == total_element_count) {
        return failure(
            partial_result_atom_code_v1::complete_result_not_partial,
            extent_count, observed_count);
    }
    return {};
}

}  // namespace

partial_result_atom_status_v1 emit_partial_result_atom_v1(
    const mutable_state_atom_plane_v1 &state,
    value_generation expected_generation,
    external_atom_plane_identity_v1 result_identity,
    external_atom_plane_identity_v1 merge_algebra_identity,
    const state_dirty_extent_v1 *covered_extents,
    u32 covered_extent_count,
    partial_result_atom_v1 *result) noexcept {
    if (result != nullptr) {
        *result = {};
    }
    if (result == nullptr) {
        return failure(partial_result_atom_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(result_identity)) {
        return failure(partial_result_atom_code_v1::invalid_result_identity);
    }
    if (!valid_external_atom_plane_identity_v1(merge_algebra_identity)) {
        return failure(
            partial_result_atom_code_v1::invalid_merge_algebra_identity);
    }
    const mutable_state_atom_plane_status_v1 state_status =
        validate_mutable_state_atom_plane_v1(state);
    if (!state_status) {
        return failure(partial_result_atom_code_v1::invalid_source_state, 0u,
            state_status.subject, state_status.code);
    }
    if (expected_generation.value == 0u
        || expected_generation.value != state.generation.value) {
        return failure(partial_result_atom_code_v1::stale_source_generation,
            0u, expected_generation.value);
    }
    if (covered_extent_count == 0u || covered_extents == nullptr) {
        return failure(
            partial_result_atom_code_v1::missing_covered_extents);
    }
    u64 covered_count = 0u;
    for (u32 index = 0u; index < covered_extent_count; ++index) {
        if (covered_count > std::numeric_limits<u64>::max()
                - covered_extents[index].element_count) {
            return failure(
                partial_result_atom_code_v1::covered_element_count_mismatch,
                index, covered_count);
        }
        covered_count += covered_extents[index].element_count;
    }
    const partial_result_atom_status_v1 coverage_status = validate_coverage(
        covered_extents, covered_extent_count, state.element_count,
        covered_count);
    if (!coverage_status) {
        return coverage_status;
    }
    result->result_identity = result_identity;
    result->source_state_identity = state.plane_identity;
    result->merge_algebra_identity = merge_algebra_identity;
    result->axis = state.axis;
    result->persistent_order = state.persistent_order;
    result->generation = state.generation;
    result->numeric = state.numeric;
    result->quantization = state.quantization;
    result->values = state.values;
    result->location = state.location;
    result->covered_extents = covered_extents;
    result->total_element_count = state.element_count;
    result->covered_element_count = covered_count;
    result->value_bytes = state.value_bytes;
    result->covered_extent_count = covered_extent_count;
    return {};
}

partial_result_atom_status_v1 validate_partial_result_atom_v1(
    const partial_result_atom_v1 &result) noexcept {
    if (result.schema_version != partial_result_atom_schema_v1
        || result.reserved != 0u || result.reserved1 != 0u
        || !valid_external_atom_plane_identity_v1(result.result_identity)
        || !valid_external_atom_plane_identity_v1(
            result.source_state_identity)
        || result.values == nullptr || result.value_bytes == 0u
        || result.generation.value == 0u || !valid_axis_identity(result.axis)
        || !valid_identity(result.persistent_order)
        || !valid_location(result.location)) {
        return failure(partial_result_atom_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(
            result.merge_algebra_identity)) {
        return failure(
            partial_result_atom_code_v1::invalid_merge_algebra_identity);
    }
    return validate_coverage(result.covered_extents,
        result.covered_extent_count, result.total_element_count,
        result.covered_element_count);
}

}  // namespace cellerator::execution::atom_plane
