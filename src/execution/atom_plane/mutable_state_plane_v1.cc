#include "Cellerator/execution/atom_plane/mutable_state_plane_v1.hh"

#include <limits>

namespace cellerator::execution::atom_plane {
namespace {

mutable_state_atom_plane_status_v1 failure(
    mutable_state_atom_plane_code_v1 code,
    u32 dirty_extent_index = 0u,
    u64 subject = 0u) noexcept {
    return {code, dirty_extent_index, subject};
}

bool valid_numeric(const value_numeric_policy &numeric) noexcept {
    return numeric.storage != numeric_type::invalid
        && numeric.dequantized != numeric_type::invalid
        && numeric.accumulation != numeric_type::invalid
        && numeric.reserved == 0u;
}

bool valid_quantization(const quantization_descriptor &quantization) noexcept {
    if (quantization.kind == quantization_kind::none) {
        return quantization.scales == nullptr
            && quantization.offsets == nullptr
            && quantization.group_count == 0u;
    }
    return (quantization.kind == quantization_kind::per_value_plane
            || quantization.kind == quantization_kind::per_module
            || quantization.kind == quantization_kind::per_block)
        && quantization.scales != nullptr && quantization.group_count != 0u
        && quantization.scale_type != numeric_type::invalid;
}

}  // namespace

mutable_state_atom_plane_status_v1 validate_mutable_state_atom_plane_v1(
    const mutable_state_atom_plane_v1 &plane) noexcept {
    if (plane.schema_version != mutable_state_atom_plane_schema_v1
        || plane.reserved != 0u || plane.reserved1 != 0u
        || plane.element_count == 0u) {
        return failure(mutable_state_atom_plane_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(plane.plane_identity)) {
        return failure(mutable_state_atom_plane_code_v1::invalid_plane_identity);
    }
    if (!valid_axis_identity(plane.axis)) {
        return failure(mutable_state_atom_plane_code_v1::invalid_axis);
    }
    if (!valid_identity(plane.persistent_order)) {
        return failure(
            mutable_state_atom_plane_code_v1::invalid_persistent_order);
    }
    if (plane.generation.value == 0u) {
        return failure(mutable_state_atom_plane_code_v1::missing_generation);
    }
    if (!valid_numeric(plane.numeric)) {
        return failure(
            mutable_state_atom_plane_code_v1::invalid_numeric_policy);
    }
    if (!valid_quantization(plane.quantization)) {
        return failure(
            mutable_state_atom_plane_code_v1::invalid_quantization);
    }
    if (!valid_location(plane.location)) {
        return failure(mutable_state_atom_plane_code_v1::invalid_location);
    }
    if (plane.values == nullptr || plane.value_bytes == 0u) {
        return failure(mutable_state_atom_plane_code_v1::missing_values);
    }
    if (plane.value_capacity_bytes < plane.value_bytes) {
        return failure(mutable_state_atom_plane_code_v1::insufficient_capacity,
            0u, plane.value_capacity_bytes);
    }
    if (plane.dirty_extent_count != 0u && plane.dirty_extents == nullptr) {
        return failure(
            mutable_state_atom_plane_code_v1::missing_dirty_extents);
    }
    u64 previous_end = 0u;
    for (u32 index = 0u; index < plane.dirty_extent_count; ++index) {
        const state_dirty_extent_v1 &extent = plane.dirty_extents[index];
        if (extent.element_count == 0u) {
            return failure(
                mutable_state_atom_plane_code_v1::empty_dirty_extent, index);
        }
        if (extent.element_offset > std::numeric_limits<u64>::max()
                - extent.element_count
            || extent.element_offset + extent.element_count
                > plane.element_count) {
            return failure(
                mutable_state_atom_plane_code_v1::dirty_extent_out_of_range,
                index, extent.element_offset);
        }
        if (index != 0u && extent.element_offset < previous_end) {
            return failure(mutable_state_atom_plane_code_v1::
                    overlapping_or_unsorted_dirty_extent,
                index, extent.element_offset);
        }
        previous_end = extent.element_offset + extent.element_count;
    }
    return {};
}

}  // namespace cellerator::execution::atom_plane
