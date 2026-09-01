#include "Cellerator/execution/atom_plane/dense_result_atom_v1.hh"

namespace cellerator::execution::atom_plane {
namespace {

dense_result_atom_status_v1 failure(
    dense_result_atom_code_v1 code,
    u64 subject = 0u,
    mutable_state_atom_plane_code_v1 state_code =
        mutable_state_atom_plane_code_v1::success) noexcept {
    return {code, state_code, 0u, 0u, subject};
}

bool same_location(device_location lhs, device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

bool same_numeric(value_numeric_policy lhs, value_numeric_policy rhs) noexcept {
    return lhs.storage == rhs.storage && lhs.dequantized == rhs.dequantized
        && lhs.accumulation == rhs.accumulation;
}

bool same_quantization(
    const quantization_descriptor &lhs,
    const quantization_descriptor &rhs) noexcept {
    return lhs.kind == rhs.kind && lhs.scale_type == rhs.scale_type
        && lhs.offset_type == rhs.offset_type && lhs.scales == rhs.scales
        && lhs.offsets == rhs.offsets && lhs.group_count == rhs.group_count;
}

}  // namespace

dense_result_atom_status_v1 emit_persistent_order_dense_result_atom_v1(
    const mutable_state_atom_plane_v1 &state,
    value_generation expected_generation,
    external_atom_plane_identity_v1 result_identity,
    dense_result_atom_v1 *result) noexcept {
    if (result != nullptr) {
        *result = {};
    }
    if (result == nullptr) {
        return failure(dense_result_atom_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(result_identity)) {
        return failure(dense_result_atom_code_v1::invalid_result_identity);
    }
    const mutable_state_atom_plane_status_v1 state_status =
        validate_mutable_state_atom_plane_v1(state);
    if (!state_status) {
        return failure(dense_result_atom_code_v1::invalid_source_state,
            state_status.subject, state_status.code);
    }
    if (expected_generation.value == 0u
        || expected_generation.value != state.generation.value) {
        return failure(dense_result_atom_code_v1::stale_source_generation,
            expected_generation.value);
    }
    result->result_identity = result_identity;
    result->source_state_identity = state.plane_identity;
    result->axis = state.axis;
    result->persistent_order = state.persistent_order;
    result->generation = state.generation;
    result->numeric = state.numeric;
    result->quantization = state.quantization;
    result->values = state.values;
    result->location = state.location;
    result->element_count = state.element_count;
    result->value_bytes = state.value_bytes;
    return {};
}

dense_result_atom_status_v1 validate_persistent_order_dense_result_atom_v1(
    const mutable_state_atom_plane_v1 &state,
    const dense_result_atom_v1 &result) noexcept {
    const mutable_state_atom_plane_status_v1 state_status =
        validate_mutable_state_atom_plane_v1(state);
    if (!state_status) {
        return failure(dense_result_atom_code_v1::invalid_source_state,
            state_status.subject, state_status.code);
    }
    if (result.schema_version != dense_result_atom_schema_v1
        || result.reserved != 0u
        || !valid_external_atom_plane_identity_v1(result.result_identity)) {
        return failure(dense_result_atom_code_v1::invalid_argument);
    }
    if (!same_external_atom_plane_identity_v1(
            result.source_state_identity, state.plane_identity)) {
        return failure(dense_result_atom_code_v1::source_identity_mismatch);
    }
    if (!same_axis_identity(result.axis, state.axis)) {
        return failure(dense_result_atom_code_v1::axis_mismatch);
    }
    if (!same_identity(result.persistent_order, state.persistent_order)) {
        return failure(dense_result_atom_code_v1::persistent_order_mismatch);
    }
    if (result.generation.value != state.generation.value) {
        return failure(dense_result_atom_code_v1::generation_mismatch,
            result.generation.value);
    }
    if (!same_numeric(result.numeric, state.numeric)) {
        return failure(dense_result_atom_code_v1::numeric_policy_mismatch);
    }
    if (!same_quantization(result.quantization, state.quantization)) {
        return failure(dense_result_atom_code_v1::quantization_mismatch);
    }
    if (result.values != state.values) {
        return failure(dense_result_atom_code_v1::values_mismatch);
    }
    if (!same_location(result.location, state.location)) {
        return failure(dense_result_atom_code_v1::location_mismatch);
    }
    if (result.element_count != state.element_count
        || result.value_bytes != state.value_bytes) {
        return failure(dense_result_atom_code_v1::extent_mismatch,
            result.element_count);
    }
    return {};
}

}  // namespace cellerator::execution::atom_plane
