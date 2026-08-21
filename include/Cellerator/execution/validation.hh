#pragma once

#include <Cellerator/execution/operands.hh>

#if defined(__CUDACC__)
#define CELLERATOR_EXECUTION_VALIDATION_HD __host__ __device__
#else
#define CELLERATOR_EXECUTION_VALIDATION_HD
#endif

namespace cellerator::execution {

CELLERATOR_EXECUTION_VALIDATION_HD constexpr biological_validation_code
validate_sequence_domain(const sequence_domain &domain) noexcept {
    if (!valid_handle(domain.genome_domain))
        return biological_validation_code::invalid_identity;
    if (domain.owned_begin > domain.owned_end
        || domain.owned_end > domain.local_base_count
        || domain.halo_left > domain.owned_begin
        || domain.halo_right > domain.local_base_count - domain.owned_end)
        return biological_validation_code::invalid_sequence_domain;
    if (domain.global_base_begin
        > ~u64{0} - static_cast<u64>(domain.local_base_count))
        return biological_validation_code::invalid_sequence_domain;
    return biological_validation_code::ok;
}

CELLERATOR_EXECUTION_VALIDATION_HD constexpr biological_validation_code
validate_dense_tensor(const dense_tensor_view &view) noexcept {
    if (!valid_location(view.location))
        return biological_validation_code::invalid_residency;
    if (view.value_type == numeric_type::invalid
        || view.rank == 0u || view.rank > biological_operand_max_axes)
        return biological_validation_code::invalid_shape;
    bool empty = false;
    for (u32 axis = 0u; axis < view.rank; ++axis) {
        if (!valid_axis_identity(view.axes[axis]))
            return biological_validation_code::invalid_identity;
        empty = empty || view.shape[axis] == 0u;
        if (view.shape[axis] != 0u && view.stride[axis] == 0)
            return biological_validation_code::invalid_shape;
    }
    if (!empty && view.data == nullptr)
        return biological_validation_code::missing_pointer;
    return biological_validation_code::ok;
}

CELLERATOR_EXECUTION_VALIDATION_HD constexpr biological_validation_code
validate_bit_plane(const bit_plane_view &view) noexcept {
    if (!valid_axis_identity(view.coordinate_axis))
        return biological_validation_code::invalid_identity;
    if (!valid_location(view.location))
        return biological_validation_code::invalid_residency;
    const u32 required_words = static_cast<u32>(
        (static_cast<u64>(view.base_count) + 31u) / 32u);
    if (view.word_count != required_words)
        return biological_validation_code::invalid_count;
    if (view.word_count != 0u
        && (view.low == nullptr || view.high == nullptr
            || view.validity == nullptr))
        return biological_validation_code::missing_pointer;
    return biological_validation_code::ok;
}

CELLERATOR_EXECUTION_VALIDATION_HD constexpr biological_validation_code
validate_event_stream(const event_stream_view &view) noexcept {
    if (!valid_axis_identity(view.event_axis))
        return biological_validation_code::invalid_identity;
    const biological_validation_code domain_status =
        validate_sequence_domain(view.source_domain);
    if (domain_status != biological_validation_code::ok) return domain_status;
    if (!valid_location(view.location))
        return biological_validation_code::invalid_residency;
    if (view.stored_records > view.total_matches
        || view.dropped_records != view.total_matches - view.stored_records)
        return biological_validation_code::invalid_count;
    if (view.stored_records != 0u
        && (view.local_position == nullptr || view.rule_id == nullptr
            || view.attributes == nullptr || view.strand == nullptr))
        return biological_validation_code::missing_pointer;
    if (view.ordering != event_ordering::unordered
        && view.ordering != event_ordering::coordinate_stable
        && view.ordering != event_ordering::predicate_then_coordinate)
        return biological_validation_code::invalid_ordering;
    return biological_validation_code::ok;
}

CELLERATOR_EXECUTION_VALIDATION_HD constexpr biological_validation_code
validate_segment_stream(const segment_stream_view &view) noexcept {
    if (!valid_axis_identity(view.segment_axis))
        return biological_validation_code::invalid_identity;
    const biological_validation_code domain_status =
        validate_sequence_domain(view.source_domain);
    if (domain_status != biological_validation_code::ok) return domain_status;
    if (!valid_location(view.location))
        return biological_validation_code::invalid_residency;
    if (view.segment_count != 0u
        && (view.begin == nullptr || view.end == nullptr
            || view.class_id == nullptr))
        return biological_validation_code::missing_pointer;
    return biological_validation_code::ok;
}

CELLERATOR_EXECUTION_VALIDATION_HD constexpr biological_validation_code
validate_sparse_relation(const sparse_relation_view &view) noexcept {
    if (!valid_axis_identity(view.source_axis)
        || !valid_axis_identity(view.destination_axis)
        || !valid_handle(view.structure))
        return biological_validation_code::invalid_identity;
    if (!valid_location(view.location))
        return biological_validation_code::invalid_residency;
    const bool has_projection = valid_handle(view.projection);
    if (has_projection != (view.projection_data != nullptr)
        || (!has_projection && view.projection_bytes != 0u))
        return biological_validation_code::missing_pointer;
    return biological_validation_code::ok;
}

CELLERATOR_EXECUTION_VALIDATION_HD constexpr biological_validation_code
validate_scalar_parameter(const scalar_parameter_view &view) noexcept {
    if (!valid_location(view.location))
        return biological_validation_code::invalid_residency;
    if (view.value_type == numeric_type::invalid)
        return biological_validation_code::invalid_shape;
    if (view.element_count != 0u && view.data == nullptr)
        return biological_validation_code::missing_pointer;
    return biological_validation_code::ok;
}

CELLERATOR_EXECUTION_VALIDATION_HD constexpr biological_validation_code
validate_operand(const biological_operand_view &view) noexcept {
    switch (view.kind) {
    case operand_kind::dense_tensor:
        return validate_dense_tensor(view.storage.dense);
    case operand_kind::bit_plane:
        return validate_bit_plane(view.storage.bits);
    case operand_kind::event_stream:
        return validate_event_stream(view.storage.events);
    case operand_kind::segment_stream:
        return validate_segment_stream(view.storage.segments);
    case operand_kind::sparse_relation:
        return validate_sparse_relation(view.storage.relation);
    case operand_kind::scalar_or_small_parameter:
        return validate_scalar_parameter(view.storage.parameter);
    }
    return biological_validation_code::invalid_operand_kind;
}

CELLERATOR_EXECUTION_VALIDATION_HD constexpr bool same_dense_contract(
    const dense_tensor_view &lhs,
    const dense_tensor_view &rhs) noexcept {
    if (lhs.rank != rhs.rank || lhs.value_type != rhs.value_type)
        return false;
    for (u32 axis = 0u; axis < lhs.rank; ++axis) {
        if (lhs.shape[axis] != rhs.shape[axis]
            || lhs.stride[axis] != rhs.stride[axis]
            || !same_axis_identity(lhs.axes[axis], rhs.axes[axis]))
            return false;
    }
    return true;
}

} // namespace cellerator::execution

#undef CELLERATOR_EXECUTION_VALIDATION_HD
