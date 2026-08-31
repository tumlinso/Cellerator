#include <Cellerator/compute/candidate/segment/segment_v2.hh>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cellerator::compute::segment {
namespace {

segment_result_v2 error(segment_status_v2 code, const char *message) noexcept {
    return {code, message};
}

bool valid_operation(segment_operation_v2 value) noexcept {
    return value == segment_operation_v2::reduce
        || value == segment_operation_v2::normalize;
}

bool valid_direction(segment_direction_v2 value) noexcept {
    return value == segment_direction_v2::forward
        || value == segment_direction_v2::backward;
}

bool valid_reduction(segment_reduce_kind_v2 value) noexcept {
    return value == segment_reduce_kind_v2::sum
        || value == segment_reduce_kind_v2::mean
        || value == segment_reduce_kind_v2::minimum
        || value == segment_reduce_kind_v2::maximum
        || value == segment_reduce_kind_v2::sum_of_squares
        || value == segment_reduce_kind_v2::first_second_moments;
}

bool valid_normalization(segment_normalize_kind_v2 value) noexcept {
    return value == segment_normalize_kind_v2::log_sum_exp
        || value == segment_normalize_kind_v2::softmax
        || value == segment_normalize_kind_v2::log_softmax
        || value == segment_normalize_kind_v2::l1
        || value == segment_normalize_kind_v2::l2
        || value == segment_normalize_kind_v2::rms;
}

bool valid_mechanism(segment_mechanism_v2 value) noexcept {
    return value == segment_mechanism_v2::warp_per_output
        || value == segment_mechanism_v2::cta_per_output
        || value == segment_mechanism_v2::large_segment_cta;
}

bool valid_order(segment_storage_order_v2 value) noexcept {
    return value == segment_storage_order_v2::logical_edge
        || value == segment_storage_order_v2::projection
        || value == segment_storage_order_v2::cover_native;
}

bool valid_width(segment_local_index_width_v2 value) noexcept {
    return value == segment_local_index_width_v2::u16
        || value == segment_local_index_width_v2::u32
        || value == segment_local_index_width_v2::u64;
}

bool valid_nonfinite(segment_nonfinite_policy_v2 value) noexcept {
    return value == segment_nonfinite_policy_v2::propagate
        || value == segment_nonfinite_policy_v2::reject;
}

bool all_zero(const std::uint8_t *values, std::uint32_t count) noexcept {
    for (std::uint32_t index = 0u; index < count; ++index)
        if (values[index] != 0u) return false;
    return true;
}

bool interval_fits(std::uint64_t begin, std::uint64_t count,
    std::uint64_t extent) noexcept {
    return begin <= extent && count <= extent - begin;
}

} // namespace

segment_result_v2 validate_segment_plan_v2(
    const segment_plan_v2 &plan) noexcept {
    if (plan.schema_version != segment_schema_version_v2)
        return error(segment_status_v2::unsupported_schema,
            "segment v2 schema is unsupported");
    if (!valid_operation(plan.operation) || !valid_direction(plan.direction)
        || !valid_reduction(plan.reduction)
        || !valid_normalization(plan.normalization)
        || !valid_mechanism(plan.mechanism)
        || !valid_order(plan.storage_order)
        || !valid_width(plan.local_index_width)
        || !valid_nonfinite(plan.nan_policy)
        || !valid_nonfinite(plan.infinity_policy)
        || !all_zero(plan.reserved0, 2u)
        || !all_zero(plan.reserved1, 5u)
        || !all_zero(plan.reserved2, 7u))
        return error(segment_status_v2::invalid_argument,
            "segment v2 enum or reserved field is invalid");
    if (plan.operation == segment_operation_v2::reduce
        && plan.direction != segment_direction_v2::forward)
        return error(segment_status_v2::invalid_argument,
            "segment reductions expose forward execution only");
    if (!execution::valid_axis_identity(plan.values_axis)
        || !execution::valid_axis_identity(plan.segment_axis)
        || !execution::valid_axis_identity(plan.dense_axis)
        || plan.partition_identity == 0u || plan.operation_identity == 0u
        || plan.stage_identity == 0u)
        return error(segment_status_v2::invalid_identity,
            "segment v2 identity is invalid");
    if (plan.dense_width == 0u
        || !interval_fits(plan.component_value_begin,
            plan.local_value_count, plan.global_value_count)
        || !interval_fits(plan.component_segment_begin,
            plan.local_segment_count, plan.global_segment_count)
        || (plan.local_segment_count == 0u && plan.local_value_count != 0u)
        || (plan.local_segment_count != 0u
            && plan.maximum_segment_length == 0u))
        return error(segment_status_v2::invalid_shape,
            "segment v2 local component does not fit global extents");
    if (plan.local_index_width == segment_local_index_width_v2::u16
        && plan.local_value_count > std::numeric_limits<std::uint16_t>::max())
        return error(segment_status_v2::invalid_shape,
            "segment v2 local component exceeds u16 index width");
    if (plan.local_index_width == segment_local_index_width_v2::u32
        && plan.local_value_count > std::numeric_limits<std::uint32_t>::max())
        return error(segment_status_v2::invalid_shape,
            "segment v2 local component exceeds u32 index width");
    if (plan.input_type != execution::numeric_type::f32
        || plan.accumulation_type != execution::numeric_type::f32
        || plan.output_type != execution::numeric_type::f32)
        return error(segment_status_v2::unsupported_numeric_policy,
            "segment v2 requires FP32 input, accumulation, and output");
    if (!std::isfinite(plan.epsilon) || plan.epsilon < 0.0f
        || (plan.operation == segment_operation_v2::normalize
            && (plan.normalization == segment_normalize_kind_v2::l2
                || plan.normalization == segment_normalize_kind_v2::rms)
            && plan.epsilon == 0.0f))
        return error(segment_status_v2::invalid_argument,
            "segment v2 normalization epsilon is invalid");
    return {};
}

segment_result_v2 validate_segment_partition_offsets_v2_host(
    const segment_plan_v2 &plan,
    const std::uint64_t *offsets,
    std::uint64_t offset_count) noexcept {
    const segment_result_v2 valid = validate_segment_plan_v2(plan);
    if (!valid) return valid;
    const std::uint64_t expected =
        static_cast<std::uint64_t>(plan.local_segment_count) + 1u;
    if (offset_count != expected || offsets == nullptr)
        return error(segment_status_v2::invalid_partition,
            "segment v2 partition offset shape is invalid");
    if (offsets[0] != 0u
        || offsets[offset_count - 1u] != plan.local_value_count)
        return error(segment_status_v2::invalid_partition,
            "segment v2 partition endpoints do not cover local values");
    std::uint64_t maximum_length = 0u;
    for (std::uint64_t index = 1u; index < offset_count; ++index) {
        if (offsets[index] < offsets[index - 1u]
            || offsets[index] > plan.local_value_count)
            return error(segment_status_v2::invalid_partition,
                "segment v2 partition offsets are not monotonic");
        const std::uint64_t length = offsets[index] - offsets[index - 1u];
        if (length > maximum_length) maximum_length = length;
    }
    if (maximum_length > plan.maximum_segment_length)
        return error(segment_status_v2::invalid_partition,
            "segment v2 maximum segment length understates the partition");
    return {};
}

segment_workspace_requirements_v2 query_segment_workspace_v2(
    const segment_plan_v2 &) noexcept {
    return {};
}

std::uint32_t segment_output_planes_v2(
    const segment_plan_v2 &plan) noexcept {
    return plan.operation == segment_operation_v2::reduce
            && plan.reduction == segment_reduce_kind_v2::first_second_moments
        ? 2u : 1u;
}

} // namespace cellerator::compute::segment
