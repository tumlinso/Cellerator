#include <Cellerator/compute/candidate/segment/reduce_v2.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cellerator::compute::segment {
namespace {

segment_result_v2 error(segment_status_v2 code, const char *message) noexcept {
    return {code, message};
}

} // namespace

segment_result_v2 reference_segment_reduce_v2(
    const segment_plan_v2 &plan,
    const std::uint64_t *offsets,
    std::uint64_t offset_count,
    const float *values,
    std::uint64_t value_element_count,
    float *output,
    float *second_moment_output,
    std::uint64_t output_element_count) noexcept {
    const segment_result_v2 valid = validate_segment_plan_v2(plan);
    if (!valid) return valid;
    if (plan.operation != segment_operation_v2::reduce
        || plan.direction != segment_direction_v2::forward)
        return error(segment_status_v2::invalid_argument,
            "segment reduction requires a forward reduction plan");
    const segment_result_v2 partition_valid =
        validate_segment_partition_offsets_v2_host(plan, offsets, offset_count);
    if (!partition_valid) return partition_valid;
    if (plan.local_value_count != 0u
        && plan.dense_width > std::numeric_limits<std::uint64_t>::max()
            / plan.local_value_count)
        return error(segment_status_v2::invalid_shape,
            "segment reduction input shape overflows");
    const std::uint64_t expected_values =
        plan.local_value_count * plan.dense_width;
    const std::uint64_t expected_output =
        static_cast<std::uint64_t>(plan.local_segment_count) * plan.dense_width;
    const bool paired = plan.reduction
        == segment_reduce_kind_v2::first_second_moments;
    if (value_element_count != expected_values
        || output_element_count != expected_output
        || (expected_values != 0u && values == nullptr)
        || (expected_output != 0u && output == nullptr)
        || (paired && expected_output != 0u && second_moment_output == nullptr))
        return error(segment_status_v2::invalid_shape,
            "segment reduction host operand shape is invalid");

    for (std::uint32_t segment = 0u;
         segment < plan.local_segment_count; ++segment) {
        const std::uint64_t begin = offsets[segment];
        const std::uint64_t end = offsets[segment + 1u];
        const std::uint64_t count = end - begin;
        for (std::uint32_t column = 0u; column < plan.dense_width; ++column) {
            double sum = 0.0;
            double squares = 0.0;
            float minimum = std::numeric_limits<float>::infinity();
            float maximum = -std::numeric_limits<float>::infinity();
            bool nan = false;
            for (std::uint64_t index = begin; index < end; ++index) {
                const float value = values[index * plan.dense_width + column];
                nan |= std::isnan(value);
                sum += value;
                squares += static_cast<double>(value)
                    * static_cast<double>(value);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            const std::uint64_t position =
                static_cast<std::uint64_t>(segment)
                    * plan.dense_width + column;
            if (nan) {
                output[position] = std::numeric_limits<float>::quiet_NaN();
                if (paired) second_moment_output[position] = output[position];
                continue;
            }
            switch (plan.reduction) {
                case segment_reduce_kind_v2::sum:
                    output[position] = static_cast<float>(sum); break;
                case segment_reduce_kind_v2::mean:
                    output[position] = count == 0u ? 0.0f
                        : static_cast<float>(sum / static_cast<double>(count));
                    break;
                case segment_reduce_kind_v2::minimum:
                    output[position] = minimum; break;
                case segment_reduce_kind_v2::maximum:
                    output[position] = maximum; break;
                case segment_reduce_kind_v2::sum_of_squares:
                    output[position] = static_cast<float>(squares); break;
                case segment_reduce_kind_v2::first_second_moments:
                    output[position] = count == 0u ? 0.0f
                        : static_cast<float>(sum / static_cast<double>(count));
                    second_moment_output[position] = count == 0u ? 0.0f
                        : static_cast<float>(squares
                            / static_cast<double>(count));
                    break;
            }
        }
    }
    return {};
}

} // namespace cellerator::compute::segment
