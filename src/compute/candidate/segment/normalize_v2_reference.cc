#include <Cellerator/compute/candidate/segment/normalize_v2.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cellerator::compute::segment {
namespace {

segment_result_v2 error(segment_status_v2 code, const char *message) noexcept {
    return {code, message};
}

bool multiply_fits(std::uint64_t left, std::uint64_t right,
    std::uint64_t &product) noexcept {
    if (left != 0u
        && right > std::numeric_limits<std::uint64_t>::max() / left)
        return false;
    product = left * right;
    return true;
}

std::uint64_t output_rows(const segment_plan_v2 &plan) noexcept {
    return plan.normalization == segment_normalize_kind_v2::log_sum_exp
        ? plan.local_segment_count : plan.local_value_count;
}

float stable_log_sum_exp(const float *values, std::uint64_t begin,
    std::uint64_t end, std::uint32_t width, std::uint32_t column) noexcept {
    float maximum = -std::numeric_limits<float>::infinity();
    std::uint64_t positive_infinities = 0u;
    for (std::uint64_t index = begin; index < end; ++index) {
        const float value = values[index * width + column];
        if (std::isnan(value)) return std::numeric_limits<float>::quiet_NaN();
        if (std::isinf(value) && !std::signbit(value)) ++positive_infinities;
        maximum = std::max(maximum, value);
    }
    if (positive_infinities != 0u)
        return std::numeric_limits<float>::infinity();
    if (begin == end || (std::isinf(maximum) && std::signbit(maximum)))
        return -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    for (std::uint64_t index = begin; index < end; ++index)
        sum += std::exp(static_cast<double>(
            values[index * width + column] - maximum));
    return maximum + static_cast<float>(std::log(sum));
}

float norm_denominator(const segment_plan_v2 &plan, const float *values,
    std::uint64_t begin, std::uint64_t end,
    std::uint32_t column) noexcept {
    double accumulator = 0.0;
    for (std::uint64_t index = begin; index < end; ++index) {
        const float value = values[index * plan.dense_width + column];
        if (plan.normalization == segment_normalize_kind_v2::l1)
            accumulator += std::abs(static_cast<double>(value));
        else
            accumulator += static_cast<double>(value)
                * static_cast<double>(value);
    }
    if (plan.normalization == segment_normalize_kind_v2::l1)
        return static_cast<float>(accumulator) + plan.epsilon;
    if (plan.normalization == segment_normalize_kind_v2::rms) {
        const std::uint64_t count = end - begin;
        if (count != 0u) accumulator /= static_cast<double>(count);
    }
    return static_cast<float>(std::sqrt(accumulator + plan.epsilon));
}

} // namespace

segment_result_v2 validate_segment_normalize_values_v2_host(
    const segment_plan_v2 &plan,
    const float *values,
    std::uint64_t element_count) noexcept {
    const segment_result_v2 valid = validate_segment_plan_v2(plan);
    if (!valid) return valid;
    if (plan.operation != segment_operation_v2::normalize)
        return error(segment_status_v2::invalid_argument,
            "normalization requires a normalize plan");
    std::uint64_t expected = 0u;
    if (!multiply_fits(plan.local_value_count, plan.dense_width, expected)
        || element_count != expected || (expected != 0u && values == nullptr))
        return error(segment_status_v2::invalid_shape,
            "normalization host value shape is invalid");
    for (std::uint64_t index = 0u; index < element_count; ++index) {
        if ((plan.nan_policy == segment_nonfinite_policy_v2::reject
                && std::isnan(values[index]))
            || (plan.infinity_policy == segment_nonfinite_policy_v2::reject
                && std::isinf(values[index])))
            return error(segment_status_v2::nonfinite_input,
                "normalization rejected nonfinite host input");
    }
    return {};
}

segment_result_v2 reference_segment_normalize_forward_v2(
    const segment_plan_v2 &plan,
    const std::uint64_t *offsets,
    std::uint64_t offset_count,
    const float *values,
    std::uint64_t value_element_count,
    float *output,
    std::uint64_t output_element_count) noexcept {
    const segment_result_v2 values_valid =
        validate_segment_normalize_values_v2_host(
            plan, values, value_element_count);
    if (!values_valid) return values_valid;
    if (plan.direction != segment_direction_v2::forward)
        return error(segment_status_v2::invalid_argument,
            "normalization forward requires a forward plan");
    const segment_result_v2 partition_valid =
        validate_segment_partition_offsets_v2_host(plan, offsets, offset_count);
    if (!partition_valid) return partition_valid;
    std::uint64_t expected_output = 0u;
    if (!multiply_fits(output_rows(plan), plan.dense_width, expected_output)
        || output_element_count != expected_output
        || (expected_output != 0u && output == nullptr))
        return error(segment_status_v2::invalid_shape,
            "normalization forward output shape is invalid");

    for (std::uint32_t segment = 0u;
         segment < plan.local_segment_count; ++segment) {
        const std::uint64_t begin = offsets[segment];
        const std::uint64_t end = offsets[segment + 1u];
        const std::uint64_t count = end - begin;
        for (std::uint32_t column = 0u; column < plan.dense_width; ++column) {
            const float log_sum_exp = stable_log_sum_exp(
                values, begin, end, plan.dense_width, column);
            if (plan.normalization == segment_normalize_kind_v2::log_sum_exp) {
                output[static_cast<std::uint64_t>(segment)
                    * plan.dense_width + column] = log_sum_exp;
                continue;
            }
            if (plan.normalization == segment_normalize_kind_v2::softmax
                || plan.normalization
                    == segment_normalize_kind_v2::log_softmax) {
                std::uint64_t positive_infinities = 0u;
                for (std::uint64_t index = begin; index < end; ++index) {
                    const float value = values[index * plan.dense_width + column];
                    positive_infinities += static_cast<std::uint64_t>(
                        std::isinf(value) && !std::signbit(value));
                }
                for (std::uint64_t index = begin; index < end; ++index) {
                    const std::uint64_t position =
                        index * plan.dense_width + column;
                    if (std::isnan(log_sum_exp)) {
                        output[position] =
                            std::numeric_limits<float>::quiet_NaN();
                    } else if (positive_infinities != 0u) {
                        const bool positive_infinity = std::isinf(values[position])
                            && !std::signbit(values[position]);
                        if (plan.normalization
                                == segment_normalize_kind_v2::softmax)
                            output[position] = positive_infinity
                                ? 1.0f / static_cast<float>(positive_infinities)
                                : 0.0f;
                        else
                            output[position] = positive_infinity
                                ? -std::log(static_cast<float>(positive_infinities))
                                : -std::numeric_limits<float>::infinity();
                    } else if (std::isinf(log_sum_exp)
                        && std::signbit(log_sum_exp)) {
                        const float mass = count == 0u ? 0.0f
                            : 1.0f / static_cast<float>(count);
                        output[position] = plan.normalization
                                == segment_normalize_kind_v2::softmax
                            ? mass : std::log(mass);
                    } else {
                        const float log_value = values[position] - log_sum_exp;
                        output[position] = plan.normalization
                                == segment_normalize_kind_v2::softmax
                            ? std::exp(log_value) : log_value;
                    }
                }
                continue;
            }
            const float denominator = norm_denominator(
                plan, values, begin, end, column);
            for (std::uint64_t index = begin; index < end; ++index) {
                const std::uint64_t position =
                    index * plan.dense_width + column;
                output[position] = denominator == 0.0f
                    ? 0.0f : values[position] / denominator;
            }
        }
    }
    return {};
}

segment_result_v2 reference_segment_normalize_backward_v2(
    const segment_plan_v2 &plan,
    const std::uint64_t *offsets,
    std::uint64_t offset_count,
    const float *values,
    const float *forward_output,
    const float *output_gradient,
    std::uint64_t output_element_count,
    float *input_gradient,
    std::uint64_t input_gradient_element_count) noexcept {
    std::uint64_t value_elements = 0u;
    if (!multiply_fits(plan.local_value_count, plan.dense_width,
            value_elements))
        return error(segment_status_v2::invalid_shape,
            "normalization backward value shape overflows");
    const segment_result_v2 values_valid =
        validate_segment_normalize_values_v2_host(
            plan, values, value_elements);
    if (!values_valid) return values_valid;
    if (plan.direction != segment_direction_v2::backward)
        return error(segment_status_v2::invalid_argument,
            "normalization backward requires a backward plan");
    const segment_result_v2 partition_valid =
        validate_segment_partition_offsets_v2_host(plan, offsets, offset_count);
    if (!partition_valid) return partition_valid;
    std::uint64_t expected_output = 0u;
    std::uint64_t expected_input = 0u;
    if (!multiply_fits(output_rows(plan), plan.dense_width, expected_output)
        || !multiply_fits(plan.local_value_count, plan.dense_width,
            expected_input)
        || output_element_count != expected_output
        || input_gradient_element_count != expected_input
        || (expected_output != 0u
            && (forward_output == nullptr || output_gradient == nullptr))
        || (expected_input != 0u && input_gradient == nullptr))
        return error(segment_status_v2::invalid_shape,
            "normalization backward operand shape is invalid");

    for (std::uint32_t segment = 0u;
         segment < plan.local_segment_count; ++segment) {
        const std::uint64_t begin = offsets[segment];
        const std::uint64_t end = offsets[segment + 1u];
        const std::uint64_t count = end - begin;
        for (std::uint32_t column = 0u; column < plan.dense_width; ++column) {
            if (plan.normalization == segment_normalize_kind_v2::log_sum_exp) {
                const std::uint64_t summary =
                    static_cast<std::uint64_t>(segment)
                    * plan.dense_width + column;
                std::uint64_t positive_infinities = 0u;
                if (std::isinf(forward_output[summary])
                    && !std::signbit(forward_output[summary]))
                    for (std::uint64_t index = begin; index < end; ++index) {
                        const float value =
                            values[index * plan.dense_width + column];
                        positive_infinities += static_cast<std::uint64_t>(
                            std::isinf(value) && !std::signbit(value));
                    }
                for (std::uint64_t index = begin; index < end; ++index) {
                    const std::uint64_t position =
                        index * plan.dense_width + column;
                    if (std::isnan(forward_output[summary])) {
                        input_gradient[position] =
                            std::numeric_limits<float>::quiet_NaN();
                    } else if (positive_infinities != 0u) {
                        const bool selected = std::isinf(values[position])
                            && !std::signbit(values[position]);
                        input_gradient[position] = selected
                            ? output_gradient[summary]
                                / static_cast<float>(positive_infinities)
                            : 0.0f;
                    } else if (std::isinf(forward_output[summary])
                        && std::signbit(forward_output[summary])) {
                        input_gradient[position] = count == 0u ? 0.0f
                            : output_gradient[summary]
                                / static_cast<float>(count);
                    } else {
                        input_gradient[position] = output_gradient[summary]
                            * std::exp(values[position]
                                - forward_output[summary]);
                    }
                }
                continue;
            }
            double dot = 0.0;
            double gradient_sum = 0.0;
            for (std::uint64_t index = begin; index < end; ++index) {
                const std::uint64_t position =
                    index * plan.dense_width + column;
                dot += static_cast<double>(output_gradient[position])
                    * values[position];
                gradient_sum += output_gradient[position];
                if (plan.normalization == segment_normalize_kind_v2::softmax)
                    dot += static_cast<double>(output_gradient[position])
                        * (forward_output[position] - values[position]);
            }
            if (plan.normalization == segment_normalize_kind_v2::softmax) {
                dot = 0.0;
                for (std::uint64_t index = begin; index < end; ++index) {
                    const std::uint64_t position =
                        index * plan.dense_width + column;
                    dot += static_cast<double>(output_gradient[position])
                        * forward_output[position];
                }
            }
            const float denominator = norm_denominator(
                plan, values, begin, end, column);
            for (std::uint64_t index = begin; index < end; ++index) {
                const std::uint64_t position =
                    index * plan.dense_width + column;
                if (plan.normalization == segment_normalize_kind_v2::softmax) {
                    input_gradient[position] = forward_output[position]
                        * (output_gradient[position] - static_cast<float>(dot));
                } else if (plan.normalization
                    == segment_normalize_kind_v2::log_softmax) {
                    input_gradient[position] = output_gradient[position]
                        - std::exp(forward_output[position])
                            * static_cast<float>(gradient_sum);
                } else if (denominator == 0.0f) {
                    input_gradient[position] = 0.0f;
                } else if (plan.normalization
                    == segment_normalize_kind_v2::l1) {
                    const float sign = values[position] > 0.0f ? 1.0f
                        : (values[position] < 0.0f ? -1.0f : 0.0f);
                    input_gradient[position] = output_gradient[position]
                            / denominator
                        - sign * static_cast<float>(dot)
                            / (denominator * denominator);
                } else {
                    const float scale = plan.normalization
                            == segment_normalize_kind_v2::rms && count != 0u
                        ? static_cast<float>(count) : 1.0f;
                    input_gradient[position] = output_gradient[position]
                            / denominator
                        - values[position] * static_cast<float>(dot)
                            / (scale * denominator * denominator * denominator);
                }
            }
        }
    }
    return {};
}

} // namespace cellerator::compute::segment
