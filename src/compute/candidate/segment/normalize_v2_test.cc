#include <Cellerator/compute/candidate/segment/normalize_v2.hh>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using namespace cellerator;

execution::axis_identity axis(std::uint32_t value) {
    return {{value, 1u}, {value + 1u, 1u}, {value + 2u, 1u},
        {value + 3u, 1u}};
}

compute::segment::segment_plan_v2 plan(
    compute::segment::segment_normalize_kind_v2 kind,
    compute::segment::segment_direction_v2 direction) {
    compute::segment::segment_plan_v2 result{};
    result.operation = compute::segment::segment_operation_v2::normalize;
    result.direction = direction;
    result.normalization = kind;
    result.values_axis = axis(1u);
    result.segment_axis = axis(10u);
    result.dense_axis = axis(20u);
    result.partition_identity = 30u;
    result.global_value_count = 6u;
    result.global_segment_count = 4u;
    result.local_value_count = 6u;
    result.local_segment_count = 4u;
    result.dense_width = 2u;
    result.maximum_segment_length = 3u;
    result.epsilon = 1.0e-6f;
    result.operation_identity = 40u;
    result.stage_identity = 50u;
    return result;
}

bool close(float left, float right, float tolerance = 2.0e-5f) {
    return std::abs(left - right) <= tolerance;
}

} // namespace

int main() {
    using namespace cellerator::compute::segment;
    const std::array<std::uint64_t, 5> offsets{{0u, 0u, 1u, 4u, 6u}};
    const std::array<float, 12> values{{
        2.0f, -1.0f,
        0.0f, 2.0f,
        1.0f, 2.0f,
        2.0f, 2.0f,
        -3.0f, 4.0f,
        3.0f, -4.0f}};
    for (const auto kind : std::array<segment_normalize_kind_v2, 6>{{
            segment_normalize_kind_v2::log_sum_exp,
            segment_normalize_kind_v2::softmax,
            segment_normalize_kind_v2::log_softmax,
            segment_normalize_kind_v2::l1,
            segment_normalize_kind_v2::l2,
            segment_normalize_kind_v2::rms}}) {
        auto forward_plan = plan(kind, segment_direction_v2::forward);
        std::array<float, 12> forward{};
        const std::uint64_t output_count =
            kind == segment_normalize_kind_v2::log_sum_exp ? 8u : 12u;
        if (!reference_segment_normalize_forward_v2(forward_plan,
                offsets.data(), offsets.size(), values.data(), values.size(),
                forward.data(), output_count))
            return 1;
        auto backward_plan = forward_plan;
        backward_plan.direction = segment_direction_v2::backward;
        std::array<float, 12> gradient{};
        gradient.fill(1.0f);
        std::array<float, 12> backward{};
        if (!reference_segment_normalize_backward_v2(backward_plan,
                offsets.data(), offsets.size(), values.data(), forward.data(),
                gradient.data(), output_count, backward.data(), backward.size()))
            return 2;
        for (float value : backward)
            if (!std::isfinite(value)) return 3;
    }

    auto softmax_plan = plan(segment_normalize_kind_v2::softmax,
        segment_direction_v2::forward);
    std::array<float, 12> softmax{};
    if (!reference_segment_normalize_forward_v2(softmax_plan,
            offsets.data(), offsets.size(), values.data(), values.size(),
            softmax.data(), softmax.size()))
        return 4;
    if (!close(softmax[0], 1.0f) || !close(softmax[1], 1.0f)) return 5;
    for (std::uint32_t column = 0u; column < 2u; ++column) {
        float sum = 0.0f;
        for (std::uint64_t index = 1u; index < 4u; ++index)
            sum += softmax[index * 2u + column];
        if (!close(sum, 1.0f)) return 6;
    }

    auto nonfinite_values = values;
    nonfinite_values[2] = std::numeric_limits<float>::infinity();
    nonfinite_values[4] = std::numeric_limits<float>::infinity();
    if (!reference_segment_normalize_forward_v2(softmax_plan,
            offsets.data(), offsets.size(), nonfinite_values.data(),
            nonfinite_values.size(), softmax.data(), softmax.size()))
        return 7;
    if (!close(softmax[2], 0.5f) || !close(softmax[4], 0.5f)
        || !close(softmax[6], 0.0f))
        return 8;

    softmax_plan.nan_policy = segment_nonfinite_policy_v2::reject;
    nonfinite_values[2] = std::numeric_limits<float>::quiet_NaN();
    if (validate_segment_normalize_values_v2_host(softmax_plan,
            nonfinite_values.data(), nonfinite_values.size()).code
        != segment_status_v2::nonfinite_input)
        return 9;
    return 0;
}
