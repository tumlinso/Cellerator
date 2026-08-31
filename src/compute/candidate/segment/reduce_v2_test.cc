#include <Cellerator/compute/candidate/segment/reduce_v2.hh>

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
    compute::segment::segment_reduce_kind_v2 kind) {
    compute::segment::segment_plan_v2 result{};
    result.reduction = kind;
    result.values_axis = axis(1u);
    result.segment_axis = axis(10u);
    result.dense_axis = axis(20u);
    result.partition_identity = 30u;
    result.global_value_count = 4u;
    result.global_segment_count = 3u;
    result.local_value_count = 4u;
    result.local_segment_count = 3u;
    result.dense_width = 1u;
    result.maximum_segment_length = 3u;
    result.operation_identity = 40u;
    result.stage_identity = 50u;
    return result;
}

bool close(float left, float right) {
    return std::abs(left - right) <= 1.0e-6f;
}

} // namespace

int main() {
    using namespace cellerator::compute::segment;
    const std::array<std::uint64_t, 4> offsets{{0u, 0u, 1u, 4u}};
    const std::array<float, 4> values{{2.0f, -1.0f, 3.0f, 4.0f}};
    for (const auto kind : std::array<segment_reduce_kind_v2, 6>{{
            segment_reduce_kind_v2::sum,
            segment_reduce_kind_v2::mean,
            segment_reduce_kind_v2::minimum,
            segment_reduce_kind_v2::maximum,
            segment_reduce_kind_v2::sum_of_squares,
            segment_reduce_kind_v2::first_second_moments}}) {
        std::array<float, 3> output{};
        std::array<float, 3> second{};
        if (!reference_segment_reduce_v2(plan(kind), offsets.data(),
                offsets.size(), values.data(), values.size(), output.data(),
                second.data(), output.size()))
            return 1;
        if (kind == segment_reduce_kind_v2::sum
            && (!close(output[0], 0.0f) || !close(output[2], 6.0f)))
            return 2;
        if (kind == segment_reduce_kind_v2::mean
            && (!close(output[0], 0.0f) || !close(output[2], 2.0f)))
            return 3;
        if (kind == segment_reduce_kind_v2::minimum
            && (!std::isinf(output[0]) || !close(output[2], -1.0f)))
            return 4;
        if (kind == segment_reduce_kind_v2::maximum
            && (!std::isinf(output[0]) || !std::signbit(output[0])
                || !close(output[2], 4.0f)))
            return 5;
        if (kind == segment_reduce_kind_v2::sum_of_squares
            && !close(output[2], 26.0f))
            return 6;
        if (kind == segment_reduce_kind_v2::first_second_moments
            && (!close(output[2], 2.0f)
                || !close(second[2], 26.0f / 3.0f)))
            return 7;
    }
    return 0;
}
