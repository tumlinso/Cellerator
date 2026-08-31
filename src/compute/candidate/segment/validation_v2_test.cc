#include <Cellerator/compute/candidate/segment/normalize_v2.hh>
#include <Cellerator/compute/candidate/segment/portfolio_v2.hh>
#include <Cellerator/compute/candidate/segment/reduce_v2.hh>

#include <array>
#include <cmath>
#include <cstdint>

namespace {

using namespace cellerator;

constexpr std::uint32_t width = 3u;
constexpr std::uint32_t segment_count = 8u;
constexpr std::uint64_t value_count = 145u;
constexpr std::uint64_t value_elements = value_count * width;
constexpr std::uint64_t segment_elements = segment_count * width;

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
    result.mechanism =
        compute::segment::segment_mechanism_v2::large_segment_cta;
    result.storage_order =
        compute::segment::segment_storage_order_v2::cover_native;
    result.values_axis = axis(1u);
    result.segment_axis = axis(10u);
    result.dense_axis = axis(20u);
    result.partition_identity = 30u;
    result.global_value_count = 0x100000000ULL + value_count;
    result.global_segment_count = 0x100000000ULL + segment_count;
    result.component_value_begin = result.global_value_count - value_count;
    result.component_segment_begin =
        result.global_segment_count - segment_count;
    result.local_value_count = value_count;
    result.local_segment_count = segment_count;
    result.dense_width = width;
    result.maximum_segment_length = 33u;
    result.epsilon = 1.0e-3f;
    result.operation_identity = 40u;
    result.stage_identity = 50u;
    return result;
}

double objective(const float *output, const float *gradient,
    std::uint64_t count) {
    double result = 0.0;
    for (std::uint64_t index = 0u; index < count; ++index)
        result += static_cast<double>(output[index]) * gradient[index];
    return result;
}

} // namespace

int main() {
    using namespace cellerator::compute::segment;
    const std::array<std::uint64_t, segment_count + 1u> offsets{{
        0u, 0u, 1u, 16u, 32u, 49u, 80u, 112u, 145u}};
    std::array<float, value_elements> values{};
    for (std::uint64_t index = 0u; index < values.size(); ++index)
        values[index] = 0.2f
            + static_cast<float>(static_cast<int>(index % 17u) - 8) * 0.07f
            + static_cast<float>(index % 3u) * 0.013f;
    const std::array<std::uint64_t, 5> probes{{0u, 2u, 47u, 241u, 434u}};

    for (const auto kind : std::array<segment_normalize_kind_v2, 6>{{
            segment_normalize_kind_v2::log_sum_exp,
            segment_normalize_kind_v2::softmax,
            segment_normalize_kind_v2::log_softmax,
            segment_normalize_kind_v2::l1,
            segment_normalize_kind_v2::l2,
            segment_normalize_kind_v2::rms}}) {
        const std::uint64_t output_count =
            kind == segment_normalize_kind_v2::log_sum_exp
            ? segment_elements : value_elements;
        auto forward_plan = plan(kind, segment_direction_v2::forward);
        std::array<float, value_elements> forward{};
        if (!reference_segment_normalize_forward_v2(forward_plan,
                offsets.data(), offsets.size(), values.data(), values.size(),
                forward.data(), output_count))
            return 1;
        std::array<float, value_elements> output_gradient{};
        for (std::uint64_t index = 0u; index < output_count; ++index)
            output_gradient[index] = 0.1f
                + static_cast<float>(index % 7u) * 0.03f;
        std::array<float, value_elements> input_gradient{};
        auto backward_plan = forward_plan;
        backward_plan.direction = segment_direction_v2::backward;
        if (!reference_segment_normalize_backward_v2(backward_plan,
                offsets.data(), offsets.size(), values.data(), forward.data(),
                output_gradient.data(), output_count, input_gradient.data(),
                input_gradient.size()))
            return 2;

        constexpr float step = 2.0e-3f;
        for (const std::uint64_t probe : probes) {
            auto plus = values;
            auto minus = values;
            plus[probe] += step;
            minus[probe] -= step;
            std::array<float, value_elements> plus_output{};
            std::array<float, value_elements> minus_output{};
            if (!reference_segment_normalize_forward_v2(forward_plan,
                    offsets.data(), offsets.size(), plus.data(), plus.size(),
                    plus_output.data(), output_count)
                || !reference_segment_normalize_forward_v2(forward_plan,
                    offsets.data(), offsets.size(), minus.data(), minus.size(),
                    minus_output.data(), output_count))
                return 3;
            const double numerical = (objective(plus_output.data(),
                    output_gradient.data(), output_count)
                - objective(minus_output.data(), output_gradient.data(),
                    output_count)) / (2.0 * step);
            if (std::abs(numerical - input_gradient[probe]) > 4.0e-2)
                return 4;
        }
    }

    auto reduction_plan = plan(segment_normalize_kind_v2::softmax,
        segment_direction_v2::forward);
    reduction_plan.operation = segment_operation_v2::reduce;
    reduction_plan.reduction = segment_reduce_kind_v2::first_second_moments;
    std::array<float, segment_elements> first{};
    std::array<float, segment_elements> second{};
    if (!reference_segment_reduce_v2(reduction_plan, offsets.data(),
            offsets.size(), values.data(), values.size(), first.data(),
            second.data(), first.size()))
        return 5;
    if (first[0] != 0.0f || second[0] != 0.0f
        || !std::isfinite(first.back()) || !std::isfinite(second.back()))
        return 6;

    segment_prepared_manifest_v2 manifest{};
    if (!build_segment_prepared_manifest_v2(reduction_plan,
            value_count + 19u, manifest)
        || manifest.physical_holes != 19u
        || manifest.useful_interactions != value_elements)
        return 7;
    return 0;
}
