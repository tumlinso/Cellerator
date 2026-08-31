#include <Cellerator/compute/candidate/segment/segment_v2.hh>

#include <array>
#include <cstdint>
#include <limits>

namespace {

using namespace cellerator;

execution::axis_identity axis(std::uint64_t value) {
    execution::axis_identity result{};
    const auto compact = static_cast<std::uint32_t>(value);
    result.domain = {compact, compact + 1u};
    result.order = {compact + 2u, compact + 3u};
    result.geometry = {compact + 4u, compact + 5u};
    result.partition = {compact + 6u, compact + 7u};
    return result;
}

compute::segment::segment_plan_v2 valid_plan() {
    compute::segment::segment_plan_v2 plan{};
    plan.values_axis = axis(10u);
    plan.segment_axis = axis(20u);
    plan.dense_axis = axis(30u);
    plan.partition_identity = 40u;
    plan.global_value_count =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())
        + 4096u;
    plan.global_segment_count =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())
        + 32u;
    plan.component_value_begin = plan.global_value_count - 6u;
    plan.component_segment_begin = plan.global_segment_count - 3u;
    plan.local_value_count = 6u;
    plan.local_segment_count = 3u;
    plan.dense_width = 33u;
    plan.maximum_segment_length = 3u;
    plan.local_index_width =
        compute::segment::segment_local_index_width_v2::u16;
    plan.operation_identity = 50u;
    plan.stage_identity = 60u;
    return plan;
}

} // namespace

int main() {
    using namespace cellerator::compute::segment;
    auto plan = valid_plan();
    const std::array<std::uint64_t, 4> offsets{{0u, 1u, 4u, 6u}};
    if (!validate_segment_plan_v2(plan)
        || !validate_segment_partition_offsets_v2_host(
            plan, offsets.data(), offsets.size()))
        return 1;

    plan.reduction = segment_reduce_kind_v2::first_second_moments;
    if (segment_output_planes_v2(plan) != 2u) return 2;

    auto invalid = plan;
    invalid.local_index_width = segment_local_index_width_v2::u16;
    invalid.local_value_count =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint16_t>::max())
        + 1u;
    invalid.component_value_begin = 0u;
    if (validate_segment_plan_v2(invalid)) return 3;

    invalid = plan;
    invalid.stage_identity = 0u;
    if (validate_segment_plan_v2(invalid).code
        != segment_status_v2::invalid_identity)
        return 4;

    invalid = plan;
    invalid.maximum_segment_length = 2u;
    if (validate_segment_partition_offsets_v2_host(
            invalid, offsets.data(), offsets.size()).code
        != segment_status_v2::invalid_partition)
        return 5;

    invalid = plan;
    invalid.operation = segment_operation_v2::normalize;
    invalid.normalization = segment_normalize_kind_v2::rms;
    invalid.epsilon = 0.0f;
    if (validate_segment_plan_v2(invalid).code
        != segment_status_v2::invalid_argument)
        return 6;

    return 0;
}
