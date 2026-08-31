#include <Cellerator/compute/candidate/segment/cover_v2.hh>

#include <array>
#include <cstdint>
#include <limits>

namespace {

using namespace cellerator;

execution::axis_identity axis(std::uint32_t value) {
    return {{value, 1u}, {value + 1u, 1u}, {value + 2u, 1u},
        {value + 3u, 1u}};
}

compute::segment::segment_plan_v2 plan() {
    compute::segment::segment_plan_v2 result{};
    result.operation = compute::segment::segment_operation_v2::normalize;
    result.normalization =
        compute::segment::segment_normalize_kind_v2::softmax;
    result.storage_order =
        compute::segment::segment_storage_order_v2::cover_native;
    result.values_axis = axis(1u);
    result.segment_axis = axis(10u);
    result.dense_axis = axis(20u);
    result.partition_identity = 30u;
    result.global_value_count =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())
        + 100u;
    result.global_segment_count = 3u;
    result.component_value_begin = result.global_value_count - 6u;
    result.local_value_count = 6u;
    result.local_segment_count = 3u;
    result.dense_width = 17u;
    result.maximum_segment_length = 3u;
    result.operation_identity = 40u;
    result.stage_identity = 50u;
    return result;
}

} // namespace

int main() {
    using namespace cellerator::compute::segment;
    const auto prepared = plan();
    const std::array<std::uint64_t, 4> offsets{{0u, 1u, 4u, 6u}};
    const std::array<std::uint64_t, 6> mapping{{
        prepared.component_value_begin + 2u,
        prepared.component_value_begin + 0u,
        prepared.component_value_begin + 5u,
        prepared.component_value_begin + 1u,
        prepared.component_value_begin + 4u,
        prepared.component_value_begin + 3u}};
    const std::array<segment_cover_owner_v2, 6> owners{{
        segment_cover_owner_v2::mma,
        segment_cover_owner_v2::residual,
        segment_cover_owner_v2::mma,
        segment_cover_owner_v2::mma,
        segment_cover_owner_v2::residual,
        segment_cover_owner_v2::residual}};
    segment_cover_native_host_view_v2 cover{
        offsets.data(), offsets.size(), mapping.data(), owners.data(),
        mapping.size(), 9u, 3u, 3u};
    std::array<std::uint8_t, 6> marks{};
    segment_cover_validation_receipt_v2 receipt{};
    if (!validate_segment_cover_native_partition_v2_host(prepared, cover,
            {marks.data(), marks.size()}, receipt))
        return 1;
    if (!receipt.exact_disjoint_ownership
        || !receipt.holes_are_non_biological || receipt.physical_holes != 3u)
        return 2;

    auto duplicate = mapping;
    duplicate[5] = duplicate[0];
    cover.local_to_global_value = duplicate.data();
    if (validate_segment_cover_native_partition_v2_host(prepared, cover,
            {marks.data(), marks.size()}, receipt))
        return 3;

    cover.local_to_global_value = mapping.data();
    cover.physical_slot_count = 5u;
    if (validate_segment_cover_native_partition_v2_host(prepared, cover,
            {marks.data(), marks.size()}, receipt))
        return 4;
    return 0;
}
