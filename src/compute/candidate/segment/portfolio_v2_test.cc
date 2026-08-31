#include <Cellerator/compute/candidate/segment/portfolio_v2.hh>

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
    result.direction = compute::segment::segment_direction_v2::backward;
    result.normalization =
        compute::segment::segment_normalize_kind_v2::log_softmax;
    result.mechanism =
        compute::segment::segment_mechanism_v2::large_segment_cta;
    result.storage_order =
        compute::segment::segment_storage_order_v2::cover_native;
    result.values_axis = axis(1u);
    result.segment_axis = axis(10u);
    result.dense_axis = axis(20u);
    result.partition_identity = 30u;
    result.global_value_count =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())
        + 10u;
    result.global_segment_count = 2u;
    result.component_value_begin = result.global_value_count - 4u;
    result.local_value_count = 4u;
    result.local_segment_count = 2u;
    result.dense_width = 33u;
    result.maximum_segment_length = 3u;
    result.operation_identity = 40u;
    result.stage_identity = 50u;
    return result;
}

} // namespace

int main() {
    using namespace cellerator::compute::segment;
    std::array<segment_candidate_descriptor_v2, 162> candidates{};
    segment_candidate_buffer_v2 buffer{
        candidates.data(), static_cast<std::uint32_t>(candidates.size()), 0u};
    if (segment_candidate_count_v2() != candidates.size()
        || !enumerate_segment_candidates_v2(buffer)
        || buffer.count != candidates.size()
        || !validate_segment_candidate_catalog_v2(
            candidates.data(), buffer.count))
        return 1;
    candidates[161].candidate_identity = candidates[0].candidate_identity;
    if (validate_segment_candidate_catalog_v2(candidates.data(), buffer.count))
        return 2;

    segment_prepared_manifest_v2 manifest{};
    if (!build_segment_prepared_manifest_v2(plan(), 7u, manifest)) return 3;
    if (manifest.logical_values != 4u || manifest.physical_slots != 7u
        || manifest.physical_holes != 3u
        || manifest.useful_interactions != 132u
        || manifest.threads_per_cta != 512u
        || !manifest.requires_measurement || manifest.production_promoted)
        return 4;
    return 0;
}
