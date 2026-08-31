#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_candidates_v1.hh>

#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {
namespace {

constexpr transpose_candidate_v1 candidates[]{
    {0x5453504152534501u, 0x5453504152535301u, 0x5453504152534b01u,
        "sm70.transpose.sparse.source_owner",
        transpose_candidate_kind_v1::sparse_source_owner, 1u,
        std::numeric_limits<std::uint32_t>::max(), false, false, true, 0u},
    {0x544d4d4131360001u, 0x544d4d4131365301u, 0x544d4d4131364b01u,
        "sm70.transpose.mma16.source_owner",
        transpose_candidate_kind_v1::mma16_source_owner, 16u,
        std::numeric_limits<std::uint32_t>::max(), true, true, true, 0u},
};

} // namespace

transpose_candidate_catalog_v1 query_transpose_candidates_v1() noexcept {
    return {candidates, sizeof(candidates) / sizeof(candidates[0])};
}

transpose_status_v1 validate_transpose_candidate_v1(
    const transpose_candidate_v1 &candidate) noexcept {
    if (candidate.candidate_id == 0u || candidate.stage_id == 0u
        || candidate.kernel_id == 0u || candidate.stable_name == nullptr
        || candidate.stable_name[0] == '\0' || candidate.width_min == 0u
        || candidate.width_min > candidate.width_max
        || !candidate.requires_measurement)
        return transpose_status_v1::invalid_argument;
    if (candidate.kind == transpose_candidate_kind_v1::sparse_source_owner)
        return !candidate.requires_full_mma_groups && !candidate.experimental
            ? transpose_status_v1::success
            : transpose_status_v1::invalid_argument;
    if (candidate.kind == transpose_candidate_kind_v1::mma16_source_owner)
        return candidate.width_min == 16u && candidate.requires_full_mma_groups
            && candidate.experimental
            ? transpose_status_v1::success
            : transpose_status_v1::invalid_argument;
    return transpose_status_v1::invalid_argument;
}

transpose_status_v1 execute_transpose_reference_v1(
    const transpose_reference_request_v1 &request) noexcept {
    if (validate_transpose_cover_v1(request.cover)
            != transpose_status_v1::success
        || request.projection_values == nullptr
        || request.destination_gradient == nullptr
        || request.local_destination_count == 0u || request.dense_width == 0u
        || request.source_gradient == nullptr
        || request.cover.owner_count > std::numeric_limits<std::uint64_t>::max()
            / request.dense_width
        || request.source_gradient_count
            < request.cover.owner_count * request.dense_width)
        return transpose_status_v1::invalid_argument;

    for (std::uint64_t owner_index = 0u;
        owner_index < request.cover.owner_count; ++owner_index) {
        const source_owner_schedule_v1 &owner = request.cover.owners[owner_index];
        for (std::uint32_t column = 0u; column < request.dense_width; ++column) {
            float total = 0.0f;
            for (std::uint64_t local = 0u; local < owner.placement_count; ++local) {
                const transpose_edge_placement_v1 &edge =
                    request.cover.placements[owner.placement_begin + local];
                if (edge.local_destination_index
                    >= request.local_destination_count)
                    return transpose_status_v1::invalid_cover;
                total += request.projection_values[edge.projection_value_position]
                    * request.destination_gradient[
                        static_cast<std::uint64_t>(edge.local_destination_index)
                            * request.dense_width + column];
            }
            request.source_gradient[owner_index * request.dense_width + column]
                = total;
        }
    }
    return transpose_status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
