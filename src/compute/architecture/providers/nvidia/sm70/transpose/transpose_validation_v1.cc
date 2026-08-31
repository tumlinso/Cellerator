#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_validation_v1.hh>

#include <cmath>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {

transpose_status_v1 validate_transpose_exact_v1(
    const transpose_validation_request_v1 &request,
    float *reference_workspace,
    std::uint64_t reference_workspace_count,
    transpose_validation_report_v1 *report) noexcept {
    if (report == nullptr || request.candidate_source_gradient == nullptr
        || reference_workspace == nullptr || request.absolute_tolerance < 0.0f
        || !std::isfinite(request.absolute_tolerance)
        || request.reference.cover.owner_count
            > std::numeric_limits<std::uint64_t>::max()
                / request.reference.dense_width)
        return transpose_status_v1::invalid_argument;
    const std::uint64_t output_count = request.reference.cover.owner_count
        * request.reference.dense_width;
    if (output_count == 0u || reference_workspace_count < output_count
        || request.candidate_source_gradient_count < output_count)
        return transpose_status_v1::insufficient_capacity;

    transpose_reference_request_v1 reference = request.reference;
    reference.source_gradient = reference_workspace;
    reference.source_gradient_count = reference_workspace_count;
    const transpose_status_v1 status = execute_transpose_reference_v1(reference);
    if (status != transpose_status_v1::success) return status;

    *report = {};
    report->first_mismatch = ~std::uint64_t{0u};
    report->compared_outputs = output_count;
    report->visited_owner_segments = request.reference.cover.owner_count;
    report->visited_edges = request.reference.cover.placement_count;
    for (std::uint64_t index = 0u; index < output_count; ++index) {
        const float difference = std::fabs(reference_workspace[index]
            - request.candidate_source_gradient[index]);
        if (!std::isfinite(difference)) {
            report->maximum_absolute_error = difference;
            report->first_mismatch = index;
            return transpose_status_v1::invalid_argument;
        }
        if (difference > report->maximum_absolute_error)
            report->maximum_absolute_error = difference;
        if (difference > request.absolute_tolerance
            && report->first_mismatch == ~std::uint64_t{0u})
            report->first_mismatch = index;
    }
    return report->first_mismatch == ~std::uint64_t{0u}
        ? transpose_status_v1::success
        : transpose_status_v1::invalid_cover;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
