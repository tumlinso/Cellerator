#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_candidates_v1.hh>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {

struct transpose_validation_request_v1 {
    transpose_reference_request_v1 reference{};
    const float *candidate_source_gradient = nullptr;
    std::uint64_t candidate_source_gradient_count = 0u;
    float absolute_tolerance = 0.0f;
};

struct transpose_validation_report_v1 {
    std::uint64_t compared_outputs = 0u;
    std::uint64_t visited_edges = 0u;
    std::uint64_t visited_owner_segments = 0u;
    float maximum_absolute_error = 0.0f;
    std::uint64_t first_mismatch = ~std::uint64_t{0u};
};

transpose_status_v1 validate_transpose_exact_v1(
    const transpose_validation_request_v1 &request,
    float *reference_workspace,
    std::uint64_t reference_workspace_count,
    transpose_validation_report_v1 *report) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
