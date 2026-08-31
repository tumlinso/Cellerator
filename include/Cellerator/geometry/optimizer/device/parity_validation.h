#pragma once

#include "Cellerator/geometry/optimizer/device/exact_census.h"
#include "Cellerator/geometry/optimizer/device/proposal_scoring.h"

#include <cstdint>

namespace cellerator::geometry::optimizer::device {

struct optimizer_device_resource_receipt_v1 {
    char receipt_uuid[37]{};
    std::uint32_t compute_major = 0;
    std::uint32_t compute_minor = 0;
    std::uint32_t device_ordinal = 0;
    bool accelerator_lease_held = false;
    bool benchmark_mutex_held = false;
    std::uint16_t reserved = 0;
};

enum class parity_validation_status : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_resource_receipt,
    score_mismatch,
    census_mismatch,
};

struct optimizer_device_parity_report_v1 {
    parity_validation_status status = parity_validation_status::invalid_argument;
    std::uint64_t compared_scores = 0;
    std::uint64_t compared_census = 0;
    std::uint64_t first_mismatch = 0;
};

// Validates external workflow authority only; it never acquires a resource.
// require_timing additionally requires the repository benchmark mutex receipt.
parity_validation_status validate_optimizer_device_receipt_v1(
        const optimizer_device_resource_receipt_v1& receipt,
        bool require_sm70,
        bool require_timing) noexcept;

// Compare after the caller has ordered asynchronous D2H copies on its stream.
// The function allocates nothing and performs no CUDA synchronization.
optimizer_device_parity_report_v1 compare_optimizer_device_results_v1(
        const proposal_score_result_v1* host_scores,
        const proposal_score_result_v1* copied_device_scores,
        std::uint64_t score_count,
        const exact_census_result_v1* host_census,
        const exact_census_result_v1* copied_device_census,
        std::uint64_t census_count) noexcept;

}  // namespace cellerator::geometry::optimizer::device
