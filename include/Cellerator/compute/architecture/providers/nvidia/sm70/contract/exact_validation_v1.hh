#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {

struct exact_validation_request_v1 {
    const edge_ref_v1 *edges = nullptr;
    std::uint64_t global_edge_begin = 0u;
    std::uint32_t local_edge_count = 0u;
    const float *source = nullptr;
    std::uint32_t source_count = 0u;
    const float *destination = nullptr;
    std::uint32_t destination_count = 0u;
    std::uint32_t dense_width = 0u;
    const float *candidate = nullptr;
    std::uint32_t candidate_count = 0u;
    output_order_v1 candidate_order = output_order_v1::logical_edge;
    double absolute_tolerance = 0.0;
    double relative_tolerance = 0.0;
};

struct exact_validation_result_v1 {
    std::uint64_t checked_edge_count = 0u;
    std::uint64_t first_failing_global_edge = 0u;
    double maximum_absolute_error = 0.0;
    double maximum_relative_error = 0.0;
    bool exact_match = false;
    bool within_tolerance = false;
};

status_v1 validate_exact_contraction_v1(
    const exact_validation_request_v1 &request,
    exact_validation_result_v1 *result) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
