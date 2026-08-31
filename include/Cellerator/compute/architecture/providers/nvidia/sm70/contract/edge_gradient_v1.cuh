#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {

struct edge_gradient_binding_v1 {
    support_view_v1 support{};
    const __half *source_activation = nullptr;
    const __half *destination_gradient = nullptr;
    std::uint32_t dense_width = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    output_order_v1 gradient_order = output_order_v1::projection_native;
    float *edge_gradient = nullptr;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_edge_gradient_binding_v1(
    const edge_gradient_binding_v1 &binding) noexcept;

// Writes each biological edge gradient exactly once. Projection-primary mode
// never reconstructs or scans the logical edge array.
status_v1 enqueue_direct_edge_gradient_v1(
    const edge_gradient_binding_v1 &binding,
    sparse_candidate_v1 candidate) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
