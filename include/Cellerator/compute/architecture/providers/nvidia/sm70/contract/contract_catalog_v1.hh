#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {

enum class candidate_kind_v1 : std::uint8_t {
    sparse_thread_per_edge = 0u,
    sparse_warp_per_edge = 1u,
    sparse_cooperative_group = 2u,
    rectangular_mma_exact_residual = 3u
};

struct catalog_entry_v1 {
    std::uint64_t stable_candidate_id = 0u;
    const char *unique_kernel_name = nullptr;
    candidate_kind_v1 kind = candidate_kind_v1::sparse_thread_per_edge;
    std::uint32_t minimum_width = 0u;
    std::uint32_t maximum_width = 0u;
    std::uint16_t tile_rows = 0u;
    std::uint16_t tile_columns = 0u;
    bool exact = true;
    bool supports_logical_output = true;
    bool supports_projection_output = true;
    bool requires_rectangular_cover = false;
    bool requires_cuda_execution_resource = true;
    bool requires_measurement = true;
    bool promoted = false;
};

struct planner_problem_v1 {
    std::uint32_t dense_width = 0u;
    std::uint32_t local_edge_count = 0u;
    std::uint32_t rectangular_tile_count = 0u;
    output_order_v1 required_output_order = output_order_v1::logical_edge;
    bool cuda_execution_resource_available = false;
};

struct planner_candidate_v1 {
    const catalog_entry_v1 *entry = nullptr;
    bool eligible = false;
    bool empirical_measurement_required = true;
};

const catalog_entry_v1 *catalog_v1(std::size_t *count) noexcept;
planner_candidate_v1 evaluate_candidate_v1(const planner_problem_v1 &problem,
    candidate_kind_v1 candidate) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
