#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_n16_n32_v1.cuh>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

struct apply_wide_panels_request_v1 {
    compact_apply_component_v1 component{};
    std::uint32_t panel_begin = 0u;
    std::uint32_t panel_count = 0u;
    std::uint64_t global_panel_base = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

struct apply_wide_panels_shape_v1 {
    std::uint32_t grid_x = 0u;
    std::uint32_t grid_y = 0u;
    std::uint32_t block_x = 0u;
    std::uint32_t columns_per_panel = 16u;
};

apply_launch_status_v1 validate_apply_wide_panels_v1(
    const apply_wide_panels_request_v1 &request,
    apply_wide_panels_shape_v1 *shape) noexcept;

apply_launch_status_v1 enqueue_apply_wide_panels_v1(
    const apply_wide_panels_request_v1 &request) noexcept;

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
