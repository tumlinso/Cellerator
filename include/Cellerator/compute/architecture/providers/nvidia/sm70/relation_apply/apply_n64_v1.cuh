#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_n16_n32_v1.cuh>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

enum class apply_n64_variant_v1 : std::uint8_t {
    direct_global = 1u,
    shared_a = 2u,
    software_pipeline = 3u,
};

struct apply_n64_request_v1 {
    compact_apply_component_v1 component{};
    apply_n64_variant_v1 variant = apply_n64_variant_v1::direct_global;
    std::uint8_t reserved[7]{};
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

struct apply_n64_launch_shape_v1 {
    std::uint32_t grid_x = 0u;
    std::uint32_t block_x = 0u;
    std::uint32_t dynamic_shared_bytes = 0u;
    std::uint32_t output_owner_warps = 0u;
};

apply_launch_status_v1 validate_apply_n64_v1(
    const apply_n64_request_v1 &request,
    apply_n64_launch_shape_v1 *shape) noexcept;

apply_launch_status_v1 enqueue_apply_n64_v1(
    const apply_n64_request_v1 &request) noexcept;

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
