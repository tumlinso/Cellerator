#include "Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_ptx_m8n8k4_experiment_v1.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {
namespace {

__global__ void apply_ptx_m8n8k4_probe_kernel_v1(
    const std::uint32_t *packed_a,
    const std::uint32_t *packed_b,
    float *accumulators,
    std::uint32_t warp_count) {
    const std::uint32_t warp = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (warp >= warp_count || lane >= 32u) {
        return;
    }
    const std::size_t packed_offset =
        (static_cast<std::size_t>(warp) * 32u + lane) * 2u;
    const std::uint32_t a0 = packed_a[packed_offset];
    const std::uint32_t a1 = packed_a[packed_offset + 1u];
    const std::uint32_t b0 = packed_b[packed_offset];
    const std::uint32_t b1 = packed_b[packed_offset + 1u];
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    float d4 = 0.0f;
    float d5 = 0.0f;
    float d6 = 0.0f;
    float d7 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3),
          "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    const std::size_t output_offset =
        (static_cast<std::size_t>(warp) * 32u + lane) * 8u;
    accumulators[output_offset] = d0;
    accumulators[output_offset + 1u] = d1;
    accumulators[output_offset + 2u] = d2;
    accumulators[output_offset + 3u] = d3;
    accumulators[output_offset + 4u] = d4;
    accumulators[output_offset + 5u] = d5;
    accumulators[output_offset + 6u] = d6;
    accumulators[output_offset + 7u] = d7;
}

}  // namespace

apply_launch_status_v1 validate_apply_ptx_m8n8k4_experiment_v1(
    const apply_ptx_m8n8k4_request_v1 &request) noexcept {
    return request.packed_a != nullptr && request.packed_b != nullptr
            && request.accumulators != nullptr && request.warp_count != 0u
            && request.profiler_correlation_id != 0u
        ? apply_launch_status_v1::success
        : apply_launch_status_v1::invalid_argument;
}

apply_launch_status_v1 enqueue_apply_ptx_m8n8k4_experiment_v1(
    const apply_ptx_m8n8k4_request_v1 &request) noexcept {
    const apply_launch_status_v1 status =
        validate_apply_ptx_m8n8k4_experiment_v1(request);
    if (status != apply_launch_status_v1::success) {
        return status;
    }
    apply_ptx_m8n8k4_probe_kernel_v1<<<request.warp_count, 32u, 0u,
        request.stream>>>(request.packed_a, request.packed_b,
        request.accumulators, request.warp_count);
    return cudaGetLastError() == cudaSuccess
        ? apply_launch_status_v1::success
        : apply_launch_status_v1::cuda_failure;
}

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
