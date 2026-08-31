#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_n16_n32_v1.cuh>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

inline constexpr bool apply_ptx_m8n8k4_experimental_v1 = true;
inline constexpr bool apply_ptx_m8n8k4_requires_measurement_v1 = true;

// Isolated register-level experiment. Each warp consumes two packed f16x2 A
// and B registers per lane and emits eight f32 accumulator registers per lane.
// It is not a promoted relation-apply path or a substitute for WMMA validation.
struct apply_ptx_m8n8k4_request_v1 {
    const std::uint32_t *packed_a = nullptr;
    const std::uint32_t *packed_b = nullptr;
    float *accumulators = nullptr;
    std::uint32_t warp_count = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

apply_launch_status_v1 validate_apply_ptx_m8n8k4_experiment_v1(
    const apply_ptx_m8n8k4_request_v1 &request) noexcept;

apply_launch_status_v1 enqueue_apply_ptx_m8n8k4_experiment_v1(
    const apply_ptx_m8n8k4_request_v1 &request) noexcept;

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
