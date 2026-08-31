#pragma once

#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/residual_kernels.h"

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {

struct separate_residual_v1 {
    residual_apply_v1 residual{};
    float* residual_buffer = nullptr;
    std::uint64_t residual_stride = 0;
};

struct combine_mma_residual_v1 {
    const float* mma_contribution = nullptr;
    const float* residual_contribution = nullptr;
    float* output = nullptr;
    std::uint64_t row_count = 0;
    std::uint64_t mma_stride = 0;
    std::uint64_t residual_stride = 0;
    std::uint64_t output_stride = 0;
    std::uint32_t width = 0;
    float mma_scale = 1.0F;
    float residual_scale = 1.0F;
    float beta = 1.0F;
};

residual_launch_status launch_separate_buffer_residual_v1(
        const separate_residual_v1& apply,
        void* caller_stream) noexcept;

// The caller explicitly orders the MMA and residual producer streams before
// launching this combine; no event or device synchronization is hidden here.
residual_launch_status launch_combine_mma_residual_v1(
        const combine_mma_residual_v1& combine,
        void* caller_stream) noexcept;

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
