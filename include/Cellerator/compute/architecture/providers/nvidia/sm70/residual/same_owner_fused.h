#pragma once

#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/residual_kernels.h"

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {

struct same_owner_mma_residual_v1 {
    residual_apply_v1 residual{};
    const float* mma_contribution = nullptr;
    std::uint64_t mma_stride = 0;
    float mma_scale = 1.0F;
};

residual_launch_status launch_same_owner_mma_residual_v1(
        const same_owner_mma_residual_v1& apply,
        void* caller_stream) noexcept;

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
