#pragma once

#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/residual_kernels.h"

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {

struct pinned_feature_major_residual_v1 {
    const std::uint64_t* row_offsets = nullptr;
    const std::uint64_t* column_indices = nullptr;
    const float* values = nullptr;
    const float* feature_major_input = nullptr;
    float* output = nullptr;
    const std::uint64_t* rows = nullptr;
    std::uint64_t row_count = 0;
    std::uint64_t edge_count = 0;
    std::uint64_t input_row_count = 0;
    std::uint64_t feature_stride = 0;
    std::uint64_t output_stride = 0;
    std::uint64_t input_order_id = 0;
    std::uint64_t value_generation = 0;
    std::uint32_t width = 0;
    float alpha = 1.0F;
    float beta = 1.0F;
};

residual_launch_status launch_pinned_feature_major_residual_v1(
        const pinned_feature_major_residual_v1& apply,
        void* caller_stream) noexcept;

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
