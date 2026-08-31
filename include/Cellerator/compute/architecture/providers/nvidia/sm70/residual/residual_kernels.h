#pragma once

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {

struct residual_apply_v1 {
    const std::uint64_t* row_offsets = nullptr;
    const std::uint64_t* column_indices = nullptr;
    const float* values = nullptr;
    const float* dense_input = nullptr;
    float* output = nullptr;
    const std::uint64_t* rows = nullptr;
    std::uint64_t row_count = 0;
    std::uint64_t structure_row_count = 0;
    std::uint64_t edge_count = 0;
    std::uint64_t dense_input_rows = 0;
    std::uint64_t input_stride = 0;
    std::uint64_t output_stride = 0;
    std::uint32_t width = 0;
    float alpha = 1.0F;
    float beta = 1.0F;
};

enum class residual_launch_status : std::uint32_t {
    success = 0, invalid_argument, invalid_extent, launch_failed
};

residual_launch_status launch_warp_per_row_residual_v1(
        const residual_apply_v1& apply, void* caller_stream) noexcept;

residual_launch_status launch_cta_high_degree_residual_v1(
        const residual_apply_v1& apply, void* caller_stream) noexcept;

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
