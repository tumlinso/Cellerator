#pragma once

#include <Cellerator/interop/formats/device_views.cuh>

#include <cstdint>

#include <cuda_runtime.h>

namespace cellerator::compute::sparse {

struct masked_row_scale_log1p_params {
    const float *row_sums;
    const std::uint8_t *row_mask;
    float target_sum;
};

int masked_row_scale_log1p_inplace(
    ::cellerator::interop::formats::device::BlockedEllView *matrix,
    const masked_row_scale_log1p_params *params,
    cudaStream_t stream = nullptr);

int masked_row_scale_log1p_inplace(
    ::cellerator::interop::formats::device::SlicedEllView *matrix,
    const masked_row_scale_log1p_params *params,
    cudaStream_t stream = nullptr);

int masked_row_scale_log1p_inplace(
    ::cellerator::interop::formats::device::CompressedView *matrix,
    const masked_row_scale_log1p_params *params,
    cudaStream_t stream = nullptr);

} // namespace cellerator::compute::sparse
