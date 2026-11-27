#pragma once

#include <Cellerator/tensor/base/compressed.hh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <cstdint>

#include <cuda_runtime.h>

namespace cellerator::ops::matrix::sparse {

struct masked_row_scale_log1p_params {
    const float *row_sums;
    const std::uint8_t *row_mask;
    float target_sum;
};

int masked_row_scale_log1p_inplace(
    ::cellerator::tensor::base::device::BlockedEllView *matrix,
    const masked_row_scale_log1p_params *params,
    cudaStream_t stream = nullptr);

int masked_row_scale_log1p_inplace(
    ::cellerator::tensor::base::device::SlicedEllView *matrix,
    const masked_row_scale_log1p_params *params,
    cudaStream_t stream = nullptr);

// CSR with writable binary16 payload bytes and aligned, valid device arrays.
// The caller retains storage until the supplied stream completes.
int masked_row_scale_log1p_inplace(
    const ::cellerator::tensor::base::CompressedView<uint32_t, void> *matrix,
    const masked_row_scale_log1p_params *params,
    cudaStream_t stream = nullptr);

int masked_row_scale_log1p_inplace(
    const ::cellerator::tensor::base::CompressedView<int32_t, void> *matrix,
    const masked_row_scale_log1p_params *params,
    cudaStream_t stream = nullptr);

int masked_row_scale_log1p_inplace(
    const ::cellerator::tensor::base::CompressedView<int64_t, void> *matrix,
    const masked_row_scale_log1p_params *params,
    cudaStream_t stream = nullptr);

} // namespace cellerator::ops::matrix::sparse
