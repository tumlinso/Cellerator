#pragma once

#include <Cellerator/tensor/base/compressed.hh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <cstdint>

#include <cuda_runtime.h>

namespace cellerator::compute::sparse {

struct column_moments_view {
    unsigned int columns;
    float *sum;
    float *sum_of_squares;
    float *nonzero_count;
    float *active_row_count;
};

int zero_column_moments(column_moments_view *moments, cudaStream_t stream = nullptr);

int accumulate_column_moments(
    const ::cellerator::tensor::base::device::BlockedEllView *matrix,
    const std::uint8_t *row_mask,
    column_moments_view *moments,
    cudaStream_t stream = nullptr);

int accumulate_column_moments(
    const ::cellerator::tensor::base::device::SlicedEllView *matrix,
    const std::uint8_t *row_mask,
    column_moments_view *moments,
    cudaStream_t stream = nullptr);

// CSR over device arrays retained through completion on stream.
int accumulate_column_moments(
    const tensor::base::CompressedView<uint32_t, const __half> *matrix,
    const std::uint8_t *row_mask,
    column_moments_view *moments,
    cudaStream_t stream = nullptr);

} // namespace cellerator::compute::sparse
