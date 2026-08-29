#pragma once

#include <Cellerator/matrix/views.cuh>

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
    const matrix::device::blocked_ell_view *matrix,
    const std::uint8_t *row_mask,
    column_moments_view *moments,
    cudaStream_t stream = nullptr);

int accumulate_column_moments(
    const matrix::device::sliced_ell_view *matrix,
    const std::uint8_t *row_mask,
    column_moments_view *moments,
    cudaStream_t stream = nullptr);

int accumulate_column_moments(
    const matrix::device::compressed_view *matrix,
    const std::uint8_t *row_mask,
    column_moments_view *moments,
    cudaStream_t stream = nullptr);

} // namespace cellerator::compute::sparse
