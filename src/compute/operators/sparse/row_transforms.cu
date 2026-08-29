#include <Cellerator/compute/operators/sparse/row_transforms.cuh>

#include <cuda_fp16.h>

namespace cellerator::compute::sparse {
namespace {

__device__ inline __half scaled_log1p(__half input, float sum, float target) {
    const float value = __half2float(input);
    const float scale = sum > 0.0f ? target / sum : 0.0f;
    return value != 0.0f ? __float2half(log1pf(value * scale)) : __float2half(0.0f);
}

__global__ void blocked_ell_kernel(matrix::device::blocked_ell_view matrix,
                                   masked_row_scale_log1p_params params) {
    const unsigned long tid = static_cast<unsigned long>(blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned long stride = static_cast<unsigned long>(gridDim.x * blockDim.x);
    const unsigned long count = static_cast<unsigned long>(matrix.rows) * matrix.ell_cols;
    for (unsigned long index = tid; index < count; index += stride) {
        const unsigned int row = static_cast<unsigned int>(index / matrix.ell_cols);
        if (params.row_mask == nullptr || params.row_mask[row] != 0u) {
            matrix.val[index] = scaled_log1p(matrix.val[index], params.row_sums[row], params.target_sum);
        }
    }
}

__global__ void sliced_ell_kernel(matrix::device::sliced_ell_view matrix,
                                  masked_row_scale_log1p_params params) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    const unsigned int warp_stride = gridDim.x * (blockDim.x >> 5u);
    for (unsigned int row = warp; row < matrix.rows; row += warp_stride) {
        if (params.row_mask != nullptr && params.row_mask[row] == 0u) continue;
        unsigned int slice = matrix.slice_rows != 0u ? row / matrix.slice_rows : 0u;
        while (slice + 1u < matrix.slice_count && row >= matrix.slice_row_offsets[slice + 1u]) ++slice;
        if (slice >= matrix.slice_count) continue;
        const unsigned int width = matrix.slice_widths[slice];
        const unsigned long base = static_cast<unsigned long>(matrix.slice_slot_offsets[slice])
            + static_cast<unsigned long>(row - matrix.slice_row_offsets[slice]) * width;
        for (unsigned int slot = lane; slot < width; slot += 32u) {
            const unsigned long index = base + slot;
            if (matrix.col_idx[index] < matrix.cols) {
                matrix.val[index] = scaled_log1p(matrix.val[index], params.row_sums[row], params.target_sum);
            }
        }
    }
}

__global__ void compressed_kernel(matrix::device::compressed_view matrix,
                                  masked_row_scale_log1p_params params) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    const unsigned int warp_stride = gridDim.x * (blockDim.x >> 5u);
    for (unsigned int row = warp; row < matrix.rows; row += warp_stride) {
        if (params.row_mask != nullptr && params.row_mask[row] == 0u) continue;
        const unsigned int end = matrix.major_ptr[row + 1u];
        for (unsigned int index = matrix.major_ptr[row] + lane; index < end; index += 32u) {
            matrix.val[index] = scaled_log1p(matrix.val[index], params.row_sums[row], params.target_sum);
        }
    }
}

unsigned int row_blocks(unsigned int rows) {
    unsigned int blocks = (rows + 7u) >> 3u;
    if (blocks < 1u) blocks = 1u;
    return blocks > 4096u ? 4096u : blocks;
}

} // namespace

int masked_row_scale_log1p_inplace(matrix::device::blocked_ell_view *matrix,
                                   const masked_row_scale_log1p_params *params,
                                   cudaStream_t stream) {
    if (matrix == nullptr || params == nullptr || params->row_sums == nullptr || params->target_sum < 0.0f) return 0;
    const unsigned long count = static_cast<unsigned long>(matrix->rows) * matrix->ell_cols;
    unsigned int blocks = static_cast<unsigned int>((count + 255u) >> 8u);
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    blocked_ell_kernel<<<blocks, 256, 0, stream>>>(*matrix, *params);
    return cudaGetLastError() == cudaSuccess;
}

int masked_row_scale_log1p_inplace(matrix::device::sliced_ell_view *matrix,
                                   const masked_row_scale_log1p_params *params,
                                   cudaStream_t stream) {
    if (matrix == nullptr || params == nullptr || params->row_sums == nullptr || params->target_sum < 0.0f) return 0;
    sliced_ell_kernel<<<row_blocks(matrix->rows), 256, 0, stream>>>(*matrix, *params);
    return cudaGetLastError() == cudaSuccess;
}

int masked_row_scale_log1p_inplace(matrix::device::compressed_view *matrix,
                                   const masked_row_scale_log1p_params *params,
                                   cudaStream_t stream) {
    if (matrix == nullptr || params == nullptr || params->row_sums == nullptr || params->target_sum < 0.0f
        || matrix->axis != ::cellerator::matrix::device::compressed_by_row) return 0;
    compressed_kernel<<<row_blocks(matrix->rows), 256, 0, stream>>>(*matrix, *params);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace cellerator::compute::sparse
