#include <Cellerator/compute/operators/sparse/row_transforms.cuh>

#include "compressed_fp16.cuh"
#include "sliced_ell_row.cuh"

namespace cellerator::compute::sparse {
namespace {

__device__ inline __half scaled_log1p(__half input, float sum, float target) {
    const float value = __half2float(input);
    const float scale = sum > 0.0f ? target / sum : 0.0f;
    return value != 0.0f ? __float2half(log1pf(value * scale)) : __float2half(0.0f);
}

__global__ void blocked_ell_kernel(::cellerator::tensor::base::device::BlockedEllView matrix,
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

__global__ void sliced_ell_kernel(::cellerator::tensor::base::device::SlicedEllView matrix,
                                  masked_row_scale_log1p_params params) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    const unsigned int warp_stride = gridDim.x * (blockDim.x >> 5u);
    for (unsigned int row = warp; row < matrix.rows; row += warp_stride) {
        if (params.row_mask != nullptr && params.row_mask[row] == 0u) continue;
        const auto slots = detail::locate_sliced_ell_row(matrix, row);
        for (unsigned int slot = lane; slot < slots.width; slot += 32u) {
            const unsigned long index = slots.slot_offset + slot;
            if (matrix.col_idx[index] < matrix.cols) {
                matrix.val[index] = scaled_log1p(matrix.val[index], params.row_sums[row], params.target_sum);
            }
        }
    }
}

template<class Index, class Offset>
__global__ void compressed_kernel(tensor::base::CompressedView<Index, void> matrix,
                                  masked_row_scale_log1p_params params) {
    const auto &s = matrix.structure;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    const unsigned int warp_stride = gridDim.x * (blockDim.x >> 5u);
    for (unsigned int row = warp; row < uint32_t(s.rows); row += warp_stride) {
        if (params.row_mask != nullptr && params.row_mask[row] == 0u) continue;
        const Offset end = Offset(s.major_ptr[row + 1u]);
        for (Offset i = Offset(s.major_ptr[row]) + lane; i < end; i += 32u) {
            const __half value = scaled_log1p(detail::load_fp16(matrix.values, i), params.row_sums[row], params.target_sum);
            detail::store_fp16(matrix.values, i, value);
        }
    }
}

unsigned int row_blocks(unsigned int rows) {
    unsigned int blocks = (rows + 7u) >> 3u;
    if (blocks < 1u) blocks = 1u;
    return blocks > 4096u ? 4096u : blocks;
}

template<class Index>
int scale_compressed(const tensor::base::CompressedView<Index, void> *matrix,
                     const masked_row_scale_log1p_params *params, cudaStream_t stream) {
    if (matrix == nullptr || params == nullptr || params->row_sums == nullptr || params->target_sum < 0.0f
        || !detail::valid_csr_fp16(*matrix)) return 0;
    if (matrix->structure.nnz == 0) return 1;
    const uint32_t blocks = row_blocks(uint32_t(matrix->structure.rows));
    if constexpr (sizeof(Index) == sizeof(uint32_t)) {
        // The warp's last offset increment must fit in the loop counter.
        if (matrix->structure.nnz <= UINT32_MAX - 31u) {
            compressed_kernel<Index, uint32_t><<<blocks, 256, 0, stream>>>(*matrix, *params);
            return cudaGetLastError() == cudaSuccess;
        }
    }
    compressed_kernel<Index, uint64_t><<<blocks, 256, 0, stream>>>(*matrix, *params);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

int masked_row_scale_log1p_inplace(::cellerator::tensor::base::device::BlockedEllView *matrix,
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

int masked_row_scale_log1p_inplace(::cellerator::tensor::base::device::SlicedEllView *matrix,
                                   const masked_row_scale_log1p_params *params,
                                   cudaStream_t stream) {
    if (matrix == nullptr || params == nullptr || params->row_sums == nullptr || params->target_sum < 0.0f) return 0;
    sliced_ell_kernel<<<row_blocks(matrix->rows), 256, 0, stream>>>(*matrix, *params);
    return cudaGetLastError() == cudaSuccess;
}

int masked_row_scale_log1p_inplace(const tensor::base::CompressedView<uint32_t, void> *matrix,
                                   const masked_row_scale_log1p_params *params,
                                   cudaStream_t stream) {
    return scale_compressed(matrix, params, stream);
}

int masked_row_scale_log1p_inplace(const tensor::base::CompressedView<int32_t, void> *matrix,
                                   const masked_row_scale_log1p_params *params,
                                   cudaStream_t stream) {
    return scale_compressed(matrix, params, stream);
}

int masked_row_scale_log1p_inplace(const tensor::base::CompressedView<int64_t, void> *matrix,
                                   const masked_row_scale_log1p_params *params,
                                   cudaStream_t stream) {
    return scale_compressed(matrix, params, stream);
}

} // namespace cellerator::compute::sparse
