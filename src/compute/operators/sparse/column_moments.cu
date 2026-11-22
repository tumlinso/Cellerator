#include <Cellerator/compute/operators/sparse/column_moments.cuh>

#include "compressed_fp16.cuh"
#include "sliced_ell_row.cuh"

namespace cellerator::compute::sparse {
namespace {

template<class Column>
__device__ inline void add_value(Column column, float value, column_moments_view moments) {
    if (column >= moments.columns || value == 0.0f) return;
    atomicAdd(moments.sum + column, value);
    atomicAdd(moments.sum_of_squares + column, value * value);
    atomicAdd(moments.nonzero_count + column, 1.0f);
}

__global__ void count_rows_kernel(unsigned int rows, const std::uint8_t *mask, float *count) {
    float local = 0.0f;
    for (unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += gridDim.x * blockDim.x) {
        local += mask == nullptr || mask[row] != 0u ? 1.0f : 0.0f;
    }
    if (local != 0.0f) atomicAdd(count, local);
}

__global__ void blocked_ell_kernel(::cellerator::tensor::base::device::BlockedEllView matrix,
                                   const std::uint8_t *row_mask,
                                   column_moments_view moments) {
    const unsigned long stride = static_cast<unsigned long>(gridDim.x * blockDim.x);
    const unsigned long count = static_cast<unsigned long>(matrix.rows) * matrix.ell_cols;
    const unsigned int width = matrix.block_size != 0u ? matrix.ell_cols / matrix.block_size : 0u;
    for (unsigned long index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride) {
        const unsigned int row = static_cast<unsigned int>(index / matrix.ell_cols);
        if (row_mask != nullptr && row_mask[row] == 0u) continue;
        const unsigned int ell_column = static_cast<unsigned int>(index % matrix.ell_cols);
        const unsigned int slot = matrix.block_size != 0u ? ell_column / matrix.block_size : 0u;
        const unsigned int lane = matrix.block_size != 0u ? ell_column % matrix.block_size : 0u;
        const unsigned int block_column = width != 0u
            ? matrix.blockColIdx[static_cast<unsigned long>(row / matrix.block_size) * width + slot]
            : 0xffffffffu;
        const unsigned int column = block_column != 0xffffffffu
            ? block_column * matrix.block_size + lane
            : matrix.cols;
        add_value(column, __half2float(matrix.val[index]), moments);
    }
}

__global__ void sliced_ell_kernel(::cellerator::tensor::base::device::SlicedEllView matrix,
                                  const std::uint8_t *row_mask,
                                  column_moments_view moments) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    const unsigned int warp_stride = gridDim.x * (blockDim.x >> 5u);
    for (unsigned int row = warp; row < matrix.rows; row += warp_stride) {
        if (row_mask != nullptr && row_mask[row] == 0u) continue;
        const auto slots = detail::locate_sliced_ell_row(matrix, row);
        for (unsigned int slot = lane; slot < slots.width; slot += 32u) {
            const unsigned long index = slots.slot_offset + slot;
            add_value(matrix.col_idx[index], __half2float(matrix.val[index]), moments);
        }
    }
}

template<class Index, class Offset>
__global__ void compressed_kernel(tensor::base::CompressedView<Index, const void> matrix,
                                  const std::uint8_t *row_mask,
                                  column_moments_view moments) {
    const auto &s = matrix.structure;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    const unsigned int warp_stride = gridDim.x * (blockDim.x >> 5u);
    for (unsigned int row = warp; row < uint32_t(s.rows); row += warp_stride) {
        if (row_mask != nullptr && row_mask[row] == 0u) continue;
        const Offset end = Offset(s.major_ptr[row + 1u]);
        for (Offset i = Offset(s.major_ptr[row]) + lane; i < end; i += 32u) {
            add_value(Offset(s.minor_idx[i]), __half2float(detail::load_fp16(matrix.values, i)), moments);
        }
    }
}

unsigned int row_blocks(unsigned int rows) {
    unsigned int blocks = (rows + 7u) >> 3u;
    if (blocks < 1u) blocks = 1u;
    return blocks > 4096u ? 4096u : blocks;
}

int count_rows(unsigned int rows, const std::uint8_t *mask, column_moments_view *moments, cudaStream_t stream) {
    if (moments->active_row_count == nullptr) return 1;
    unsigned int blocks = (rows + 255u) >> 8u;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    count_rows_kernel<<<blocks, 256, 0, stream>>>(rows, mask, moments->active_row_count);
    return cudaGetLastError() == cudaSuccess;
}

bool valid(const column_moments_view *moments) {
    return moments != nullptr && moments->sum != nullptr && moments->sum_of_squares != nullptr
        && moments->nonzero_count != nullptr;
}

template<class Index>
int accumulate_compressed(const tensor::base::CompressedView<Index, const void> *matrix,
                          const uint8_t *row_mask, column_moments_view *moments, cudaStream_t stream) {
    if (matrix == nullptr || !valid(moments) || moments->columns != matrix->structure.cols
        || !detail::valid_csr_fp16(*matrix)) return 0;
    const uint32_t rows = uint32_t(matrix->structure.rows);
    if (matrix->structure.nnz == 0) return count_rows(rows, row_mask, moments, stream);
    const uint32_t blocks = row_blocks(rows);
    if constexpr (sizeof(Index) == sizeof(uint32_t)) {
        // The warp's last offset increment must fit in the loop counter.
        if (matrix->structure.nnz <= UINT32_MAX - 31u) {
            compressed_kernel<Index, uint32_t><<<blocks, 256, 0, stream>>>(*matrix, row_mask, *moments);
            return cudaGetLastError() == cudaSuccess && count_rows(rows, row_mask, moments, stream);
        }
    }
    compressed_kernel<Index, uint64_t><<<blocks, 256, 0, stream>>>(*matrix, row_mask, *moments);
    return cudaGetLastError() == cudaSuccess && count_rows(rows, row_mask, moments, stream);
}

} // namespace

int zero_column_moments(column_moments_view *moments, cudaStream_t stream) {
    if (!valid(moments)) return 0;
    if (cudaMemsetAsync(moments->sum, 0, static_cast<std::size_t>(moments->columns) * sizeof(float), stream) != cudaSuccess) return 0;
    if (cudaMemsetAsync(moments->sum_of_squares, 0, static_cast<std::size_t>(moments->columns) * sizeof(float), stream) != cudaSuccess) return 0;
    if (cudaMemsetAsync(moments->nonzero_count, 0, static_cast<std::size_t>(moments->columns) * sizeof(float), stream) != cudaSuccess) return 0;
    return moments->active_row_count == nullptr
        || cudaMemsetAsync(moments->active_row_count, 0, sizeof(float), stream) == cudaSuccess;
}

int accumulate_column_moments(const ::cellerator::tensor::base::device::BlockedEllView *matrix,
                              const std::uint8_t *row_mask,
                              column_moments_view *moments,
                              cudaStream_t stream) {
    if (matrix == nullptr || !valid(moments) || moments->columns != matrix->cols) return 0;
    const unsigned long count = static_cast<unsigned long>(matrix->rows) * matrix->ell_cols;
    unsigned int blocks = static_cast<unsigned int>((count + 255u) >> 8u);
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    blocked_ell_kernel<<<blocks, 256, 0, stream>>>(*matrix, row_mask, *moments);
    return cudaGetLastError() == cudaSuccess && count_rows(matrix->rows, row_mask, moments, stream);
}

int accumulate_column_moments(const ::cellerator::tensor::base::device::SlicedEllView *matrix,
                              const std::uint8_t *row_mask,
                              column_moments_view *moments,
                              cudaStream_t stream) {
    if (matrix == nullptr || !valid(moments) || moments->columns != matrix->cols) return 0;
    sliced_ell_kernel<<<row_blocks(matrix->rows), 256, 0, stream>>>(*matrix, row_mask, *moments);
    return cudaGetLastError() == cudaSuccess && count_rows(matrix->rows, row_mask, moments, stream);
}

int accumulate_column_moments(const tensor::base::CompressedView<uint32_t, const void> *matrix,
                              const std::uint8_t *row_mask,
                              column_moments_view *moments,
                              cudaStream_t stream) {
    return accumulate_compressed(matrix, row_mask, moments, stream);
}

int accumulate_column_moments(const tensor::base::CompressedView<int32_t, const void> *matrix,
                              const std::uint8_t *row_mask,
                              column_moments_view *moments,
                              cudaStream_t stream) {
    return accumulate_compressed(matrix, row_mask, moments, stream);
}

int accumulate_column_moments(const tensor::base::CompressedView<int64_t, const void> *matrix,
                              const std::uint8_t *row_mask,
                              column_moments_view *moments,
                              cudaStream_t stream) {
    return accumulate_compressed(matrix, row_mask, moments, stream);
}

} // namespace cellerator::compute::sparse
