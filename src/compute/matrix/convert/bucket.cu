#include <Cellerator/compute/matrix/convert/bucket.cuh>

#include <cub/cub.cuh>

#include <cstdio>

namespace cellerator::compute::matrix::convert::bucket {

namespace {

__host__ int cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

__device__ __forceinline__ unsigned int global_tid_1d() {
    return (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ __forceinline__ unsigned int global_stride_1d() {
    return (unsigned int) (gridDim.x * blockDim.x);
}

__global__ void count_major_nnz(
    types::dim_t major_dim,
    const types::ptr_t * __restrict__ major_ptr,
    types::idx_t * __restrict__ major_nnz) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < major_dim) {
        major_nnz[i] = major_ptr[i + 1u] - major_ptr[i];
        i += stride;
    }
}

__global__ void init_major_identity(types::dim_t major_dim, types::idx_t * __restrict__ order) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < major_dim) {
        order[i] = i;
        i += stride;
    }
}

__global__ void fill_equal_count_bucket_offsets(
    types::dim_t major_dim,
    types::idx_t bucket_count,
    types::idx_t * __restrict__ bucket_offsets) {
    for (types::idx_t bucket = 0u; bucket <= bucket_count; ++bucket) {
        bucket_offsets[bucket] = (types::idx_t) (((unsigned long long) bucket * (unsigned long long) major_dim) / (unsigned long long) bucket_count);
    }
}

__device__ __forceinline__ types::idx_t locate_part_for_row(
    types::idx_t row,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count) {
    types::idx_t lo = 0u, hi = part_count;
    while (lo + 1u < hi) {
        const types::idx_t mid = lo + ((hi - lo) >> 1u);
        if (part_row_offsets[mid] <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

__global__ void count_shard_major_nnz(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    types::idx_t * __restrict__ major_nnz) {
    unsigned int row = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (row < shard_rows) {
        const types::idx_t part = locate_part_for_row(row, part_row_offsets, part_count);
        const types::idx_t local_row = row - part_row_offsets[part];
        const types::ptr_t *ptr = part_major_ptr[part];
        major_nnz[row] = ptr[local_row + 1u] - ptr[local_row];
        row += stride;
    }
}

__global__ void gather_shifted_major_counts_from_order(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    const types::ptr_t * __restrict__ src_major_ptr,
    types::ptr_t * __restrict__ dst_major_ptr) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < major_dim) {
        const types::idx_t src = major_order[i];
        dst_major_ptr[i + 1u] = src_major_ptr[src + 1u] - src_major_ptr[src];
        i += stride;
    }
    if (global_tid_1d() == 0u) dst_major_ptr[0] = 0u;
}

__global__ void gather_shifted_shard_major_counts_from_order(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ major_order,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    types::ptr_t * __restrict__ dst_major_ptr) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < shard_rows) {
        const types::idx_t global_row = major_order[i];
        const types::idx_t part = locate_part_for_row(global_row, part_row_offsets, part_count);
        const types::idx_t local_row = global_row - part_row_offsets[part];
        const types::ptr_t *ptr = part_major_ptr[part];
        dst_major_ptr[i + 1u] = ptr[local_row + 1u] - ptr[local_row];
        i += stride;
    }
    if (global_tid_1d() == 0u) dst_major_ptr[0] = 0u;
}

__global__ void reorder_compressed_major_segments(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    const types::ptr_t * __restrict__ src_major_ptr,
    const types::idx_t * __restrict__ src_minor_idx,
    const real::storage_t * __restrict__ src_val,
    const types::ptr_t * __restrict__ dst_major_ptr,
    types::idx_t * __restrict__ dst_minor_idx,
    real::storage_t * __restrict__ dst_val) {
    unsigned int i = blockIdx.x * blockDim.y + threadIdx.y;
    while (i < major_dim) {
        const types::idx_t src_major = major_order[i];
        const types::ptr_t src_begin = src_major_ptr[src_major];
        const types::ptr_t src_end = src_major_ptr[src_major + 1u];
        const types::ptr_t dst_begin = dst_major_ptr[i];
        for (types::ptr_t offset = threadIdx.x; offset < src_end - src_begin; offset += blockDim.x) {
            dst_minor_idx[dst_begin + offset] = src_minor_idx[src_begin + offset];
            dst_val[dst_begin + offset] = src_val[src_begin + offset];
        }
        i += gridDim.x * blockDim.y;
    }
}

__global__ void reorder_shard_major_segments(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ major_order,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    const types::idx_t * const * __restrict__ part_minor_idx,
    const real::storage_t * const * __restrict__ part_val,
    const types::ptr_t * __restrict__ dst_major_ptr,
    types::idx_t * __restrict__ dst_minor_idx,
    real::storage_t * __restrict__ dst_val) {
    unsigned int i = blockIdx.x * blockDim.y + threadIdx.y;
    while (i < shard_rows) {
        const types::idx_t global_row = major_order[i];
        const types::idx_t part = locate_part_for_row(global_row, part_row_offsets, part_count);
        const types::idx_t local_row = global_row - part_row_offsets[part];
        const types::ptr_t *src_ptr = part_major_ptr[part];
        const types::idx_t *src_idx = part_minor_idx[part];
        const real::storage_t *src_val = part_val[part];
        const types::ptr_t src_begin = src_ptr[local_row];
        const types::ptr_t src_end = src_ptr[local_row + 1u];
        const types::ptr_t dst_begin = dst_major_ptr[i];
        for (types::ptr_t offset = threadIdx.x; offset < src_end - src_begin; offset += blockDim.x) {
            dst_minor_idx[dst_begin + offset] = src_idx[src_begin + offset];
            dst_val[dst_begin + offset] = src_val[src_begin + offset];
        }
        i += gridDim.x * blockDim.y;
    }
}

__global__ void scatter_inverse_major_order(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    types::idx_t * __restrict__ inverse_major_order) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < major_dim) {
        inverse_major_order[major_order[i]] = i;
        i += stride;
    }
}

void setup_1d(types::dim_t n, dim3 *grid, dim3 *block) {
    block->x = 256;
    block->y = 1;
    block->z = 1;
    grid->x = (unsigned int) ((n + 255u) >> 8);
    grid->y = 1;
    grid->z = 1;
    if (grid->x < 1u) grid->x = 1u;
    if (grid->x > 4096u) grid->x = 4096u;
}

void setup_major_segments(types::dim_t n, dim3 *grid, dim3 *block) {
    block->x = 32;
    block->y = 8;
    block->z = 1;
    grid->x = (unsigned int) ((n + block->y - 1u) / block->y);
    grid->y = 1;
    grid->z = 1;
    if (grid->x < 1u) grid->x = 1u;
    if (grid->x > 4096u) grid->x = 4096u;
}

} // namespace

types::idx_t clamp_bucket_count(types::dim_t major_dim, types::idx_t requested) {
    if (major_dim == 0u) return 1u;
    if (requested < 1u) return 1u;
    if (requested > major_dim) return (types::idx_t) major_dim;
    return requested;
}

int major_nnz_bucket_sort_scratch_bytes(types::dim_t major_dim, std::size_t *out_bytes) {
    std::size_t bytes = 0;
    if (out_bytes == 0) return 0;
    if (cub::DeviceRadixSort::SortPairs(0, bytes, (const types::idx_t *) 0, (types::idx_t *) 0, (const types::idx_t *) 0, (types::idx_t *) 0, major_dim, 0, sizeof(types::idx_t) * 8) != cudaSuccess) return 0;
    *out_bytes = bytes;
    return 1;
}

int major_nnz_bucket_scan_scratch_bytes(types::dim_t major_dim, std::size_t *out_bytes) {
    std::size_t bytes = 0;
    if (out_bytes == 0) return 0;
    if (cub::DeviceScan::ExclusiveSum(0, bytes, (const types::ptr_t *) 0, (types::ptr_t *) 0, major_dim + 1u) != cudaSuccess) return 0;
    *out_bytes = bytes;
    return 1;
}

int build_major_nnz_bucket_plan_library_raw(
    const types::ptr_t *d_major_ptr,
    types::dim_t major_dim,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream) {
    dim3 grid, block;

    if (d_major_ptr == 0 || d_major_nnz == 0 || d_major_nnz_sorted == 0 || d_major_order_in == 0 || d_major_order_out == 0) return 0;
    if (major_dim == 0u) {
        if (d_bucket_offsets != 0 && !cuda_check(cudaMemsetAsync(d_bucket_offsets, 0, sizeof(types::idx_t), stream), "cudaMemsetAsync empty bucket offsets")) return 0;
        return 1;
    }

    setup_1d(major_dim, &grid, &block);
    count_major_nnz<<<grid, block, 0, stream>>>(major_dim, d_major_ptr, d_major_nnz);
    if (cudaGetLastError() != cudaSuccess) return 0;
    init_major_identity<<<grid, block, 0, stream>>>(major_dim, d_major_order_in);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (!cuda_check(cub::DeviceRadixSort::SortPairs(d_sort_tmp, sort_tmp_bytes, d_major_nnz, d_major_nnz_sorted, d_major_order_in, d_major_order_out, major_dim, 0, sizeof(types::idx_t) * 8, stream), "cub bucket sort pairs")) return 0;
    if (d_bucket_offsets != 0) {
        fill_equal_count_bucket_offsets<<<1, 1, 0, stream>>>(major_dim, clamp_bucket_count(major_dim, requested_bucket_count), d_bucket_offsets);
        if (cudaGetLastError() != cudaSuccess) return 0;
    }
    return 1;
}

int build_major_nnz_bucket_plan_custom_raw(
    const types::ptr_t *d_major_ptr,
    types::dim_t major_dim,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream) {
    return build_major_nnz_bucket_plan_library_raw(d_major_ptr, major_dim, d_major_nnz, d_major_nnz_sorted, d_major_order_in, d_major_order_out, d_bucket_offsets, requested_bucket_count, d_sort_tmp, sort_tmp_bytes, stream);
}

int build_major_nnz_bucket_plan_raw(
    const types::ptr_t *d_major_ptr,
    types::dim_t major_dim,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream) {
    return build_major_nnz_bucket_plan_library_raw(d_major_ptr, major_dim, d_major_nnz, d_major_nnz_sorted, d_major_order_in, d_major_order_out, d_bucket_offsets, requested_bucket_count, d_sort_tmp, sort_tmp_bytes, stream);
}

int build_shard_major_nnz_bucket_plan_raw(
    const types::ptr_t * const *d_part_major_ptr,
    const types::idx_t *d_part_row_offsets,
    types::idx_t part_count,
    types::dim_t shard_rows,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream) {
    dim3 grid, block;

    if (d_part_major_ptr == 0 || d_part_row_offsets == 0 || d_major_nnz == 0 || d_major_nnz_sorted == 0 || d_major_order_in == 0 || d_major_order_out == 0) return 0;
    if (shard_rows == 0u) {
        if (d_bucket_offsets != 0 && !cuda_check(cudaMemsetAsync(d_bucket_offsets, 0, sizeof(types::idx_t), stream), "cudaMemsetAsync empty shard bucket offsets")) return 0;
        return 1;
    }

    setup_1d(shard_rows, &grid, &block);
    count_shard_major_nnz<<<grid, block, 0, stream>>>(shard_rows, d_part_row_offsets, part_count, d_part_major_ptr, d_major_nnz);
    if (cudaGetLastError() != cudaSuccess) return 0;
    init_major_identity<<<grid, block, 0, stream>>>(shard_rows, d_major_order_in);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (!cuda_check(cub::DeviceRadixSort::SortPairs(d_sort_tmp, sort_tmp_bytes, d_major_nnz, d_major_nnz_sorted, d_major_order_in, d_major_order_out, shard_rows, 0, sizeof(types::idx_t) * 8, stream), "cub shard bucket sort pairs")) return 0;
    if (d_bucket_offsets != 0) {
        fill_equal_count_bucket_offsets<<<1, 1, 0, stream>>>(shard_rows, clamp_bucket_count(shard_rows, requested_bucket_count), d_bucket_offsets);
        if (cudaGetLastError() != cudaSuccess) return 0;
    }
    return 1;
}

int rebuild_compressed_major_order_raw(
    const types::ptr_t *d_src_major_ptr,
    const types::idx_t *d_src_minor_idx,
    const real::storage_t *d_src_val,
    types::dim_t major_dim,
    const types::idx_t *d_major_order,
    types::ptr_t *d_dst_major_ptr,
    types::idx_t *d_dst_minor_idx,
    real::storage_t *d_dst_val,
    types::idx_t *d_inverse_major_order,
    void *d_scan_tmp,
    std::size_t scan_tmp_bytes,
    cudaStream_t stream) {
    dim3 grid, block, seg_grid, seg_block;

    if (d_src_major_ptr == 0 || d_src_minor_idx == 0 || d_src_val == 0 || d_major_order == 0 || d_dst_major_ptr == 0 || d_dst_minor_idx == 0 || d_dst_val == 0) return 0;
    if (major_dim == 0u) return 1;

    setup_1d(major_dim, &grid, &block);
    gather_shifted_major_counts_from_order<<<grid, block, 0, stream>>>(major_dim, d_major_order, d_src_major_ptr, d_dst_major_ptr);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (!cuda_check(cub::DeviceScan::ExclusiveSum(d_scan_tmp, scan_tmp_bytes, d_dst_major_ptr, d_dst_major_ptr, major_dim + 1u, stream), "cub bucket major scan")) return 0;

    setup_major_segments(major_dim, &seg_grid, &seg_block);
    reorder_compressed_major_segments<<<seg_grid, seg_block, 0, stream>>>(major_dim, d_major_order, d_src_major_ptr, d_src_minor_idx, d_src_val, d_dst_major_ptr, d_dst_minor_idx, d_dst_val);
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (d_inverse_major_order != 0) {
        scatter_inverse_major_order<<<grid, block, 0, stream>>>(major_dim, d_major_order, d_inverse_major_order);
        if (cudaGetLastError() != cudaSuccess) return 0;
    }
    return 1;
}

int rebuild_bucketed_shard_compressed_raw(
    const types::ptr_t * const *d_part_major_ptr,
    const types::idx_t * const *d_part_minor_idx,
    const real::storage_t * const *d_part_val,
    const types::idx_t *d_part_row_offsets,
    types::idx_t part_count,
    types::dim_t shard_rows,
    const types::idx_t *d_major_order,
    types::ptr_t *d_dst_major_ptr,
    types::idx_t *d_dst_minor_idx,
    real::storage_t *d_dst_val,
    types::idx_t *d_inverse_major_order,
    void *d_scan_tmp,
    std::size_t scan_tmp_bytes,
    cudaStream_t stream) {
    dim3 grid, block, seg_grid, seg_block;

    if (d_part_major_ptr == 0 || d_part_minor_idx == 0 || d_part_val == 0 || d_part_row_offsets == 0 || d_major_order == 0 || d_dst_major_ptr == 0 || d_dst_minor_idx == 0 || d_dst_val == 0) return 0;
    if (shard_rows == 0u) return 1;

    setup_1d(shard_rows, &grid, &block);
    gather_shifted_shard_major_counts_from_order<<<grid, block, 0, stream>>>(shard_rows, d_major_order, d_part_row_offsets, part_count, d_part_major_ptr, d_dst_major_ptr);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (!cuda_check(cub::DeviceScan::ExclusiveSum(d_scan_tmp, scan_tmp_bytes, d_dst_major_ptr, d_dst_major_ptr, shard_rows + 1u, stream), "cub shard bucket major scan")) return 0;

    setup_major_segments(shard_rows, &seg_grid, &seg_block);
    reorder_shard_major_segments<<<seg_grid, seg_block, 0, stream>>>(shard_rows, d_major_order, d_part_row_offsets, part_count, d_part_major_ptr, d_part_minor_idx, d_part_val, d_dst_major_ptr, d_dst_minor_idx, d_dst_val);
    if (cudaGetLastError() != cudaSuccess) return 0;

    if (d_inverse_major_order != 0) {
        scatter_inverse_major_order<<<grid, block, 0, stream>>>(shard_rows, d_major_order, d_inverse_major_order);
        if (cudaGetLastError() != cudaSuccess) return 0;
    }
    return 1;
}

} // namespace cellerator::compute::matrix::convert::bucket
