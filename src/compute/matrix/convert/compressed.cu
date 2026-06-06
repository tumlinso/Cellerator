#include <Cellerator/compute/matrix/convert/compressed.cuh>

#include <cub/cub.cuh>
#include <cusparse.h>

#include <cstdio>

namespace cellerator::compute::matrix::convert {

namespace {

struct thread_cusparse_cache {
    int device = -1;
    cusparseHandle_t handle = 0;

    ~thread_cusparse_cache() {
        if (handle != 0) (void) cusparseDestroy(handle);
    }
};

__host__ int cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

__host__ int cusparse_check(cusparseStatus_t status, const char *label) {
    if (status == CUSPARSE_STATUS_SUCCESS) return 1;
    std::fprintf(stderr, "cuSPARSE error at %s: %d\n", label, (int) status);
    return 0;
}

__host__ int acquire_cusparse(cudaStream_t stream, cusparseHandle_t *handle) {
    thread_local thread_cusparse_cache cache;
    int device = -1;

    if (!cuda_check(cudaGetDevice(&device), "cudaGetDevice acquire cuSPARSE handle")) return 0;
    if (cache.handle == 0 || cache.device != device) {
        if (cache.handle != 0) {
            if (!cusparse_check(cusparseDestroy(cache.handle), "cusparseDestroy stale handle")) return 0;
            cache.handle = 0;
        }
        if (!cusparse_check(cusparseCreate(&cache.handle), "cusparseCreate")) return 0;
        cache.device = device;
    }
    if (!cusparse_check(cusparseSetStream(cache.handle, stream), "cusparseSetStream")) return 0;
    *handle = cache.handle;
    return 1;
}

__device__ __forceinline__ unsigned int global_tid_1d() {
    return (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ __forceinline__ unsigned int global_stride_1d() {
    return (unsigned int) (gridDim.x * blockDim.x);
}

__global__ void gather_half_by_permutation(
    types::nnz_t nnz,
    const types::idx_t * __restrict__ permutation,
    const real::storage_t * __restrict__ src_val,
    real::storage_t * __restrict__ dst_val) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < nnz) {
        dst_val[i] = src_val[permutation[i]];
        i += stride;
    }
}

__device__ __forceinline__ types::idx_t lower_bound_coo_major(
    const types::idx_t * __restrict__ majors,
    types::nnz_t nnz,
    types::idx_t key) {
    types::idx_t lo = 0u, hi = nnz;
    while (lo < hi) {
        const types::idx_t mid = lo + ((hi - lo) >> 1u);
        if (majors[mid] < key) lo = mid + 1u;
        else hi = mid;
    }
    return lo;
}

__global__ void build_sorted_coo_ptr(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t * __restrict__ majors,
    types::ptr_t * __restrict__ ptr) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i <= cDim) {
        ptr[i] = lower_bound_coo_major(majors, nnz, i);
        i += stride;
    }
}

__global__ void shift_ptr_idx_count(
    types::nnz_t nnz,
    const types::idx_t * __restrict__ major_idx,
    types::ptr_t * __restrict__ ptr) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < nnz) {
        atomicAdd(ptr + major_idx[i] + 1u, 1u);
        i += stride;
    }
}

__global__ void init_scatter_heads(
    types::dim_t major_dim,
    const types::ptr_t * __restrict__ ptr,
    types::ptr_t * __restrict__ heads) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < major_dim) {
        heads[i] = ptr[i];
        i += stride;
    }
}

__global__ void scatter_coo_to_compressed(
    types::nnz_t nnz,
    const types::idx_t * __restrict__ major_idx,
    const types::idx_t * __restrict__ minor_idx,
    const real::storage_t * __restrict__ val,
    types::ptr_t * __restrict__ heads,
    types::idx_t * __restrict__ out_minor,
    real::storage_t * __restrict__ out_val) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < nnz) {
        const types::idx_t major = major_idx[i];
        const types::ptr_t dst = atomicAdd(heads + major, 1u);
        out_minor[dst] = minor_idx[i];
        out_val[dst] = val[i];
        i += stride;
    }
}

__global__ void count_transpose_targets(
    types::dim_t cDim,
    const types::ptr_t * __restrict__ cAxPtr,
    const types::idx_t * __restrict__ uAxIdx,
    types::ptr_t * __restrict__ out_uAxPtr) {
    unsigned int major = blockIdx.x * blockDim.y + threadIdx.y;
    while (major < cDim) {
        for (types::ptr_t i = cAxPtr[major] + threadIdx.x; i < cAxPtr[major + 1u]; i += blockDim.x) {
            atomicAdd(out_uAxPtr + uAxIdx[i] + 1u, 1u);
        }
        major += gridDim.x * blockDim.y;
    }
}

__global__ void scatter_transpose(
    types::dim_t cDim,
    const types::ptr_t * __restrict__ cAxPtr,
    const types::idx_t * __restrict__ uAxIdx,
    const real::storage_t * __restrict__ val,
    types::ptr_t * __restrict__ heads,
    types::idx_t * __restrict__ out_cAxIdx,
    real::storage_t * __restrict__ out_val) {
    unsigned int major = blockIdx.x * blockDim.y + threadIdx.y;
    while (major < cDim) {
        for (types::ptr_t i = cAxPtr[major] + threadIdx.x; i < cAxPtr[major + 1u]; i += blockDim.x) {
            const types::idx_t minor = uAxIdx[i];
            const types::ptr_t dst = atomicAdd(heads + minor, 1u);
            out_cAxIdx[dst] = major;
            out_val[dst] = val[i];
        }
        major += gridDim.x * blockDim.y;
    }
}

__global__ void transpose_coo_entries_kernel(
    types::nnz_t nnz,
    const types::idx_t * __restrict__ src_row,
    const types::idx_t * __restrict__ src_col,
    const real::storage_t * __restrict__ src_val,
    types::idx_t * __restrict__ dst_row,
    types::idx_t * __restrict__ dst_col,
    real::storage_t * __restrict__ dst_val) {
    unsigned int i = global_tid_1d();
    const unsigned int stride = global_stride_1d();

    while (i < nnz) {
        dst_row[i] = src_col[i];
        dst_col[i] = src_row[i];
        dst_val[i] = src_val[i];
        i += stride;
    }
}

void setup_1d(types::dim_t n, int *blocks) {
    *blocks = (int) ((n + 255u) >> 8);
    if (*blocks < 1) *blocks = 1;
    if (*blocks > 4096) *blocks = 4096;
}

void setup_transpose(types::dim_t cDim, dim3 *grid, dim3 *block) {
    block->x = 32;
    block->y = 8;
    block->z = 1;
    grid->x = (unsigned int) ((cDim + block->y - 1u) / block->y);
    grid->y = 1;
    grid->z = 1;
    if (grid->x < 1u) grid->x = 1u;
    if (grid->x > 4096u) grid->x = 4096u;
}

} // namespace

int compressed_from_coo_library_workspace_bytes(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    cudaStream_t stream,
    std::size_t *bytes_out) {
    cusparseHandle_t handle = 0;
    std::size_t sort_bytes = 0;

    if (bytes_out == 0) return 0;
    *bytes_out = 0;
    if (nnz == 0u) return 1;
    if (!acquire_cusparse(stream, &handle)) return 0;
    if (!cusparse_check(
            cusparseXcoosort_bufferSizeExt(
                handle,
                (int) cDim,
                (int) uDim,
                (int) nnz,
                reinterpret_cast<const int *>(d_cAxIdx),
                reinterpret_cast<const int *>(d_uAxIdx),
                &sort_bytes),
            "cusparseXcoosort_bufferSizeExt")) return 0;
    *bytes_out = sort_bytes;
    return 1;
}

int build_compressed_from_coo_library_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::idx_t *d_sort_cAxIdx,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    types::idx_t *d_permutation,
    void *d_sort_tmp,
    std::size_t sort_bytes,
    cudaStream_t stream) {
    cusparseHandle_t handle = 0;
    int blocks = 0;
    std::size_t required_bytes = 0;

    if (!cuda_check(cudaMemsetAsync(d_cAxPtr, 0, (std::size_t) (cDim + 1u) * sizeof(types::ptr_t), stream), "cudaMemsetAsync sorted compressed ptr")) return 0;
    if (nnz == 0u) return 1;
    if (d_sort_cAxIdx == 0 || d_out_uAx == 0 || d_out_val == 0 || d_permutation == 0 || d_sort_tmp == 0) return 0;
    if (!acquire_cusparse(stream, &handle)) return 0;
    if (!compressed_from_coo_library_workspace_bytes(cDim, uDim, nnz, d_sort_cAxIdx, d_out_uAx, stream, &required_bytes)) return 0;
    if (sort_bytes < required_bytes) return 0;

    if (!cuda_check(cudaMemcpyAsync(d_sort_cAxIdx, d_cAxIdx, (std::size_t) nnz * sizeof(types::idx_t), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync COO sort majors")) return 0;
    if (!cuda_check(cudaMemcpyAsync(d_out_uAx, d_uAxIdx, (std::size_t) nnz * sizeof(types::idx_t), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync COO sort minors")) return 0;
    if (!cusparse_check(cusparseCreateIdentityPermutation(handle, (int) nnz, reinterpret_cast<int *>(d_permutation)), "cusparseCreateIdentityPermutation")) return 0;
    if (!cusparse_check(
            cusparseXcoosortByRow(
                handle,
                (int) cDim,
                (int) uDim,
                (int) nnz,
                reinterpret_cast<int *>(d_sort_cAxIdx),
                reinterpret_cast<int *>(d_out_uAx),
                reinterpret_cast<int *>(d_permutation),
                d_sort_tmp),
            "cusparseXcoosortByRow")) return 0;
    if (!cusparse_check(
            cusparseXcoo2csr(
                handle,
                reinterpret_cast<const int *>(d_sort_cAxIdx),
                (int) nnz,
                (int) cDim,
                reinterpret_cast<int *>(d_cAxPtr),
                CUSPARSE_INDEX_BASE_ZERO),
            "cusparseXcoo2csr")) return 0;

    setup_1d(nnz, &blocks);
    gather_half_by_permutation<<<blocks, 256, 0, stream>>>(nnz, d_permutation, d_val, d_out_val);
    return cudaGetLastError() == cudaSuccess;
}

int build_compressed_from_coo_custom_raw(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream) {
    int blocks_nnz = 0, blocks_cdim = 0;

    if (cudaMemsetAsync(d_cAxPtr, 0, (std::size_t) (cDim + 1u) * sizeof(types::ptr_t), stream) != cudaSuccess) return 0;
    if (nnz == 0u) return 1;
    setup_1d(nnz, &blocks_nnz);
    setup_1d(cDim, &blocks_cdim);

    shift_ptr_idx_count<<<blocks_nnz, 256, 0, stream>>>(nnz, d_cAxIdx, d_cAxPtr);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (cub::DeviceScan::InclusiveSum(d_scan_tmp, scan_bytes, d_cAxPtr, d_cAxPtr, cDim + 1u, stream) != cudaSuccess) return 0;
    init_scatter_heads<<<blocks_cdim, 256, 0, stream>>>(cDim, d_cAxPtr, d_heads);
    if (cudaGetLastError() != cudaSuccess) return 0;
    scatter_coo_to_compressed<<<blocks_nnz, 256, 0, stream>>>(nnz, d_cAxIdx, d_uAxIdx, d_val, d_heads, d_out_uAx, d_out_val);
    return cudaGetLastError() == cudaSuccess;
}

int build_compressed_from_sorted_coo_custom_raw(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    cudaStream_t stream) {
    int blocks_nnz = 0, blocks_cdim = 0;
    setup_1d(nnz, &blocks_nnz);
    setup_1d(cDim + 1u, &blocks_cdim);

    build_sorted_coo_ptr<<<blocks_cdim, 256, 0, stream>>>(cDim, nnz, d_cAxIdx, d_cAxPtr);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (nnz == 0u) return 1;
    if (d_out_uAx != d_uAxIdx && !cuda_check(cudaMemcpyAsync(d_out_uAx, d_uAxIdx, (std::size_t) nnz * sizeof(types::idx_t), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync sorted COO minor idx")) return 0;
    if (d_out_val != d_val && !cuda_check(cudaMemcpyAsync(d_out_val, d_val, (std::size_t) nnz * sizeof(real::storage_t), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync sorted COO values")) return 0;
    return 1;
}

int compressed_from_coo_sorted_workspace_bytes(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    cudaStream_t stream,
    std::size_t *bytes_out) {
    return compressed_from_coo_library_workspace_bytes(cDim, uDim, nnz, d_cAxIdx, d_uAxIdx, stream, bytes_out);
}

int build_compressed_from_coo_sorted_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::idx_t *d_sort_cAxIdx,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    types::idx_t *d_permutation,
    void *d_sort_tmp,
    std::size_t sort_bytes,
    cudaStream_t stream) {
    return build_compressed_from_coo_library_raw(cDim, uDim, nnz, d_cAxIdx, d_uAxIdx, d_val, d_cAxPtr, d_sort_cAxIdx, d_out_uAx, d_out_val, d_permutation, d_sort_tmp, sort_bytes, stream);
}

int build_compressed_from_coo_raw(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream) {
    return build_compressed_from_coo_custom_raw(cDim, nnz, d_cAxIdx, d_uAxIdx, d_val, d_cAxPtr, d_heads, d_out_uAx, d_out_val, d_scan_tmp, scan_bytes, stream);
}

int build_cs_from_coo_raw(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream) {
    return build_compressed_from_coo_raw(cDim, nnz, d_cAxIdx, d_uAxIdx, d_val, d_cAxPtr, d_heads, d_out_uAx, d_out_val, d_scan_tmp, scan_bytes, stream);
}

int compressed_transpose_library_workspace_bytes(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    cudaStream_t stream,
    std::size_t *bytes_out) {
    cusparseHandle_t handle = 0;
    std::size_t bytes = 0;

    if (bytes_out == 0) return 0;
    *bytes_out = 0;
    if (nnz == 0u) return 1;
    if (!acquire_cusparse(stream, &handle)) return 0;
    if (!cusparse_check(
            cusparseCsr2cscEx2_bufferSize(
                handle,
                (int) cDim,
                (int) uDim,
                (int) nnz,
                d_val,
                reinterpret_cast<const int *>(d_cAxPtr),
                reinterpret_cast<const int *>(d_uAxIdx),
                d_out_val,
                reinterpret_cast<int *>(d_out_uAxPtr),
                reinterpret_cast<int *>(d_out_cAxIdx),
                CUDA_R_16F,
                CUSPARSE_ACTION_NUMERIC,
                CUSPARSE_INDEX_BASE_ZERO,
                CUSPARSE_CSR2CSC_ALG1,
                &bytes),
            "cusparseCsr2cscEx2_bufferSize")) return 0;
    *bytes_out = bytes;
    return 1;
}

int build_compressed_transpose_library_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_tmp,
    std::size_t tmp_bytes,
    cudaStream_t stream) {
    cusparseHandle_t handle = 0;
    std::size_t required_bytes = 0;

    if (!cuda_check(cudaMemsetAsync(d_out_uAxPtr, 0, (std::size_t) (uDim + 1u) * sizeof(types::ptr_t), stream), "cudaMemsetAsync transpose out ptr")) return 0;
    if (nnz == 0u) return 1;
    if (d_tmp == 0 || !acquire_cusparse(stream, &handle)) return 0;
    if (!compressed_transpose_library_workspace_bytes(cDim, uDim, nnz, d_cAxPtr, d_uAxIdx, d_val, d_out_uAxPtr, d_out_cAxIdx, d_out_val, stream, &required_bytes)) return 0;
    if (tmp_bytes < required_bytes) return 0;
    return cusparse_check(
        cusparseCsr2cscEx2(
            handle,
            (int) cDim,
            (int) uDim,
            (int) nnz,
            d_val,
            reinterpret_cast<const int *>(d_cAxPtr),
            reinterpret_cast<const int *>(d_uAxIdx),
            d_out_val,
            reinterpret_cast<int *>(d_out_uAxPtr),
            reinterpret_cast<int *>(d_out_cAxIdx),
            CUDA_R_16F,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1,
            d_tmp),
        "cusparseCsr2cscEx2");
}

int build_compressed_transpose_custom_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream) {
    dim3 grid, block;

    if (!cuda_check(cudaMemsetAsync(d_out_uAxPtr, 0, (std::size_t) (uDim + 1u) * sizeof(types::ptr_t), stream), "cudaMemsetAsync transpose out ptr")) return 0;
    if (nnz == 0u) return 1;

    setup_transpose(cDim, &grid, &block);
    count_transpose_targets<<<grid, block, 0, stream>>>(cDim, d_cAxPtr, d_uAxIdx, d_out_uAxPtr);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (!cuda_check(cub::DeviceScan::InclusiveSum(d_scan_tmp, scan_bytes, d_out_uAxPtr, d_out_uAxPtr, uDim + 1u, stream), "cub transpose scan")) return 0;
    init_scatter_heads<<<grid, block, 0, stream>>>(uDim, d_out_uAxPtr, d_heads);
    if (cudaGetLastError() != cudaSuccess) return 0;
    scatter_transpose<<<grid, block, 0, stream>>>(cDim, d_cAxPtr, d_uAxIdx, d_val, d_heads, d_out_cAxIdx, d_out_val);
    return cudaGetLastError() == cudaSuccess;
}

int build_compressed_transpose_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream) {
    if (build_compressed_transpose_custom_raw(cDim, uDim, nnz, d_cAxPtr, d_uAxIdx, d_val, d_out_uAxPtr, d_heads, d_out_cAxIdx, d_out_val, d_scan_tmp, scan_bytes, stream)) return 1;
    return build_compressed_transpose_library_raw(cDim, uDim, nnz, d_cAxPtr, d_uAxIdx, d_val, d_out_uAxPtr, d_out_cAxIdx, d_out_val, d_scan_tmp, scan_bytes, stream);
}

int build_transpose_cs_from_cs_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream) {
    return build_compressed_transpose_raw(cDim, uDim, nnz, d_cAxPtr, d_uAxIdx, d_val, d_out_uAxPtr, d_heads, d_out_cAxIdx, d_out_val, d_scan_tmp, scan_bytes, stream);
}

int transpose_coo_entries_raw(
    types::nnz_t nnz,
    const types::idx_t *d_src_rowIdx,
    const types::idx_t *d_src_colIdx,
    const real::storage_t *d_src_val,
    types::idx_t *d_dst_rowIdx,
    types::idx_t *d_dst_colIdx,
    real::storage_t *d_dst_val,
    cudaStream_t stream) {
    int blocks = 0;
    if (nnz == 0u) return 1;
    setup_1d(nnz, &blocks);
    transpose_coo_entries_kernel<<<blocks, 256, 0, stream>>>(nnz, d_src_rowIdx, d_src_colIdx, d_src_val, d_dst_rowIdx, d_dst_colIdx, d_dst_val);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace cellerator::compute::matrix::convert
