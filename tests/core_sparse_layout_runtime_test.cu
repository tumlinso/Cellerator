#include <Cellerator/core/matrix/sparse.cuh>
#include <Cellerator/compute/matrix/convert/bucket.cuh>
#include <Cellerator/compute/matrix/convert/compressed.cuh>

#include <cub/cub.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace convert = ::cellerator::compute::matrix::convert;
namespace bucket = ::cellerator::compute::matrix::convert::bucket;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char *message) {
    if (status != cudaSuccess) throw std::runtime_error(message);
}

template<typename T>
T *device_alloc(std::size_t count) {
    T *ptr = nullptr;
    if (count == 0u) return ptr;
    cuda_require(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)), "cudaMalloc failed");
    return ptr;
}

template<typename T>
void upload(T *dst, const T *src, std::size_t count) {
    if (count == 0u) return;
    cuda_require(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");
}

template<typename T>
void download(T *dst, const T *src, std::size_t count) {
    if (count == 0u) return;
    cuda_require(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");
}

bool close_half(__half value, float expected) {
    return std::fabs(__half2float(value) - expected) <= 1.0e-3f;
}

} // namespace

int main() {
    int device_count = 0;
    cuda_require(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
    require(device_count > 0, "core sparse runtime test requires a CUDA device");

    const std::uint32_t rows = 3u;
    const std::uint32_t cols = 4u;
    const std::uint32_t nnz = 5u;
    const std::uint32_t h_row[] = { 0u, 0u, 1u, 2u, 2u };
    const std::uint32_t h_col[] = { 1u, 3u, 2u, 0u, 3u };
    const __half h_val[] = {
        __float2half(1.0f),
        __float2half(2.0f),
        __float2half(3.0f),
        __float2half(4.0f),
        __float2half(5.0f)
    };

    std::uint32_t *d_row = device_alloc<std::uint32_t>(nnz);
    std::uint32_t *d_col = device_alloc<std::uint32_t>(nnz);
    __half *d_val = device_alloc<__half>(nnz);
    std::uint32_t *d_ptr = device_alloc<std::uint32_t>(rows + 1u);
    std::uint32_t *d_minor = device_alloc<std::uint32_t>(nnz);
    __half *d_out_val = device_alloc<__half>(nnz);
    upload(d_row, h_row, nnz);
    upload(d_col, h_col, nnz);
    upload(d_val, h_val, nnz);

    require(convert::build_compressed_from_sorted_coo_custom_raw(
                rows,
                nnz,
                d_row,
                d_col,
                d_val,
                d_ptr,
                d_minor,
                d_out_val,
                nullptr) != 0,
            "sorted COO to compressed failed");
    cuda_require(cudaDeviceSynchronize(), "sync after sorted COO conversion failed");

    std::uint32_t h_ptr[4] = {};
    std::uint32_t h_minor[5] = {};
    __half h_out_val[5] = {};
    download(h_ptr, d_ptr, rows + 1u);
    download(h_minor, d_minor, nnz);
    download(h_out_val, d_out_val, nnz);
    require(h_ptr[0] == 0u && h_ptr[1] == 2u && h_ptr[2] == 3u && h_ptr[3] == 5u, "compressed pointer mismatch");
    require(h_minor[0] == 1u && h_minor[1] == 3u && h_minor[2] == 2u && h_minor[3] == 0u && h_minor[4] == 3u, "compressed minor mismatch");
    require(close_half(h_out_val[4], 5.0f), "compressed values mismatch");

    std::size_t scan_bytes = 0;
    require(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, d_ptr, d_ptr, cols + 1u) == cudaSuccess, "transpose scan bytes failed");
    void *d_scan = nullptr;
    cuda_require(cudaMalloc(&d_scan, scan_bytes), "cudaMalloc scan failed");
    std::uint32_t *d_tptr = device_alloc<std::uint32_t>(cols + 1u);
    std::uint32_t *d_heads = device_alloc<std::uint32_t>(cols);
    std::uint32_t *d_tminor = device_alloc<std::uint32_t>(nnz);
    __half *d_tval = device_alloc<__half>(nnz);
    require(convert::build_compressed_transpose_custom_raw(
                rows,
                cols,
                nnz,
                d_ptr,
                d_minor,
                d_out_val,
                d_tptr,
                d_heads,
                d_tminor,
                d_tval,
                d_scan,
                scan_bytes,
                nullptr) != 0,
            "compressed transpose failed");
    cuda_require(cudaDeviceSynchronize(), "sync after transpose failed");

    std::uint32_t h_tptr[5] = {};
    download(h_tptr, d_tptr, cols + 1u);
    require(h_tptr[0] == 0u && h_tptr[1] == 1u && h_tptr[2] == 2u && h_tptr[3] == 3u && h_tptr[4] == 5u,
            "transpose pointer mismatch");

    std::size_t sort_bytes = 0;
    require(bucket::major_nnz_bucket_sort_scratch_bytes(rows, &sort_bytes) != 0, "bucket sort bytes failed");
    void *d_sort = nullptr;
    cuda_require(cudaMalloc(&d_sort, sort_bytes), "cudaMalloc sort failed");
    std::uint32_t *d_major_nnz = device_alloc<std::uint32_t>(rows);
    std::uint32_t *d_major_nnz_sorted = device_alloc<std::uint32_t>(rows);
    std::uint32_t *d_order_in = device_alloc<std::uint32_t>(rows);
    std::uint32_t *d_order_out = device_alloc<std::uint32_t>(rows);
    std::uint32_t *d_bucket_offsets = device_alloc<std::uint32_t>(3u);
    require(bucket::build_major_nnz_bucket_plan_raw(
                d_ptr,
                rows,
                d_major_nnz,
                d_major_nnz_sorted,
                d_order_in,
                d_order_out,
                d_bucket_offsets,
                2u,
                d_sort,
                sort_bytes,
                nullptr) != 0,
            "major nnz bucket plan failed");
    cuda_require(cudaDeviceSynchronize(), "sync after bucket plan failed");
    std::uint32_t h_sorted[3] = {};
    std::uint32_t h_offsets[3] = {};
    download(h_sorted, d_major_nnz_sorted, rows);
    download(h_offsets, d_bucket_offsets, 3u);
    require(h_sorted[0] == 1u && h_sorted[1] == 2u && h_sorted[2] == 2u, "sorted major nnz mismatch");
    require(h_offsets[0] == 0u && h_offsets[1] == 1u && h_offsets[2] == 3u, "bucket offsets mismatch");

    cudaFree(d_bucket_offsets);
    cudaFree(d_order_out);
    cudaFree(d_order_in);
    cudaFree(d_major_nnz_sorted);
    cudaFree(d_major_nnz);
    cudaFree(d_sort);
    cudaFree(d_tval);
    cudaFree(d_tminor);
    cudaFree(d_heads);
    cudaFree(d_tptr);
    cudaFree(d_scan);
    cudaFree(d_out_val);
    cudaFree(d_minor);
    cudaFree(d_ptr);
    cudaFree(d_val);
    cudaFree(d_col);
    cudaFree(d_row);
    return 0;
}
