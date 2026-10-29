#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

__global__ void nnz_per_row(
    const int* __restrict__ rowptr,
    int* __restrict__ row_nnz,
    int num_rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) row_nnz[i] = rowptr[i + 1] - rowptr[i];
}

inline int get_bucket_count(int num_rows, int num_buckets) {
    if (num_rows <= 0) return 1;
    if (num_buckets < 1) return 1;
    if (num_buckets > num_rows) return num_rows;
    return num_buckets;
}

inline void fill_bucket_offsets(
    int* bucket_offsets,
    int num_rows,
    int num_buckets
) {
    const int bucket_count = get_bucket_count(num_rows, num_buckets);
    const int bucket_size = num_rows / bucket_count;
    const int remainder = num_rows % bucket_count;

    int offset = 0;
    for (int bucket = 0; bucket < bucket_count; ++bucket) {
        bucket_offsets[bucket] = offset;
        offset += bucket_size + (bucket < remainder ? 1 : 0);
    }
    bucket_offsets[bucket_count] = num_rows;
}

inline cudaError_t compute_nnz_buckets(
    const int* __restrict__ rowptr,
    int* __restrict__ row_nnz,
    int* __restrict__ row_indices,
    int* __restrict__ bucket_offsets,
    int num_rows,
    int num_buckets
) {
    if (num_rows < 0) return cudaErrorInvalidValue;

    if (bucket_offsets != nullptr) {
        const int bucket_count = get_bucket_count(num_rows, num_buckets);
        int* host_bucket_offsets = new int[bucket_count + 1];
        fill_bucket_offsets(host_bucket_offsets, num_rows, num_buckets);

        cudaError_t status = cudaMemcpy(
            bucket_offsets,
            host_bucket_offsets,
            sizeof(int) * (bucket_count + 1),
            cudaMemcpyHostToDevice
        );

        delete[] host_bucket_offsets;
        if (status != cudaSuccess) return status;
    }

    if (num_rows == 0) return cudaSuccess;

    const int block_size = 256;
    const int num_blocks = (num_rows + block_size - 1) / block_size;
    nnz_per_row<<<num_blocks, block_size>>>(rowptr, row_nnz, num_rows);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;

    thrust::device_ptr<int> row_nnz_ptr(row_nnz);
    thrust::device_ptr<int> row_index_ptr(row_indices);
    thrust::sequence(row_index_ptr, row_index_ptr + num_rows);
    thrust::stable_sort_by_key(row_nnz_ptr, row_nnz_ptr + num_rows, row_index_ptr);

    return cudaSuccess;
}
