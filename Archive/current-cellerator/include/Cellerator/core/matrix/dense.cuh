#pragma once

#include <cstdlib>

#include "../types.cuh"

namespace cellerator::core::matrix {

enum {
    dense_row_major = 0u,
    dense_col_major = 1u,
    dense_host_registered = 1u << 0
};

struct alignas(16) dense {
    types::dim_t rows;
    types::dim_t cols;
    types::u32 stride;
    types::u32 order;
    types::u32 flags;

    void *storage;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ types::u32 packed_stride(types::dim_t rows, types::dim_t cols, types::u32 order) {
    return order == dense_col_major ? rows : cols;
}

__host__ __device__ __forceinline__ void init(
    dense * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::u32 order = dense_row_major,
    types::u32 stride = 0) {
    m->rows = rows;
    m->cols = cols;
    m->stride = stride != 0u ? stride : packed_stride(rows, cols, order);
    m->order = order;
    m->flags = 0u;
    m->storage = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ void attach(
    dense * __restrict__ m,
    types::dim_t rows,
    types::dim_t cols,
    real::storage_t * __restrict__ val,
    types::u32 order = dense_row_major,
    types::u32 stride = 0) {
    init(m, rows, cols, order, stride);
    m->val = val;
}

__host__ __device__ __forceinline__ std::size_t payload_elements(const dense * __restrict__ m) {
    if (m == 0) return 0u;
    return m->order == dense_col_major
        ? (std::size_t) m->cols * (std::size_t) m->stride
        : (std::size_t) m->rows * (std::size_t) m->stride;
}

__host__ __device__ __forceinline__ std::size_t payload_bytes(const dense * __restrict__ m) {
    return payload_elements(m) * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ std::size_t bytes(const dense * __restrict__ m) {
    return sizeof(*m) + payload_bytes(m);
}

__host__ __device__ __forceinline__ std::size_t offset(const dense * __restrict__ m, types::dim_t r, types::idx_t c) {
    return m->order == dense_col_major
        ? (std::size_t) c * (std::size_t) m->stride + r
        : (std::size_t) r * (std::size_t) m->stride + c;
}

__host__ __device__ __forceinline__ const real::storage_t *at(const dense * __restrict__ m, types::dim_t r, types::idx_t c) {
    if (m == 0 || m->val == 0 || r >= m->rows || c >= m->cols) return 0;
    return m->val + offset(m, r, c);
}

__host__ __device__ __forceinline__ real::storage_t *at(dense * __restrict__ m, types::dim_t r, types::idx_t c) {
    return const_cast<real::storage_t *>(at(static_cast<const dense *>(m), r, c));
}

__host__ __forceinline__ void clear(dense * __restrict__ m) {
    if ((m->flags & dense_host_registered) != 0u && m->val != 0) cudaHostUnregister(m->val);
    if (m->storage != 0) std::free(m->storage);
    init(m);
}

__host__ __forceinline__ int allocate(dense * __restrict__ m) {
    const std::size_t total_bytes = payload_bytes(m);
    void *storage = 0;

    if ((m->flags & dense_host_registered) != 0u && m->val != 0) cudaHostUnregister(m->val);
    if (m->storage != 0) std::free(m->storage);
    m->flags = 0u;
    m->storage = 0;
    m->val = 0;
    if (total_bytes == 0u) return 1;
    storage = std::malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->val = (real::storage_t *) storage;
    return 1;
}

__host__ __forceinline__ int pin(dense * __restrict__ m) {
    const std::size_t total_bytes = payload_bytes(m);
    cudaError_t err = cudaSuccess;

    if (m == 0 || m->val == 0 || total_bytes == 0u) return 1;
    if ((m->flags & dense_host_registered) != 0u) return 1;
    err = cudaHostRegister(m->val, total_bytes, cudaHostRegisterPortable);
    if (err == cudaSuccess) {
        m->flags |= dense_host_registered;
        return 1;
    }
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        m->flags |= dense_host_registered;
        return 1;
    }
    cudaGetLastError();
    return 0;
}

__host__ __forceinline__ void unpin(dense * __restrict__ m) {
    if (m == 0 || (m->flags & dense_host_registered) == 0u || m->val == 0) return;
    cudaHostUnregister(m->val);
    m->flags &= ~dense_host_registered;
}

__host__ __device__ __forceinline__ int host_registered(const dense * __restrict__ m) {
    return m != 0 && (m->flags & dense_host_registered) != 0u;
}

} // namespace cellerator::core::matrix
