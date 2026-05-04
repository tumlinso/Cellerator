#pragma once

#include "../types.cuh"

#include <cstdlib>

namespace cellerator::core::matrix {

enum {
    compressed_by_row = 0,
    compressed_by_col = 1,
    compressed_host_registered = 1u << 0
};

struct alignas(16) compressed {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
    types::u32 axis;
    types::u32 flags;

    void *storage;
    types::ptr_t *majorPtr;
    types::idx_t *minorIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ types::dim_t major_dim(const compressed * __restrict__ m) {
    return m->axis == compressed_by_col ? m->cols : m->rows;
}

__host__ __device__ __forceinline__ types::dim_t minor_dim(const compressed * __restrict__ m) {
    return m->axis == compressed_by_col ? m->rows : m->cols;
}

__host__ __device__ __forceinline__ void init(
    compressed * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0,
    types::u32 axis = compressed_by_row) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->axis = axis;
    m->flags = 0u;
    m->storage = 0;
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const compressed * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) (major_dim(m) + 1u) * sizeof(types::ptr_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const compressed * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::dim_t major = m->axis == compressed_by_col ? c : r;
    const types::idx_t minor = m->axis == compressed_by_col ? r : c;
    const types::ptr_t begin = m->majorPtr[major];
    const types::ptr_t end = m->majorPtr[major + 1u];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->minorIdx[i] == minor) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(compressed * __restrict__ m, types::dim_t r, types::idx_t c) {
    return const_cast<real::storage_t *>(at(static_cast<const compressed *>(m), r, c));
}

__host__ __forceinline__ void clear(compressed * __restrict__ m) {
    if ((m->flags & compressed_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->majorPtr);
        std::free(m->minorIdx);
        std::free(m->val);
    }
    init(m);
}

__host__ __forceinline__ int allocate(compressed * __restrict__ m) {
    const std::size_t ptr_count = (std::size_t) major_dim(m) + 1u;
    const std::size_t major_bytes = ptr_count * sizeof(types::ptr_t);
    const std::size_t minor_offset = ((major_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const std::size_t val_offset = ((minor_offset + (std::size_t) m->nnz * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const std::size_t total_bytes = val_offset + (std::size_t) m->nnz * sizeof(real::storage_t);
    void *storage = 0;

    if ((m->flags & compressed_host_registered) != 0u && m->storage != 0) cudaHostUnregister(m->storage);
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->majorPtr);
        std::free(m->minorIdx);
        std::free(m->val);
    }
    m->flags = 0u;
    m->storage = 0;
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;

    if (total_bytes == 0u) return 1;
    storage = std::malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->majorPtr = ptr_count != 0u ? (types::ptr_t *) storage : 0;
    m->minorIdx = m->nnz != 0u ? (types::idx_t *) ((char *) storage + minor_offset) : 0;
    m->val = m->nnz != 0u ? (real::storage_t *) ((char *) storage + val_offset) : 0;
    return 1;
}

__host__ __forceinline__ int pin(compressed * __restrict__ m) {
    const std::size_t ptr_count = (std::size_t) major_dim(m) + 1u;
    const std::size_t major_bytes = ptr_count * sizeof(types::ptr_t);
    const std::size_t minor_offset = ((major_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const std::size_t val_offset = ((minor_offset + (std::size_t) m->nnz * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const std::size_t total_bytes = val_offset + (std::size_t) m->nnz * sizeof(real::storage_t);
    cudaError_t err = cudaSuccess;

    if (m->storage == 0 || total_bytes == 0u) return 1;
    if ((m->flags & compressed_host_registered) != 0u) return 1;
    err = cudaHostRegister(m->storage, total_bytes, cudaHostRegisterPortable);
    if (err == cudaSuccess) {
        m->flags |= compressed_host_registered;
        return 1;
    }
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        cudaGetLastError();
        m->flags |= compressed_host_registered;
        return 1;
    }
    cudaGetLastError();
    return 0;
}

__host__ __forceinline__ void unpin(compressed * __restrict__ m) {
    if ((m->flags & compressed_host_registered) == 0u || m->storage == 0) return;
    cudaHostUnregister(m->storage);
    m->flags &= ~compressed_host_registered;
}

__host__ __device__ __forceinline__ int host_registered(const compressed * __restrict__ m) {
    return (m->flags & compressed_host_registered) != 0u;
}

} // namespace cellerator::core::matrix
