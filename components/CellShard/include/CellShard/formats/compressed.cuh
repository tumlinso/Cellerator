#pragma once

#include <CellShard/core/types.cuh>
#include <Cellerator/common/compiler_annotations.hh>
#include <Cellerator/memory/cuda/host_registration.hh>

#include <stddef.h>
#include <stdlib.h>

namespace cellshard {

enum {
    compressed_by_row = 0,
    compressed_by_col = 1,
    compressed_host_registered = 1u << 0
};

// Host payload for the CellShard compressed ingest and pack formats.
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

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE types::dim_t major_dim(const compressed * __restrict__ m) {
    return m->axis == compressed_by_col ? m->cols : m->rows;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE types::dim_t minor_dim(const compressed * __restrict__ m) {
    return m->axis == compressed_by_col ? m->rows : m->cols;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE void init(
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

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE size_t bytes(const compressed * __restrict__ m) {
    return sizeof(*m)
        + (size_t) (major_dim(m) + 1u) * sizeof(types::ptr_t)
        + (size_t) m->nnz * sizeof(types::idx_t)
        + (size_t) m->nnz * sizeof(real::storage_t);
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE const real::storage_t *at(const compressed * __restrict__ m, types::dim_t r, types::idx_t c) {
    const types::dim_t major = m->axis == compressed_by_col ? c : r;
    const types::idx_t minor = m->axis == compressed_by_col ? r : c;
    const types::ptr_t begin = m->majorPtr[major];
    const types::ptr_t end = m->majorPtr[major + 1u];
    for (types::ptr_t i = begin; i < end; ++i) {
        if (m->minorIdx[i] == minor) return m->val + i;
    }
    return 0;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE real::storage_t *at(compressed * __restrict__ m, types::dim_t r, types::idx_t c) {
    return const_cast<real::storage_t *>(at(static_cast<const compressed *>(m), r, c));
}

CELLERATOR_CUDA_FORCEINLINE void clear(compressed * __restrict__ m) {
    if ((m->flags & compressed_host_registered) != 0u && m->storage != 0) ::cellerator::memory::cuda::unregister_host_memory(m->storage);
    if (m->storage != 0) free(m->storage);
    else {
        free(m->majorPtr);
        free(m->minorIdx);
        free(m->val);
    }
    init(m);
}

CELLERATOR_CUDA_FORCEINLINE int allocate(compressed * __restrict__ m) {
    const size_t ptr_count = (size_t) major_dim(m) + 1u;
    const size_t major_bytes = ptr_count * sizeof(types::ptr_t);
    const size_t minor_offset = ((major_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const size_t val_offset = ((minor_offset + (size_t) m->nnz * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const size_t total_bytes = val_offset + (size_t) m->nnz * sizeof(real::storage_t);
    void *storage = 0;

    if ((m->flags & compressed_host_registered) != 0u && m->storage != 0) ::cellerator::memory::cuda::unregister_host_memory(m->storage);
    if (m->storage != 0) free(m->storage);
    else {
        free(m->majorPtr);
        free(m->minorIdx);
        free(m->val);
    }
    m->flags = 0u;
    m->storage = 0;
    m->majorPtr = 0;
    m->minorIdx = 0;
    m->val = 0;

    if (total_bytes == 0u) return 1;
    storage = malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->majorPtr = ptr_count != 0u ? (types::ptr_t *) storage : 0;
    m->minorIdx = m->nnz != 0u ? (types::idx_t *) ((char *) storage + minor_offset) : 0;
    m->val = m->nnz != 0u ? (real::storage_t *) ((char *) storage + val_offset) : 0;
    return 1;
}

CELLERATOR_CUDA_FORCEINLINE int pin(compressed * __restrict__ m) {
    const size_t ptr_count = (size_t) major_dim(m) + 1u;
    const size_t major_bytes = ptr_count * sizeof(types::ptr_t);
    const size_t minor_offset = ((major_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const size_t val_offset = ((minor_offset + (size_t) m->nnz * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const size_t total_bytes = val_offset + (size_t) m->nnz * sizeof(real::storage_t);

    if (m->storage == 0 || total_bytes == 0u) return 1;
    if ((m->flags & compressed_host_registered) != 0u) return 1;
    if (!::cellerator::memory::cuda::register_host_memory(m->storage, total_bytes)) return 0;
    m->flags |= compressed_host_registered;
    return 1;
}

CELLERATOR_CUDA_FORCEINLINE void unpin(compressed * __restrict__ m) {
    if ((m->flags & compressed_host_registered) == 0u || m->storage == 0) return;
    ::cellerator::memory::cuda::unregister_host_memory(m->storage);
    m->flags &= ~compressed_host_registered;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE int host_registered(const compressed * __restrict__ m) {
    return (m->flags & compressed_host_registered) != 0u;
}

} // namespace cellshard
