#pragma once

#include <stddef.h>
#include <Cellerator/common/compiler_annotations.hh>
#include <Cellerator/memory/cuda/host_registration.hh>
#include <stdlib.h>

#include <CellShard/core/types.cuh>

namespace cellshard {

enum {
    dense_row_major = 0u,
    dense_col_major = 1u,
    dense_host_registered = 1u << 0
};

// Host payload for CellShard dense fallback storage and staging.
struct alignas(16) dense {
    types::dim_t rows;
    types::dim_t cols;
    types::u32 stride;
    types::u32 order;
    types::u32 flags;

    void *storage;
    real::storage_t *val;
};

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE types::u32 packed_stride(types::dim_t rows, types::dim_t cols, types::u32 order) {
    return order == dense_col_major ? rows : cols;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE void init(
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

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE void attach(
    dense * __restrict__ m,
    types::dim_t rows,
    types::dim_t cols,
    real::storage_t * __restrict__ val,
    types::u32 order = dense_row_major,
    types::u32 stride = 0) {
    init(m, rows, cols, order, stride);
    m->val = val;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE size_t payload_elements(const dense * __restrict__ m) {
    if (m == 0) return 0u;
    return m->order == dense_col_major
        ? (size_t) m->cols * (size_t) m->stride
        : (size_t) m->rows * (size_t) m->stride;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE size_t payload_bytes(const dense * __restrict__ m) {
    return payload_elements(m) * sizeof(real::storage_t);
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE size_t bytes(const dense * __restrict__ m) {
    return sizeof(*m) + payload_bytes(m);
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE size_t offset(const dense * __restrict__ m, types::dim_t r, types::idx_t c) {
    return m->order == dense_col_major
        ? (size_t) c * (size_t) m->stride + r
        : (size_t) r * (size_t) m->stride + c;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE const real::storage_t *at(const dense * __restrict__ m, types::dim_t r, types::idx_t c) {
    if (m == 0 || m->val == 0 || r >= m->rows || c >= m->cols) return 0;
    return m->val + offset(m, r, c);
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE real::storage_t *at(dense * __restrict__ m, types::dim_t r, types::idx_t c) {
    return const_cast<real::storage_t *>(at(static_cast<const dense *>(m), r, c));
}

CELLERATOR_CUDA_FORCEINLINE void clear(dense * __restrict__ m) {
    if ((m->flags & dense_host_registered) != 0u && m->val != 0) ::cellerator::memory::cuda::unregister_host_memory(m->val);
    if (m->storage != 0) free(m->storage);
    init(m);
}

CELLERATOR_CUDA_FORCEINLINE int allocate(dense * __restrict__ m) {
    const size_t total_bytes = payload_bytes(m);
    void *storage = 0;

    if ((m->flags & dense_host_registered) != 0u && m->val != 0) ::cellerator::memory::cuda::unregister_host_memory(m->val);
    if (m->storage != 0) free(m->storage);
    m->flags = 0u;
    m->storage = 0;
    m->val = 0;
    if (total_bytes == 0u) return 1;
    storage = malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->val = (real::storage_t *) storage;
    return 1;
}

CELLERATOR_CUDA_FORCEINLINE int pin(dense * __restrict__ m) {
    const size_t total_bytes = payload_bytes(m);

    if (m == 0 || m->val == 0 || total_bytes == 0u) return 1;
    if ((m->flags & dense_host_registered) != 0u) return 1;
    if (!::cellerator::memory::cuda::register_host_memory(m->val, total_bytes)) return 0;
    m->flags |= dense_host_registered;
    return 1;
}

CELLERATOR_CUDA_FORCEINLINE void unpin(dense * __restrict__ m) {
    if (m == 0 || (m->flags & dense_host_registered) == 0u || m->val == 0) return;
    ::cellerator::memory::cuda::unregister_host_memory(m->val);
    m->flags &= ~dense_host_registered;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE int host_registered(const dense * __restrict__ m) {
    return m != 0 && (m->flags & dense_host_registered) != 0u;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE int dense_is_packed_row_major(const dense *m) {
    return m != 0 && m->order == dense_row_major && m->stride == m->cols;
}

} // namespace cellshard
