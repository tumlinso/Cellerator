#pragma once

#include "../types.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cellerator::core::matrix {

struct alignas(16) coo {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;

    void *storage;
    types::idx_t *rowIdx;
    types::idx_t *colIdx;
    real::storage_t *val;
};

__host__ __device__ __forceinline__ void init(
    coo * __restrict__ m,
    types::dim_t rows = 0,
    types::dim_t cols = 0,
    types::nnz_t nnz = 0) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->storage = 0;
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
}

__host__ __device__ __forceinline__ std::size_t bytes(const coo * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(types::idx_t)
        + (std::size_t) m->nnz * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ const real::storage_t *at(const coo * __restrict__ m, types::idx_t r, types::idx_t c) {
    for (types::nnz_t i = 0; i < m->nnz; ++i) {
        if (m->rowIdx[i] == r && m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

__host__ __device__ __forceinline__ real::storage_t *at(coo * __restrict__ m, types::idx_t r, types::idx_t c) {
    return const_cast<real::storage_t *>(at(static_cast<const coo *>(m), r, c));
}

__host__ __forceinline__ void clear(coo * __restrict__ m) {
    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->rowIdx);
        std::free(m->colIdx);
        std::free(m->val);
    }
    init(m);
}

__host__ __forceinline__ int allocate(coo * __restrict__ m) {
    const std::size_t row_bytes = (std::size_t) m->nnz * sizeof(types::idx_t);
    const std::size_t col_offset = ((row_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const std::size_t val_offset = ((col_offset + (std::size_t) m->nnz * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const std::size_t total_bytes = val_offset + (std::size_t) m->nnz * sizeof(real::storage_t);
    void *storage = 0;

    if (m->storage != 0) std::free(m->storage);
    else {
        std::free(m->rowIdx);
        std::free(m->colIdx);
        std::free(m->val);
    }
    m->storage = 0;
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
    if (m->nnz == 0u) return 1;
    storage = std::malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->rowIdx = (types::idx_t *) storage;
    m->colIdx = (types::idx_t *) ((char *) storage + col_offset);
    m->val = (real::storage_t *) ((char *) storage + val_offset);
    return 1;
}

__host__ __forceinline__ int concatenate_rows(coo * __restrict__ dst, const coo * __restrict__ top, const coo * __restrict__ bottom) {
    if (top->cols != 0u && bottom->cols != 0u && top->cols != bottom->cols) {
        std::fprintf(stderr, "Error: cannot concatenate coo matrices with different column counts\n");
        return 0;
    }

    const types::dim_t oldRows = top->rows;
    dst->rows = top->rows + bottom->rows;
    dst->cols = top->cols != 0u ? top->cols : bottom->cols;
    dst->nnz = top->nnz + bottom->nnz;
    dst->storage = 0;
    dst->rowIdx = 0;
    dst->colIdx = 0;
    dst->val = 0;
    if (!allocate(dst)) return 0;

    if (top->nnz != 0u) {
        std::memcpy(dst->rowIdx, top->rowIdx, (std::size_t) top->nnz * sizeof(types::idx_t));
        std::memcpy(dst->colIdx, top->colIdx, (std::size_t) top->nnz * sizeof(types::idx_t));
        std::memcpy(dst->val, top->val, (std::size_t) top->nnz * sizeof(real::storage_t));
    }
    for (types::nnz_t i = 0; i < bottom->nnz; ++i) dst->rowIdx[top->nnz + i] = bottom->rowIdx[i] + oldRows;
    if (bottom->nnz != 0u) {
        std::memcpy(dst->colIdx + top->nnz, bottom->colIdx, (std::size_t) bottom->nnz * sizeof(types::idx_t));
        std::memcpy(dst->val + top->nnz, bottom->val, (std::size_t) bottom->nnz * sizeof(real::storage_t));
    }
    return 1;
}

__host__ __forceinline__ int append_rows(coo * __restrict__ dst, const coo * __restrict__ src) {
    coo merged;
    init(&merged);
    if (!concatenate_rows(&merged, dst, src)) return 0;
    clear(dst);
    *dst = merged;
    return 1;
}

} // namespace cellerator::core::matrix
