#pragma once

#include <CellShard/core/types.cuh>
#include <Cellerator/common/compiler_annotations.hh>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

namespace cellshard {

// Host COO entries for CellShard ingest and packing, in their supplied order.
struct alignas(16) triplet {
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;

    void *storage;
    types::idx_t *rowIdx;
    types::idx_t *colIdx;
    real::storage_t *val;
};

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE void init(
    triplet * __restrict__ m,
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

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE size_t bytes(const triplet * __restrict__ m) {
    return sizeof(*m)
        + (size_t) m->nnz * sizeof(types::idx_t)
        + (size_t) m->nnz * sizeof(types::idx_t)
        + (size_t) m->nnz * sizeof(real::storage_t);
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE const real::storage_t *at(const triplet * __restrict__ m, types::idx_t r, types::idx_t c) {
    for (types::nnz_t i = 0; i < m->nnz; ++i) {
        if (m->rowIdx[i] == r && m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE real::storage_t *at(triplet * __restrict__ m, types::idx_t r, types::idx_t c) {
    return const_cast<real::storage_t *>(at(static_cast<const triplet *>(m), r, c));
}

CELLERATOR_CUDA_FORCEINLINE void clear(triplet * __restrict__ m) {
    if (m->storage != 0) free(m->storage);
    else {
        free(m->rowIdx);
        free(m->colIdx);
        free(m->val);
    }
    init(m);
}

CELLERATOR_CUDA_FORCEINLINE int allocate(triplet * __restrict__ m) {
    const size_t row_bytes = (size_t) m->nnz * sizeof(types::idx_t);
    const size_t col_offset = ((row_bytes + alignof(types::idx_t) - 1u) / alignof(types::idx_t)) * alignof(types::idx_t);
    const size_t val_offset = ((col_offset + (size_t) m->nnz * sizeof(types::idx_t) + alignof(real::storage_t) - 1u) / alignof(real::storage_t)) * alignof(real::storage_t);
    const size_t total_bytes = val_offset + (size_t) m->nnz * sizeof(real::storage_t);
    void *storage = 0;

    if (m->storage != 0) free(m->storage);
    else {
        free(m->rowIdx);
        free(m->colIdx);
        free(m->val);
    }
    m->storage = 0;
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
    if (m->nnz == 0u) return 1;
    storage = malloc(total_bytes);
    if (storage == 0) return 0;
    m->storage = storage;
    m->rowIdx = (types::idx_t *) storage;
    m->colIdx = (types::idx_t *) ((char *) storage + col_offset);
    m->val = (real::storage_t *) ((char *) storage + val_offset);
    return 1;
}

CELLERATOR_CUDA_FORCEINLINE int concatenate_rows(triplet * __restrict__ dst, const triplet * __restrict__ top, const triplet * __restrict__ bottom) {
    if (top->cols != 0u && bottom->cols != 0u && top->cols != bottom->cols) {
        fprintf(stderr, "Error: cannot concatenate coo matrices with different column counts\n");
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
        __builtin_memcpy(dst->rowIdx, top->rowIdx, (size_t) top->nnz * sizeof(types::idx_t));
        __builtin_memcpy(dst->colIdx, top->colIdx, (size_t) top->nnz * sizeof(types::idx_t));
        __builtin_memcpy(dst->val, top->val, (size_t) top->nnz * sizeof(real::storage_t));
    }
    for (types::nnz_t i = 0; i < bottom->nnz; ++i) dst->rowIdx[top->nnz + i] = bottom->rowIdx[i] + oldRows;
    if (bottom->nnz != 0u) {
        __builtin_memcpy(dst->colIdx + top->nnz, bottom->colIdx, (size_t) bottom->nnz * sizeof(types::idx_t));
        __builtin_memcpy(dst->val + top->nnz, bottom->val, (size_t) bottom->nnz * sizeof(real::storage_t));
    }
    return 1;
}

CELLERATOR_CUDA_FORCEINLINE int append_rows(triplet * __restrict__ dst, const triplet * __restrict__ src) {
    triplet merged;
    init(&merged);
    if (!concatenate_rows(&merged, dst, src)) return 0;
    clear(dst);
    *dst = merged;
    return 1;
}

} // namespace cellshard
