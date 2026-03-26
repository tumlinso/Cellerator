#pragma once

#include "../matrix.cuh"

namespace matrix {
namespace sparse {

template<typename Real = ::matrix::Real>
struct alignas(16) coo {
    typedef Real value_type;
    typedef Index index_type;

    Index rows;
    Index cols;
    Index nnz;
    unsigned char format;

    Index *rowIdx;
    Index *colIdx;
    Real *val;
};

template<typename Real>
__host__ __device__ __forceinline__ void init(coo<Real> * __restrict__ m, Index rows = 0, Index cols = 0, Index nnz = 0) {
    require_fp_storage<Real>();
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->format = format_coo;
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
}

template<typename Real>
__host__ __device__ __forceinline__ std::size_t bytes(const coo<Real> * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->nnz * sizeof(Index)
        + (std::size_t) m->nnz * sizeof(Index)
        + (std::size_t) m->nnz * sizeof(Real);
}

template<typename Real>
__host__ __forceinline__ void clear(coo<Real> * __restrict__ m) {
    std::free(m->rowIdx);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_coo;
}

template<typename Real>
__host__ __forceinline__ int allocate(coo<Real> * __restrict__ m) {
    std::free(m->rowIdx);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowIdx = 0;
    m->colIdx = 0;
    m->val = 0;
    if (m->nnz == 0) return 1;
    m->rowIdx = (Index *) std::malloc((std::size_t) m->nnz * sizeof(Index));
    m->colIdx = (Index *) std::malloc((std::size_t) m->nnz * sizeof(Index));
    m->val = (Real *) std::malloc((std::size_t) m->nnz * sizeof(Real));
    if (m->rowIdx == 0 || m->colIdx == 0 || m->val == 0) {
        std::free(m->rowIdx);
        std::free(m->colIdx);
        std::free(m->val);
        m->rowIdx = 0;
        m->colIdx = 0;
        m->val = 0;
        return 0;
    }
    return 1;
}

template<typename Real>
__host__ __device__ __forceinline__ const Real *at(const coo<Real> * __restrict__ m, Index r, Index c) {
    for (Index i = 0; i < m->nnz; ++i) {
        if (m->rowIdx[i] == r && m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

template<typename Real>
__host__ __device__ __forceinline__ Real *at(coo<Real> * __restrict__ m, Index r, Index c) {
    for (Index i = 0; i < m->nnz; ++i) {
        if (m->rowIdx[i] == r && m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

template<typename Real>
__host__ __forceinline__ int concatenate_rows(coo<Real> * __restrict__ dst, const coo<Real> * __restrict__ top, const coo<Real> * __restrict__ bottom) {
    if (top->cols != 0 && bottom->cols != 0 && top->cols != bottom->cols) {
        std::fprintf(stderr, "Error: cannot concatenate coo matrices with different column counts\n");
        return 0;
    }

    Index oldRows = top->rows;
    dst->rows = top->rows + bottom->rows;
    dst->cols = top->cols != 0 ? top->cols : bottom->cols;
    dst->nnz = top->nnz + bottom->nnz;
    dst->format = format_coo;
    dst->rowIdx = 0;
    dst->colIdx = 0;
    dst->val = 0;
    if (!allocate(dst)) return 0;

    if (top->nnz != 0) {
        std::memcpy(dst->rowIdx, top->rowIdx, top->nnz * sizeof(Index));
        std::memcpy(dst->colIdx, top->colIdx, top->nnz * sizeof(Index));
        std::memcpy(dst->val, top->val, top->nnz * sizeof(Real));
    }
    for (Index i = 0; i < bottom->nnz; ++i) dst->rowIdx[top->nnz + i] = bottom->rowIdx[i] + oldRows;
    if (bottom->nnz != 0) {
        std::memcpy(dst->colIdx + top->nnz, bottom->colIdx, bottom->nnz * sizeof(Index));
        std::memcpy(dst->val + top->nnz, bottom->val, bottom->nnz * sizeof(Real));
    }
    return 1;
}

template<typename Real>
__host__ __forceinline__ int append_rows(coo<Real> * __restrict__ dst, const coo<Real> * __restrict__ src) {
    if (dst->cols != 0 && src->cols != 0 && dst->cols != src->cols) {
        std::fprintf(stderr, "Error: cannot concatenate coo matrices with different column counts\n");
        return 0;
    }

    Index oldRows = dst->rows;
    Index oldNnz = dst->nnz;
    Index newNnz = dst->nnz + src->nnz;
    Index *rowIdx = 0;
    Index *colIdx = 0;
    Real *val = 0;

    dst->rows += src->rows;
    if (dst->cols == 0) dst->cols = src->cols;
    dst->nnz = newNnz;
    dst->format = format_coo;
    if (newNnz != 0) {
        rowIdx = (Index *) std::malloc((std::size_t) newNnz * sizeof(Index));
        colIdx = (Index *) std::malloc((std::size_t) newNnz * sizeof(Index));
        val = (Real *) std::malloc((std::size_t) newNnz * sizeof(Real));
        if (rowIdx == 0 || colIdx == 0 || val == 0) {
            std::free(rowIdx);
            std::free(colIdx);
            std::free(val);
            dst->rows = oldRows;
            dst->nnz = oldNnz;
            return 0;
        }
    }

    if (oldNnz != 0) {
        std::memcpy(rowIdx, dst->rowIdx, (std::size_t) oldNnz * sizeof(Index));
        std::memcpy(colIdx, dst->colIdx, (std::size_t) oldNnz * sizeof(Index));
        std::memcpy(val, dst->val, (std::size_t) oldNnz * sizeof(Real));
    }
    std::free(dst->rowIdx);
    std::free(dst->colIdx);
    std::free(dst->val);
    dst->rowIdx = rowIdx;
    dst->colIdx = colIdx;
    dst->val = val;

    for (Index i = 0; i < src->nnz; ++i) dst->rowIdx[oldNnz + i] = src->rowIdx[i] + oldRows;
    if (src->nnz != 0) {
        std::memcpy(dst->colIdx + oldNnz, src->colIdx, src->nnz * sizeof(Index));
        std::memcpy(dst->val + oldNnz, src->val, src->nnz * sizeof(Real));
    }
    return 1;
}

} // namespace sparse
} // namespace matrix
