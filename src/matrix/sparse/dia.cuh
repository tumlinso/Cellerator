#pragma once

#include "../matrix.cuh"

namespace matrix {
namespace sparse {

template<typename Real = ::matrix::Real>
struct alignas(16) dia {
    typedef Real value_type;
    typedef Index index_type;

    Index rows;
    Index cols;
    Index nnz;
    unsigned char format;

    DiagIndex *offsets;
    Real *val;
    Index num_diagonals;
};

template<typename Real>
__host__ __device__ __forceinline__ void init(dia<Real> * __restrict__ m, Index rows = 0, Index cols = 0, Index nnz = 0) {
    require_fp_storage<Real>();
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->format = format_dia;
    m->offsets = 0;
    m->val = 0;
    m->num_diagonals = 0;
}

template<typename Real>
__host__ __device__ __forceinline__ std::size_t bytes(const dia<Real> * __restrict__ m) {
    return sizeof(*m)
        + (std::size_t) m->num_diagonals * sizeof(DiagIndex)
        + (std::size_t) m->nnz * sizeof(Real);
}

template<typename Real>
__host__ __forceinline__ void clear(dia<Real> * __restrict__ m) {
    std::free(m->offsets);
    std::free(m->val);
    m->offsets = 0;
    m->val = 0;
    m->num_diagonals = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_dia;
}

template<typename Real>
__host__ __forceinline__ int allocate(dia<Real> * __restrict__ m) {
    std::free(m->offsets);
    std::free(m->val);
    m->offsets = 0;
    m->val = 0;
    if (m->num_diagonals != 0) m->offsets = (DiagIndex *) std::malloc((std::size_t) m->num_diagonals * sizeof(DiagIndex));
    if (m->nnz != 0) m->val = (Real *) std::malloc((std::size_t) m->nnz * sizeof(Real));
    if (m->num_diagonals != 0 && m->offsets == 0) return 0;
    if (m->nnz != 0 && m->val == 0) {
        std::free(m->offsets);
        m->offsets = 0;
        return 0;
    }
    return 1;
}

template<typename Real>
__host__ __device__ __forceinline__ const Real *at(const dia<Real> * __restrict__ m, Index r, Index c) {
    const DiagIndex offset = (DiagIndex) c - (DiagIndex) r;
    for (Index i = 0; i < m->num_diagonals; ++i) {
        if (m->offsets[i] == offset) return m->val + i * m->rows + r;
    }
    return 0;
}

template<typename Real>
__host__ __device__ __forceinline__ Real *at(dia<Real> * __restrict__ m, Index r, Index c) {
    const DiagIndex offset = (DiagIndex) c - (DiagIndex) r;
    for (Index i = 0; i < m->num_diagonals; ++i) {
        if (m->offsets[i] == offset) return m->val + i * m->rows + r;
    }
    return 0;
}

} // namespace sparse
} // namespace matrix
