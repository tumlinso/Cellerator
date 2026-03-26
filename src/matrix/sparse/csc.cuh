#pragma once

#include "../matrix.cuh"

namespace matrix {
namespace sparse {

template<typename Real = ::matrix::Real>
struct csc {
    typedef Real value_type;
    typedef Index index_type;

    Index rows;
    Index cols;
    Index nnz;
    unsigned char format;

    Index *colPtr;
    Index *rowIdx;
    Real *val;
};

template<typename Real>
inline void init(csc<Real> *m, Index rows = 0, Index cols = 0, Index nnz = 0) {
    require_fp_storage<Real>();
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->format = format_csc;
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
}

template<typename Real>
inline std::size_t bytes(const csc<Real> *m) {
    return sizeof(*m)
        + (std::size_t) (m->cols + 1) * sizeof(Index)
        + (std::size_t) m->nnz * sizeof(Index)
        + (std::size_t) m->nnz * sizeof(Real);
}

template<typename Real>
inline void clear(csc<Real> *m) {
    std::free(m->colPtr);
    std::free(m->rowIdx);
    std::free(m->val);
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_csc;
}

template<typename Real>
inline int allocate(csc<Real> *m) {
    std::free(m->colPtr);
    std::free(m->rowIdx);
    std::free(m->val);
    m->colPtr = 0;
    m->rowIdx = 0;
    m->val = 0;
    if (m->cols != 0) m->colPtr = (Index *) std::malloc((std::size_t) (m->cols + 1) * sizeof(Index));
    if (m->nnz != 0) {
        m->rowIdx = (Index *) std::malloc((std::size_t) m->nnz * sizeof(Index));
        m->val = (Real *) std::malloc((std::size_t) m->nnz * sizeof(Real));
    }
    if (m->cols != 0 && m->colPtr == 0) return 0;
    if (m->nnz != 0 && (m->rowIdx == 0 || m->val == 0)) {
        std::free(m->colPtr);
        std::free(m->rowIdx);
        std::free(m->val);
        m->colPtr = 0;
        m->rowIdx = 0;
        m->val = 0;
        return 0;
    }
    return 1;
}

template<typename Real>
inline const Real *at(const csc<Real> *m, Index r, Index c) {
    for (Index i = m->colPtr[c]; i < m->colPtr[c + 1]; ++i) {
        if (m->rowIdx[i] == r) return m->val + i;
    }
    return 0;
}

template<typename Real>
inline Real *at(csc<Real> *m, Index r, Index c) {
    for (Index i = m->colPtr[c]; i < m->colPtr[c + 1]; ++i) {
        if (m->rowIdx[i] == r) return m->val + i;
    }
    return 0;
}

} // namespace sparse
} // namespace matrix
