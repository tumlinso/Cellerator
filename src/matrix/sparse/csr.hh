#pragma once

namespace matrix {
namespace sparse {

template<typename Real = ::matrix::Real>
struct csr {
    typedef Real value_type;
    typedef Index index_type;

    Index rows;
    Index cols;
    Index nnz;
    unsigned char format;

    Index *rowPtr;
    Index *colIdx;
    Real *val;
};

template<typename Real>
inline void init(csr<Real> *m, Index rows = 0, Index cols = 0, Index nnz = 0) {
    require_fp_storage<Real>();
    m->rows = rows;
    m->cols = cols;
    m->nnz = nnz;
    m->format = format_csr;
    m->rowPtr = 0;
    m->colIdx = 0;
    m->val = 0;
}

template<typename Real>
inline std::size_t bytes(const csr<Real> *m) {
    return sizeof(*m)
        + (std::size_t) (m->rows + 1) * sizeof(Index)
        + (std::size_t) m->nnz * sizeof(Index)
        + (std::size_t) m->nnz * sizeof(Real);
}

template<typename Real>
inline void clear(csr<Real> *m) {
    std::free(m->rowPtr);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowPtr = 0;
    m->colIdx = 0;
    m->val = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_csr;
}

template<typename Real>
inline int allocate(csr<Real> *m) {
    std::free(m->rowPtr);
    std::free(m->colIdx);
    std::free(m->val);
    m->rowPtr = 0;
    m->colIdx = 0;
    m->val = 0;
    if (m->rows != 0) m->rowPtr = (Index *) std::malloc((std::size_t) (m->rows + 1) * sizeof(Index));
    if (m->nnz != 0) {
        m->colIdx = (Index *) std::malloc((std::size_t) m->nnz * sizeof(Index));
        m->val = (Real *) std::malloc((std::size_t) m->nnz * sizeof(Real));
    }
    if (m->rows != 0 && m->rowPtr == 0) return 0;
    if (m->nnz != 0 && (m->colIdx == 0 || m->val == 0)) {
        std::free(m->rowPtr);
        std::free(m->colIdx);
        std::free(m->val);
        m->rowPtr = 0;
        m->colIdx = 0;
        m->val = 0;
        return 0;
    }
    return 1;
}

template<typename Real>
inline const Real *at(const csr<Real> *m, Index r, Index c) {
    for (Index i = m->rowPtr[r]; i < m->rowPtr[r + 1]; ++i) {
        if (m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

template<typename Real>
inline Real *at(csr<Real> *m, Index r, Index c) {
    for (Index i = m->rowPtr[r]; i < m->rowPtr[r + 1]; ++i) {
        if (m->colIdx[i] == c) return m->val + i;
    }
    return 0;
}

} // namespace sparse
} // namespace matrix
