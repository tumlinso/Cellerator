#pragma once

namespace matrix {
namespace sparse {

template<typename Real = ::matrix::Real>
struct ell {
    typedef Real value_type;
    typedef Index index_type;

    Index rows;
    Index cols;
    Index nnz;
    unsigned char format;

    Index *colIdx;
    Real *val;
    Index max_nnz_per_row;
};

template<typename Real>
inline void init(ell<Real> *m, Index rows = 0, Index cols = 0, Index width = 0) {
    require_fp_storage<Real>();
    m->rows = rows;
    m->cols = cols;
    m->nnz = rows * width;
    m->format = format_ell;
    m->colIdx = 0;
    m->val = 0;
    m->max_nnz_per_row = width;
}

template<typename Real>
inline std::size_t bytes(const ell<Real> *m) {
    return sizeof(*m)
        + (std::size_t) m->nnz * sizeof(Index)
        + (std::size_t) m->nnz * sizeof(Real);
}

template<typename Real>
inline void clear(ell<Real> *m) {
    std::free(m->colIdx);
    std::free(m->val);
    m->colIdx = 0;
    m->val = 0;
    m->max_nnz_per_row = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_ell;
}

template<typename Real>
inline int allocate(ell<Real> *m) {
    std::free(m->colIdx);
    std::free(m->val);
    m->colIdx = 0;
    m->val = 0;
    m->nnz = m->rows * m->max_nnz_per_row;
    if (m->nnz == 0) return 1;
    m->colIdx = (Index *) std::malloc((std::size_t) m->nnz * sizeof(Index));
    m->val = (Real *) std::malloc((std::size_t) m->nnz * sizeof(Real));
    if (m->colIdx == 0 || m->val == 0) {
        std::free(m->colIdx);
        std::free(m->val);
        m->colIdx = 0;
        m->val = 0;
        return 0;
    }
    return 1;
}

template<typename Real>
inline const Real *at(const ell<Real> *m, Index r, Index c) {
    Index base = r * m->max_nnz_per_row;
    for (Index i = 0; i < m->max_nnz_per_row; ++i) {
        if (m->colIdx[base + i] == c) return m->val + base + i;
    }
    return 0;
}

template<typename Real>
inline Real *at(ell<Real> *m, Index r, Index c) {
    Index base = r * m->max_nnz_per_row;
    for (Index i = 0; i < m->max_nnz_per_row; ++i) {
        if (m->colIdx[base + i] == c) return m->val + base + i;
    }
    return 0;
}

} // namespace sparse
} // namespace matrix
