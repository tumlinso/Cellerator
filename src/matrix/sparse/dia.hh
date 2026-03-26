#pragma once

namespace matrix {
namespace sparse {

template<typename Real = ::matrix::Real>
struct dia {
    typedef Real value_type;
    typedef Index index_type;

    Index rows;
    Index cols;
    Index nnz;
    unsigned char format;

    Index *offsets;
    Real *val;
    Index num_diagonals;
};

template<typename Real>
inline void init(dia<Real> *m, Index rows = 0, Index cols = 0, Index nnz = 0) {
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
inline std::size_t bytes(const dia<Real> *m) {
    return sizeof(*m)
        + (std::size_t) m->num_diagonals * sizeof(Index)
        + (std::size_t) m->nnz * sizeof(Real);
}

template<typename Real>
inline void clear(dia<Real> *m) {
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
inline int allocate(dia<Real> *m) {
    std::free(m->offsets);
    std::free(m->val);
    m->offsets = 0;
    m->val = 0;
    if (m->num_diagonals != 0) m->offsets = (Index *) std::malloc((std::size_t) m->num_diagonals * sizeof(Index));
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
inline const Real *at(const dia<Real> *m, Index r, Index c) {
    Index offset = c - r;
    for (Index i = 0; i < m->num_diagonals; ++i) {
        if (m->offsets[i] == offset) return m->val + i * m->rows + r;
    }
    return 0;
}

template<typename Real>
inline Real *at(dia<Real> *m, Index r, Index c) {
    Index offset = c - r;
    for (Index i = 0; i < m->num_diagonals; ++i) {
        if (m->offsets[i] == offset) return m->val + i * m->rows + r;
    }
    return 0;
}

} // namespace sparse
} // namespace matrix
