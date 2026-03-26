#pragma once

namespace matrix {

template<typename ValueT = Real>
struct dense {
    typedef ValueT value_type;
    typedef Index index_type;

    Index rows;
    Index cols;
    Index nnz;
    unsigned char format;

    ValueT *val;
    Index ld;
};

template<typename ValueT>
inline void init(dense<ValueT> *m, Index rows = 0, Index cols = 0) {
    require_fp_storage<ValueT>();
    m->rows = rows;
    m->cols = cols;
    m->nnz = rows * cols;
    m->format = format_dense;
    m->val = 0;
    m->ld = cols;
}

template<typename ValueT>
inline std::size_t bytes(const dense<ValueT> *m) {
    return sizeof(*m) + (std::size_t) m->nnz * sizeof(ValueT);
}

template<typename ValueT>
inline void clear(dense<ValueT> *m) {
    std::free(m->val);
    m->val = 0;
    m->ld = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->format = format_dense;
}

template<typename ValueT>
inline int allocate(dense<ValueT> *m) {
    std::free(m->val);
    m->val = 0;
    m->nnz = m->rows * m->cols;
    m->ld = m->cols;
    if (m->nnz == 0) return 1;
    m->val = (ValueT *) std::malloc((std::size_t) m->nnz * sizeof(ValueT));
    return m->val != 0;
}

template<typename ValueT>
inline const ValueT *at(const dense<ValueT> *m, Index r, Index c) {
    return m->val + r * m->ld + c;
}

template<typename ValueT>
inline ValueT *at(dense<ValueT> *m, Index r, Index c) {
    return m->val + r * m->ld + c;
}

} // namespace matrix
