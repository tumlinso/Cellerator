#pragma once

#include "../matrix.cuh"

namespace matrix {
namespace sparse {

struct ell {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned char format;

    unsigned int *colIdx;
    __half *val;
    unsigned int max_nnz_per_row;
};

inline void init(ell *m, unsigned int rows = 0, unsigned int cols = 0, unsigned int width = 0) {
    m->rows = rows;
    m->cols = cols;
    m->nnz = rows * width;
    m->format = format_ell;
    m->colIdx = 0;
    m->val = 0;
    m->max_nnz_per_row = width;
}

inline std::size_t bytes(const ell *m) {
    return sizeof(*m)
        + (std::size_t) m->nnz * sizeof(unsigned int)
        + (std::size_t) m->nnz * sizeof(__half);
}

inline void clear(ell *m) {
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

inline int allocate(ell *m) {
    std::free(m->colIdx);
    std::free(m->val);
    m->colIdx = 0;
    m->val = 0;
    m->nnz = m->rows * m->max_nnz_per_row;
    if (m->nnz == 0) return 1;
    m->colIdx = (unsigned int *) std::malloc((std::size_t) m->nnz * sizeof(unsigned int));
    m->val = (__half *) std::malloc((std::size_t) m->nnz * sizeof(__half));
    if (m->colIdx == 0 || m->val == 0) {
        std::free(m->colIdx);
        std::free(m->val);
        m->colIdx = 0;
        m->val = 0;
        return 0;
    }
    return 1;
}

inline const __half *at(const ell *m, unsigned int r, unsigned int c) {
    unsigned int base = r * m->max_nnz_per_row;
    for (unsigned int i = 0; i < m->max_nnz_per_row; ++i) {
        if (m->colIdx[base + i] == c) return m->val + base + i;
    }
    return 0;
}

inline __half *at(ell *m, unsigned int r, unsigned int c) {
    unsigned int base = r * m->max_nnz_per_row;
    for (unsigned int i = 0; i < m->max_nnz_per_row; ++i) {
        if (m->colIdx[base + i] == c) return m->val + base + i;
    }
    return 0;
}

} // namespace sparse
} // namespace matrix
