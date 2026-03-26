#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <cuda_fp16.h>

namespace matrix {

typedef __half Real;
typedef std::uint32_t Index;

enum {
    format_none  = 0,
    format_dense = 1,
    format_csr   = 2,
    format_csc   = 3,
    format_coo   = 4,
    format_dia   = 5,
    format_ell   = 6
};

inline int checkformat(unsigned char expected, unsigned char actual, const char *name) {
    if (expected == actual) return 1;
    std::fprintf(stderr,
                 "Error: expected format %u, got %u for %s\n",
                 (unsigned int) expected,
                 (unsigned int) actual,
                 name);
    return 0;
}

struct header {
    unsigned char format;
    Index rows;
    Index cols;
    Index nnz;
};

inline float real_to_float(Real value) {
    return __half2float(value);
}

inline Real real_from_float(float value) {
    return __float2half(value);
}

inline Index find_offset_span(Index row, const Index *offsets, Index count) {
    Index lo = 0;
    Index hi = count;
    while (lo < hi) {
        Index mid = lo + ((hi - lo) >> 1);
        if (row < offsets[mid]) hi = mid;
        else if (row >= offsets[mid + 1]) lo = mid + 1;
        else return mid;
    }
    return count;
}

template<typename ValueT>
inline void require_fp_storage() {
    static_assert(sizeof(ValueT) == 2 || sizeof(ValueT) == 4,
                  "matrix values must be 16-bit or 32-bit floating storage");
}

template<typename ValueT>
inline std::size_t value_bytes() {
    require_fp_storage<ValueT>();
    return sizeof(ValueT);
}

} // namespace matrix

#include "dense.hh"
#include "sparse/csr.hh"
#include "sparse/coo.hh"
#include "sparse/dia.hh"
#include "sharded.hh"

namespace matrix {

template<typename MatrixT>
inline Index part_aux(const MatrixT *) {
    return 0;
}

template<typename ValueT>
inline Index part_aux(const sparse::dia<ValueT> *m) {
    return m->num_diagonals;
}

template<typename ValueT>
inline std::size_t part_bytes(const sharded<dense<ValueT> > *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(dense<ValueT>) + (std::size_t) m->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT>
inline std::size_t part_bytes(const sharded<sparse::csr<ValueT> > *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::csr<ValueT>)
        + (std::size_t) (m->part_rows[partId] + 1) * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT>
inline std::size_t part_bytes(const sharded<sparse::coo<ValueT> > *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::coo<ValueT>)
        + (std::size_t) m->part_nnz[partId] * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT>
inline std::size_t part_bytes(const sharded<sparse::dia<ValueT> > *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::dia<ValueT>)
        + (std::size_t) m->part_aux[partId] * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT>
inline void destroy(dense<ValueT> *m) {
    if (m == 0) return;
    clear(m);
    delete m;
}

template<typename ValueT>
inline void destroy(sparse::csr<ValueT> *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

template<typename ValueT>
inline void destroy(sparse::coo<ValueT> *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

template<typename ValueT>
inline void destroy(sparse::dia<ValueT> *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

} // namespace matrix
