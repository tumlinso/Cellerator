#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "../real.cuh"

namespace matrix {

typedef std::uint32_t Index;
typedef std::int32_t DiagIndex;

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

__host__ __device__ __forceinline__ Index find_offset_span(Index row, const Index *offsets, Index count) {
    Index lo = 0;
    Index hi = count;
    while (lo < hi) {
        Index mid = lo + ((hi - lo) >> 1);
        const Index mid_offset = offsets[mid];
        const Index next_offset = offsets[mid + 1];
        if (row < mid_offset) hi = mid;
        else if (row >= next_offset) lo = mid + 1;
        else return mid;
    }
    return count;
}

namespace sparse {

template<typename IndexT, typename RowPtrT, typename ColIdxT>
struct csr_base {
    typedef IndexT index_type;

    IndexT rows;
    IndexT cols;
    IndexT nnz;
    unsigned char format;
    RowPtrT rowPtr;
    ColIdxT colIdx;
};

} // namespace sparse

} // namespace matrix

#include "dense.cuh"
#include "sparse/csr.cuh"
#include "sparse/coo.cuh"
#include "sparse/dia.cuh"
#include "sharded.cuh"

namespace matrix {

template<typename MatrixT>
__host__ __device__ __forceinline__ Index part_aux(const MatrixT *) {
    return 0;
}

template<typename ValueT>
__host__ __device__ __forceinline__ Index part_aux(const sparse::dia<ValueT> *m) {
    return m->num_diagonals;
}

template<typename ValueT>
__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<dense<ValueT> > *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(dense<ValueT>) + (std::size_t) m->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT>
__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::csr<ValueT> > *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::csr<ValueT>)
        + (std::size_t) (m->part_rows[partId] + 1) * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT>
__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::coo<ValueT> > *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::coo<ValueT>)
        + (std::size_t) m->part_nnz[partId] * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(Index)
        + (std::size_t) m->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT>
__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::dia<ValueT> > *m, Index partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::dia<ValueT>)
        + (std::size_t) m->part_aux[partId] * sizeof(DiagIndex)
        + (std::size_t) m->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT>
__host__ __forceinline__ void destroy(dense<ValueT> *m) {
    if (m == 0) return;
    clear(m);
    delete m;
}

template<typename ValueT>
__host__ __forceinline__ void destroy(sparse::csr<ValueT> *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

template<typename ValueT>
__host__ __forceinline__ void destroy(sparse::coo<ValueT> *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

template<typename ValueT>
__host__ __forceinline__ void destroy(sparse::dia<ValueT> *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

} // namespace matrix
