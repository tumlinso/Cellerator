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

__host__ __device__ __forceinline__ float real_to_float(Real value) {
    return __half2float(value);
}

__host__ __device__ __forceinline__ Real real_from_float(float value) {
    return __float2half(value);
}

template<typename T>
__host__ __device__ __forceinline__ T ldg(const T *ptr) {
    return *ptr;
}

__host__ __device__ __forceinline__ Index find_offset_span(Index row, const Index *offsets, Index count) {
    Index lo = 0;
    Index hi = count;
    while (lo < hi) {
        Index mid = lo + ((hi - lo) >> 1);
        const Index mid_offset = ldg(offsets + mid);
        const Index next_offset = ldg(offsets + mid + 1);
        if (row < mid_offset) hi = mid;
        else if (row >= next_offset) lo = mid + 1;
        else return mid;
    }
    return count;
}

template<typename ValueT>
__host__ __device__ __forceinline__ void require_fp_storage() {
    static_assert(sizeof(ValueT) == 2 || sizeof(ValueT) == 4,
                  "matrix values must be 16-bit or 32-bit floating storage");
}

template<typename ValueT>
__host__ __device__ __forceinline__ std::size_t value_bytes() {
    require_fp_storage<ValueT>();
    return sizeof(ValueT);
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

#include "dense.hh"
#include "sparse/csr.hh"
#include "sparse/coo.hh"
#include "sparse/dia.hh"
#include "sharded.hh"

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
