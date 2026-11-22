#pragma once

#include <Cellerator/common/types.cuh>
#include <Cellerator/tensor/base/compressed.hh>
#include <Cellerator/tensor/base/coo.hh>


#include <cstddef>

namespace cellerator::compute::matrix::convert {

namespace real = ::cellerator::real;
namespace types = ::cellerator::types;

// COO coordinates are zero-based, in bounds, and resident on the device of
// stream. Caller storage remains live until stream completion. Conversions
// preserve stored occurrences, including duplicates. The workspace query
// only needs the coordinate arrays and extents.
int compressed_from_coo_library_workspace_bytes(
    const tensor::base::CooView<types::idx_t, const real::storage_t> &source,
    tensor::base::CompressedAxis axis,
    cudaStream_t stream,
    std::size_t *bytes_out);

int build_compressed_from_coo_library(
    const tensor::base::CooView<types::idx_t, const real::storage_t> &source,
    tensor::base::CompressedAxis axis,
    types::ptr_t *d_cAxPtr,
    types::idx_t *d_sort_cAxIdx,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    types::idx_t *d_permutation,
    void *d_sort_tmp,
    std::size_t sort_bytes,
    cudaStream_t stream);

int build_compressed_from_coo_custom(
    const tensor::base::CooView<types::idx_t, const real::storage_t> &source,
    tensor::base::CompressedAxis axis,
    types::ptr_t *d_cAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream);

// Coordinates must already be nondecreasing along the compressed axis. An
// output minor or value pointer equal to its input pointer avoids that copy.
int build_compressed_from_sorted_coo_custom(
    const tensor::base::CooView<types::idx_t, const real::storage_t> &source,
    tensor::base::CompressedAxis axis,
    types::ptr_t *d_cAxPtr,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    cudaStream_t stream);

// Asynchronously write the major offsets into disjoint caller storage with
// capacity for major dimension + 1 int64_t elements. Coordinates must be in
// bounds and nondecreasing along axis. The result aliases the source minor
// coordinates and values with their existing order and representation. Keep
// those buffers live through all consumers. Repeated coordinates stay repeated
// and may require canonicalization for ATen CSR/CSC.
int build_compressed_view_from_sorted_coo(
    const tensor::base::CooView<int64_t, const void> &source,
    tensor::base::CompressedAxis axis,
    int64_t *d_major_ptr,
    uint64_t major_capacity,
    tensor::base::CompressedView<int64_t, const void> *out,
    cudaStream_t stream);

int compressed_transpose_library_workspace_bytes(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    cudaStream_t stream,
    std::size_t *bytes_out);

int build_compressed_transpose_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream);

int build_compressed_transpose_custom_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream);

int build_compressed_transpose_library_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::ptr_t *d_cAxPtr,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_out_uAxPtr,
    types::idx_t *d_out_cAxIdx,
    real::storage_t *d_out_val,
    void *d_tmp,
    std::size_t tmp_bytes,
    cudaStream_t stream);

int transpose_coo_entries(
    const tensor::base::CooView<types::idx_t, const real::storage_t> &source,
    types::idx_t *d_dst_rowIdx,
    types::idx_t *d_dst_colIdx,
    real::storage_t *d_dst_val,
    cudaStream_t stream);

} // namespace cellerator::compute::matrix::convert
