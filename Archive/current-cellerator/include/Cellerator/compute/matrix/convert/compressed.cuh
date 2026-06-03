#pragma once

#include <Cellerator/core/matrix/compressed.cuh>
#include <Cellerator/core/matrix/coo.cuh>

#include <cstddef>

namespace cellerator::compute::matrix::convert {

namespace real = ::cellerator::core::real;
namespace types = ::cellerator::core::types;

int compressed_from_coo_library_workspace_bytes(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    cudaStream_t stream,
    std::size_t *bytes_out);

int build_compressed_from_coo_library_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::idx_t *d_sort_cAxIdx,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    types::idx_t *d_permutation,
    void *d_sort_tmp,
    std::size_t sort_bytes,
    cudaStream_t stream);

int build_compressed_from_coo_custom_raw(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream);

int build_compressed_from_sorted_coo_custom_raw(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    cudaStream_t stream);

int compressed_from_coo_sorted_workspace_bytes(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    cudaStream_t stream,
    std::size_t *bytes_out);

int build_compressed_from_coo_sorted_raw(
    types::dim_t cDim,
    types::dim_t uDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::idx_t *d_sort_cAxIdx,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    types::idx_t *d_permutation,
    void *d_sort_tmp,
    std::size_t sort_bytes,
    cudaStream_t stream);

int build_compressed_from_coo_raw(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
    cudaStream_t stream);

int build_cs_from_coo_raw(
    types::dim_t cDim,
    types::nnz_t nnz,
    const types::idx_t *d_cAxIdx,
    const types::idx_t *d_uAxIdx,
    const real::storage_t *d_val,
    types::ptr_t *d_cAxPtr,
    types::ptr_t *d_heads,
    types::idx_t *d_out_uAx,
    real::storage_t *d_out_val,
    void *d_scan_tmp,
    std::size_t scan_bytes,
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

int build_transpose_cs_from_cs_raw(
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

int transpose_coo_entries_raw(
    types::nnz_t nnz,
    const types::idx_t *d_src_rowIdx,
    const types::idx_t *d_src_colIdx,
    const real::storage_t *d_src_val,
    types::idx_t *d_dst_rowIdx,
    types::idx_t *d_dst_colIdx,
    real::storage_t *d_dst_val,
    cudaStream_t stream);

} // namespace cellerator::compute::matrix::convert
