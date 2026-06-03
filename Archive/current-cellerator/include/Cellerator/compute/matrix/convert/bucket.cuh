#pragma once

#include <Cellerator/core/matrix/compressed.cuh>

#include <cstddef>

namespace cellerator::compute::matrix::convert::bucket {

namespace real = ::cellerator::core::real;
namespace types = ::cellerator::core::types;

struct alignas(16) major_nnz_bucket_plan_view {
    types::dim_t major_dim;
    types::idx_t bucket_count;
    const types::idx_t *major_order;
    const types::idx_t *major_nnz_sorted;
    const types::idx_t *bucket_offsets;
    const types::idx_t *inverse_major_order;
};

types::idx_t clamp_bucket_count(types::dim_t major_dim, types::idx_t requested);
int major_nnz_bucket_sort_scratch_bytes(types::dim_t major_dim, std::size_t *out_bytes);
int major_nnz_bucket_scan_scratch_bytes(types::dim_t major_dim, std::size_t *out_bytes);

int build_major_nnz_bucket_plan_raw(
    const types::ptr_t *d_major_ptr,
    types::dim_t major_dim,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream);

int build_major_nnz_bucket_plan_custom_raw(
    const types::ptr_t *d_major_ptr,
    types::dim_t major_dim,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream);

int build_major_nnz_bucket_plan_library_raw(
    const types::ptr_t *d_major_ptr,
    types::dim_t major_dim,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream);

int build_shard_major_nnz_bucket_plan_raw(
    const types::ptr_t * const *d_part_major_ptr,
    const types::idx_t *d_part_row_offsets,
    types::idx_t part_count,
    types::dim_t shard_rows,
    types::idx_t *d_major_nnz,
    types::idx_t *d_major_nnz_sorted,
    types::idx_t *d_major_order_in,
    types::idx_t *d_major_order_out,
    types::idx_t *d_bucket_offsets,
    types::idx_t requested_bucket_count,
    void *d_sort_tmp,
    std::size_t sort_tmp_bytes,
    cudaStream_t stream);

int rebuild_compressed_major_order_raw(
    const types::ptr_t *d_src_major_ptr,
    const types::idx_t *d_src_minor_idx,
    const real::storage_t *d_src_val,
    types::dim_t major_dim,
    const types::idx_t *d_major_order,
    types::ptr_t *d_dst_major_ptr,
    types::idx_t *d_dst_minor_idx,
    real::storage_t *d_dst_val,
    types::idx_t *d_inverse_major_order,
    void *d_scan_tmp,
    std::size_t scan_tmp_bytes,
    cudaStream_t stream);

int rebuild_bucketed_shard_compressed_raw(
    const types::ptr_t * const *d_part_major_ptr,
    const types::idx_t * const *d_part_minor_idx,
    const real::storage_t * const *d_part_val,
    const types::idx_t *d_part_row_offsets,
    types::idx_t part_count,
    types::dim_t shard_rows,
    const types::idx_t *d_major_order,
    types::ptr_t *d_dst_major_ptr,
    types::idx_t *d_dst_minor_idx,
    real::storage_t *d_dst_val,
    types::idx_t *d_inverse_major_order,
    void *d_scan_tmp,
    std::size_t scan_tmp_bytes,
    cudaStream_t stream);

} // namespace cellerator::compute::matrix::convert::bucket
