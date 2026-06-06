#pragma once

#include "../ops/ops.hh"

#include <cstdint>

namespace cellerator::compute::sparse::project {

namespace runtime = ::cellerator::compute::runtime;
namespace sparse_ops = ::cellerator::compute::sparse::ops;

void csr_spmm_fwd_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void csr_spmm_fwd_f32_lib(
    const runtime::execution_context &ctx,
    runtime::cusparse_cache *cache,
    const void *matrix_token,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t nnz,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void blocked_ell_spmm_fwd_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void quantized_blocked_ell_spmm_fwd_f32(
    const runtime::execution_context &ctx,
    const sparse_ops::quantized_blocked_ell_view &matrix,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void blocked_ell_spmm_fwd_f16_f32_lib(
    const runtime::execution_context &ctx,
    runtime::cusparse_cache *cache,
    const void *matrix_token,
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void blocked_ell_spmm_fwd_f16_f16_f32_lib(
    const runtime::execution_context &ctx,
    runtime::cusparse_cache *cache,
    const void *matrix_token,
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const __half *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

namespace dist {

void launch_csr_spmm_fwd_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const float *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld);

void launch_blocked_ell_spmm_fwd_f16_f32_lib(
    runtime::fleet_context *fleet,
    runtime::cusparse_cache *cache_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const void *const *matrix_token,
    const std::uint32_t *const *block_col_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::uint32_t *block_size,
    const std::uint32_t *ell_cols,
    const float *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld);

void launch_blocked_ell_spmm_fwd_f16_f16_f32_lib(
    runtime::fleet_context *fleet,
    runtime::cusparse_cache *cache_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const void *const *matrix_token,
    const std::uint32_t *const *block_col_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::uint32_t *block_size,
    const std::uint32_t *ell_cols,
    const __half *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld);

} // namespace dist

} // namespace cellerator::compute::sparse::project
