#pragma once

#include "../../runtime/runtime.hh"
#include <Cellerator/core/quantized/blocked_ell.cuh>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace cellerator::compute::sparse::ops {

namespace runtime = ::cellerator::compute::runtime;

struct feature_affine_quantize_config {
    std::uint32_t bits = 1u;
    float scale_floor = 1.0e-4f;
    float reconstruction_weight = 1.0f;
    float range_weight = 0.01f;
    float min_dynamic_range = 0.25f;
};

struct quantized_blocked_ell_view {
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint32_t nnz = 0u;
    std::uint32_t block_size = 0u;
    std::uint32_t ell_cols = 0u;
    std::uint32_t row_stride_bytes = 0u;
    std::uint32_t bits = 0u;
    std::uint32_t decode_policy = ::cellerator::core::quantized::blocked_ell::decode_policy_unknown;
    const std::uint32_t *block_col_idx = nullptr;
    const std::uint8_t *packed_values = nullptr;
    const float *column_scales = nullptr;
    const float *column_offsets = nullptr;
    const float *row_offsets = nullptr;
};

namespace base {

void csr_row_scale_fwd_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *row_scales,
    std::uint32_t rows,
    float *out_values);

void csr_row_scale_bwd_values_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const float *grad_out,
    const float *row_scales,
    std::uint32_t rows,
    float *grad_values);

void csr_row_scale_bwd_scales_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    float *grad_scales);

void sparse_value_sum_f16_f32(
    const runtime::execution_context &ctx,
    runtime::scratch_arena *arena,
    const __half *values,
    std::uint32_t nnz,
    float *out_scalar);

void sparse_value_sum_bwd_fill_f32(
    const runtime::execution_context &ctx,
    const float *grad_scalar,
    std::uint32_t nnz,
    float *grad_values);

void csr_spmv_fwd_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    const float *vector,
    float *out);

void csr_spmv_bwd_values_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *vector,
    std::uint32_t rows,
    float *grad_values);

void csr_spmv_bwd_vector_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    std::uint32_t cols,
    float *grad_vector);

void csr_spmm_bwd_values_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *rhs,
    std::uint32_t rows,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *grad_values);

void csr_spmm_bwd_rhs_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    std::uint32_t cols,
    std::int64_t grad_out_ld,
    std::int64_t out_cols,
    float *grad_rhs,
    std::int64_t grad_rhs_ld);

void csr_spmv_fwd_f32_lib(
    const runtime::execution_context &ctx,
    runtime::cusparse_cache *cache,
    const void *matrix_token,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t nnz,
    const float *vector,
    float *out);

void csr_feature_affine_quantize_fwd_bwd_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    const float *log_scale,
    const float *offset,
    const feature_affine_quantize_config &config,
    float *reconstruction_loss,
    float *range_loss,
    float *grad_log_scale,
    float *grad_offset);

void blocked_ell_feature_affine_quantize_fwd_bwd_f16_f32(
    const runtime::execution_context &ctx,
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const float *log_scale,
    const float *offset,
    const feature_affine_quantize_config &config,
    float *reconstruction_loss,
    float *range_loss,
    float *grad_log_scale,
    float *grad_offset);

} // namespace base

namespace dist {

void reduce_sum_to_leader_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const float *const *partials,
    std::size_t count,
    float *leader_out);

void launch_csr_row_scale_fwd_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const __half *const *values,
    const float *const *row_scales,
    const std::uint32_t *rows,
    float *const *out_values);

void launch_csr_row_scale_bwd_values_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const float *const *grad_out,
    const float *const *row_scales,
    const std::uint32_t *rows,
    float *const *grad_values);

void launch_csr_row_scale_bwd_scales_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    float *const *grad_scales);

void launch_sparse_value_sum_f16_f32(
    runtime::fleet_context *fleet,
    runtime::scratch_arena *scratch_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const __half *const *values,
    const std::uint32_t *nnz,
    float *const *out_scalar);

void launch_csr_spmv_fwd_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const float *const *vector,
    float *const *out);

void launch_csr_spmv_bwd_values_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const float *const *grad_out,
    const float *const *vector,
    const std::uint32_t *rows,
    float *const *grad_values);

void launch_csr_spmv_bwd_vector_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    float *const *grad_vector);

void launch_csr_spmm_bwd_values_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const float *const *grad_out,
    const float *const *rhs,
    const std::uint32_t *rows,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *grad_values);

void launch_csr_spmm_bwd_rhs_f16_f32(
    runtime::fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::int64_t *grad_out_ld,
    const std::int64_t *out_cols,
    float *const *grad_rhs,
    const std::int64_t *grad_rhs_ld);

} // namespace dist

} // namespace cellerator::compute::sparse::ops
