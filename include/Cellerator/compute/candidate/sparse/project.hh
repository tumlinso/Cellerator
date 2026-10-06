#pragma once

#include <Cellerator/compute/operators/sparse/ops.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::sparse::project {

namespace runtime = ::cellerator::runtime;
namespace sparse_ops = ::cellerator::compute::sparse::ops;

template<class V, class X, class M, class A, class Y, class C>
inline constexpr bool supported_csr_spmm_fwd_v =
    (std::is_same_v<V, float> && std::is_same_v<X, float> &&
     std::is_same_v<M, float> && std::is_same_v<A, float> &&
     std::is_same_v<Y, float> && std::is_same_v<C, float>) ||
    (std::is_same_v<V, float> && std::is_same_v<X, double> &&
     std::is_same_v<M, double> && std::is_same_v<A, double> &&
     std::is_same_v<Y, double> && std::is_same_v<C, double>) ||
    (std::is_same_v<V, double> && std::is_same_v<X, float> &&
     std::is_same_v<M, double> && std::is_same_v<A, double> &&
     std::is_same_v<Y, double> && std::is_same_v<C, double>) ||
    (std::is_same_v<V, double> && std::is_same_v<X, double> &&
     std::is_same_v<M, double> && std::is_same_v<A, double> &&
     std::is_same_v<Y, double> && std::is_same_v<C, double>);

// Shared CSR traversal with explicit arithmetic policy. V and X describe the
// stored relation and feature types; M, A, Y, and C describe multiplication,
// accumulation, output, and affine coefficient types.
template<class V, class X, class M, class A, class Y, class C,
         std::enable_if_t<supported_csr_spmm_fwd_v<V, X, M, A, Y, C>, int> = 0>
void csr_spmm_fwd(
    const runtime::execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const V *values,
    std::uint32_t rows,
    std::uint32_t cols,
    const X *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    Y *out,
    std::int64_t out_ld,
    const std::uint32_t *value_indices = nullptr,
    C input_scale = C{1},
    C destination_scale = C{0});

extern template void csr_spmm_fwd<float, float, float, float, float, float>(
    const runtime::execution_context &, const std::uint32_t *, const std::uint32_t *,
    const float *, std::uint32_t, std::uint32_t, const float *, std::int64_t,
    std::int64_t, float *, std::int64_t, const std::uint32_t *, float, float);
extern template void csr_spmm_fwd<float, double, double, double, double, double>(
    const runtime::execution_context &, const std::uint32_t *, const std::uint32_t *,
    const float *, std::uint32_t, std::uint32_t, const double *, std::int64_t,
    std::int64_t, double *, std::int64_t, const std::uint32_t *, double, double);
extern template void csr_spmm_fwd<double, float, double, double, double, double>(
    const runtime::execution_context &, const std::uint32_t *, const std::uint32_t *,
    const double *, std::uint32_t, std::uint32_t, const float *, std::int64_t,
    std::int64_t, double *, std::int64_t, const std::uint32_t *, double, double);
extern template void csr_spmm_fwd<double, double, double, double, double, double>(
    const runtime::execution_context &, const std::uint32_t *, const std::uint32_t *,
    const double *, std::uint32_t, std::uint32_t, const double *, std::int64_t,
    std::int64_t, double *, std::int64_t, const std::uint32_t *, double, double);

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

void csr_spmm_fwd_f32(const runtime::execution_context&, const std::uint32_t*,
    const std::uint32_t*, const float*, std::uint32_t, std::uint32_t,
    const float*, std::int64_t, std::int64_t, float*, std::int64_t,
    const std::uint32_t* value_indices = nullptr, float input_scale = 1.0f,
    float destination_scale = 0.0f);

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
