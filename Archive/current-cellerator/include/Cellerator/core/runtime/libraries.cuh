#pragma once

#include "stream.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cusparse.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::core::runtime {

struct cusparse_cache {
    int device = -1;
    cusparseHandle_t handle = nullptr;
    bool owns_handle = false;
    const void *matrix_token = nullptr;
    const void *blocked_ell_token = nullptr;
    cusparseSpMatDescr_t csr_f32 = nullptr;
    cusparseSpMatDescr_t blocked_ell_f16 = nullptr;
    std::size_t spmv_bytes_non_transpose = 0;
    std::size_t spmv_bytes_transpose = 0;
    std::size_t spmm_bytes_non_transpose = 0;
    std::size_t spmm_bytes_transpose = 0;
    std::size_t blocked_ell_spmm_bytes_non_transpose = 0;
};

struct cublas_cache {
    int device = -1;
    cublasHandle_t handle = nullptr;
    bool owns_handle = false;
};

void init(cusparse_cache *cache);
void clear(cusparse_cache *cache);
cusparseHandle_t acquire_cusparse(cusparse_cache *cache, const execution_context &ctx);
cusparseSpMatDescr_t acquire_csr_f32_descriptor(
    cusparse_cache *cache,
    const execution_context &ctx,
    const void *matrix_token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t nnz,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values);
cusparseSpMatDescr_t acquire_blocked_ell_f16_descriptor(
    cusparse_cache *cache,
    const execution_context &ctx,
    const void *matrix_token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const std::uint32_t *block_col_idx,
    const __half *values);
std::size_t &cached_spmv_bytes(cusparse_cache *cache, cusparseOperation_t op);
std::size_t &cached_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op);
std::size_t &cached_blocked_ell_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op);

void init(cublas_cache *cache);
void clear(cublas_cache *cache);
cublasHandle_t acquire_cublas(cublas_cache *cache, const execution_context &ctx);

} // namespace cellerator::core::runtime
