#pragma once

#include "stream.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cusparse.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::runtime {

struct cusparse_cache {
    int device = -1;
    cusparseHandle_t handle = nullptr;
    bool owns_handle = false;
    const void *blocked_ell_token = nullptr;
    cusparseSpMatDescr_t blocked_ell_f16 = nullptr;
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

void init(cublas_cache *cache);
void clear(cublas_cache *cache);
cublasHandle_t acquire_cublas(cublas_cache *cache, const execution_context &ctx);

} // namespace cellerator::runtime
