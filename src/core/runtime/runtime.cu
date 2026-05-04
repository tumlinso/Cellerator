#include <Cellerator/core/runtime/runtime.cuh>

#include <cstring>
#include <stdexcept>

namespace cellerator::core::runtime {

void init(execution_context *ctx, int device, cudaStream_t stream) {
    if (ctx == nullptr) throw std::invalid_argument("init(execution_context) requires a context");
    ctx->device = -1;
    ctx->stream = nullptr;
    ctx->owns_stream = false;

    if (device >= 0) {
        ctx->device = device;
    } else {
        cuda_require(cudaGetDevice(&ctx->device), "cudaGetDevice(core runtime)");
    }
    cuda_require(cudaSetDevice(ctx->device), "cudaSetDevice(core runtime)");
    if (stream != nullptr) {
        ctx->stream = stream;
        ctx->owns_stream = false;
        return;
    }
    cuda_require(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(core runtime)");
    ctx->owns_stream = true;
}

void clear(execution_context *ctx) {
    if (ctx == nullptr) return;
    if (ctx->stream != nullptr && ctx->owns_stream) {
        cudaSetDevice(ctx->device);
        cudaStreamDestroy(ctx->stream);
    }
    ctx->device = -1;
    ctx->stream = nullptr;
    ctx->owns_stream = false;
}

void init(scratch_arena *arena) {
    if (arena == nullptr) throw std::invalid_argument("init(scratch_arena) requires an arena");
    arena->data = nullptr;
    arena->bytes = 0;
}

void clear(scratch_arena *arena) {
    if (arena == nullptr) return;
    if (arena->data != nullptr) cudaFree(arena->data);
    arena->data = nullptr;
    arena->bytes = 0;
}

void *request_scratch(scratch_arena *arena, std::size_t bytes) {
    if (arena == nullptr) throw std::invalid_argument("request_scratch requires an arena");
    if (bytes <= arena->bytes) return arena->data;
    if (arena->data != nullptr) {
        cudaFree(arena->data);
        arena->data = nullptr;
        arena->bytes = 0;
    }
    if (bytes == 0) return nullptr;
    cuda_require(cudaMalloc(&arena->data, bytes), "cudaMalloc(core runtime scratch)");
    arena->bytes = bytes;
    return arena->data;
}

void init(cusparse_cache *cache) {
    if (cache == nullptr) throw std::invalid_argument("init(cusparse_cache) requires a cache");
    std::memset(cache, 0, sizeof(*cache));
    cache->device = -1;
}

void clear(cusparse_cache *cache) {
    if (cache == nullptr) return;
    if (cache->blocked_ell_f16 != nullptr) {
        cusparseDestroySpMat(cache->blocked_ell_f16);
        cache->blocked_ell_f16 = nullptr;
    }
    if (cache->csr_f32 != nullptr) {
        cusparseDestroySpMat(cache->csr_f32);
        cache->csr_f32 = nullptr;
    }
    if (cache->handle != nullptr && cache->owns_handle) {
        cusparseDestroy(cache->handle);
        cache->handle = nullptr;
    }
    cache->device = -1;
    cache->owns_handle = false;
    cache->matrix_token = nullptr;
    cache->blocked_ell_token = nullptr;
    cache->spmv_bytes_non_transpose = 0;
    cache->spmv_bytes_transpose = 0;
    cache->spmm_bytes_non_transpose = 0;
    cache->spmm_bytes_transpose = 0;
    cache->blocked_ell_spmm_bytes_non_transpose = 0;
}

void init(cublas_cache *cache) {
    if (cache == nullptr) throw std::invalid_argument("init(cublas_cache) requires a cache");
    cache->device = -1;
    cache->handle = nullptr;
    cache->owns_handle = false;
}

void clear(cublas_cache *cache) {
    if (cache == nullptr) return;
    if (cache->handle != nullptr && cache->owns_handle) {
        cublasDestroy(cache->handle);
        cache->handle = nullptr;
    }
    cache->device = -1;
    cache->owns_handle = false;
}

cublasHandle_t acquire_cublas(cublas_cache *cache, const execution_context &ctx) {
    if (cache == nullptr) throw std::invalid_argument("acquire_cublas requires a cache");
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(acquire_cublas)");
    if (cache->handle == nullptr || cache->device != ctx.device) {
        clear(cache);
        if (cublasCreate(&cache->handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate(core runtime) failed");
        }
        cache->owns_handle = true;
        cache->device = ctx.device;
        (void)cublasSetMathMode(cache->handle, CUBLAS_TENSOR_OP_MATH);
    }
    if (cublasSetStream(cache->handle, ctx.stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSetStream(core runtime) failed");
    }
    return cache->handle;
}

cusparseHandle_t acquire_cusparse(cusparse_cache *cache, const execution_context &ctx) {
    if (cache == nullptr) throw std::invalid_argument("acquire_cusparse requires a cache");
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(acquire_cusparse)");
    if (cache->handle == nullptr) {
        if (cusparseCreate(&cache->handle) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("cusparseCreate(core runtime) failed");
        }
        cache->owns_handle = true;
        cache->device = ctx.device;
    }
    if (cache->device != ctx.device) {
        clear(cache);
        if (cusparseCreate(&cache->handle) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("cusparseCreate(core runtime device switch) failed");
        }
        cache->owns_handle = true;
        cache->device = ctx.device;
    }
    if (cusparseSetStream(cache->handle, ctx.stream) != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparseSetStream(core runtime) failed");
    }
    return cache->handle;
}

cusparseSpMatDescr_t acquire_csr_f32_descriptor(
    cusparse_cache *cache,
    const execution_context &ctx,
    const void *matrix_token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t nnz,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values) {
    if (cache == nullptr) throw std::invalid_argument("acquire_csr_f32_descriptor requires a cache");
    acquire_cusparse(cache, ctx);
    if (cache->matrix_token == matrix_token && cache->csr_f32 != nullptr) return cache->csr_f32;
    if (cache->csr_f32 != nullptr) {
        cusparseDestroySpMat(cache->csr_f32);
        cache->csr_f32 = nullptr;
    }
    if (cusparseCreateCsr(
            &cache->csr_f32,
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(cols),
            static_cast<std::int64_t>(nnz),
            const_cast<std::uint32_t *>(major_ptr),
            const_cast<std::uint32_t *>(minor_idx),
            const_cast<float *>(values),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparseCreateCsr(core runtime f32 cache) failed");
    }
    cache->matrix_token = matrix_token;
    cache->spmv_bytes_non_transpose = 0;
    cache->spmv_bytes_transpose = 0;
    cache->spmm_bytes_non_transpose = 0;
    cache->spmm_bytes_transpose = 0;
    return cache->csr_f32;
}

cusparseSpMatDescr_t acquire_blocked_ell_f16_descriptor(
    cusparse_cache *cache,
    const execution_context &ctx,
    const void *matrix_token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const std::uint32_t *block_col_idx,
    const __half *values) {
    if (cache == nullptr) throw std::invalid_argument("acquire_blocked_ell_f16_descriptor requires a cache");
    acquire_cusparse(cache, ctx);
    if (cache->blocked_ell_token == matrix_token && cache->blocked_ell_f16 != nullptr) return cache->blocked_ell_f16;
    if (cache->blocked_ell_f16 != nullptr) {
        cusparseDestroySpMat(cache->blocked_ell_f16);
        cache->blocked_ell_f16 = nullptr;
    }
    if (cusparseCreateBlockedEll(
            &cache->blocked_ell_f16,
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(cols),
            static_cast<std::int64_t>(block_size),
            static_cast<std::int64_t>(ell_cols),
            const_cast<std::uint32_t *>(block_col_idx),
            const_cast<__half *>(values),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_16F) != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("cusparseCreateBlockedEll(core runtime cache) failed");
    }
    cache->blocked_ell_token = matrix_token;
    cache->blocked_ell_spmm_bytes_non_transpose = 0;
    return cache->blocked_ell_f16;
}

std::size_t &cached_spmv_bytes(cusparse_cache *cache, cusparseOperation_t op) {
    if (cache == nullptr) throw std::invalid_argument("cached_spmv_bytes requires a cache");
    return op == CUSPARSE_OPERATION_NON_TRANSPOSE
        ? cache->spmv_bytes_non_transpose
        : cache->spmv_bytes_transpose;
}

std::size_t &cached_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op) {
    if (cache == nullptr) throw std::invalid_argument("cached_spmm_bytes requires a cache");
    return op == CUSPARSE_OPERATION_NON_TRANSPOSE
        ? cache->spmm_bytes_non_transpose
        : cache->spmm_bytes_transpose;
}

std::size_t &cached_blocked_ell_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op) {
    if (cache == nullptr) throw std::invalid_argument("cached_blocked_ell_spmm_bytes requires a cache");
    (void) op;
    return cache->blocked_ell_spmm_bytes_non_transpose;
}

} // namespace cellerator::core::runtime
