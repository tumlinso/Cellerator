#include "blocked_ell_descriptor.cuh"

#include <stdexcept>

namespace cellerator::runtime {

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

std::size_t &cached_blocked_ell_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op) {
    if (cache == nullptr) throw std::invalid_argument("cached_blocked_ell_spmm_bytes requires a cache");
    (void) op;
    return cache->blocked_ell_spmm_bytes_non_transpose;
}

} // namespace cellerator::runtime
