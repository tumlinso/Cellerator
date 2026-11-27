#pragma once

#include <Cellerator/runtime/libraries.cuh>

#include <stddef.h>
#include <stdint.h>

namespace cellerator::runtime {

cusparseSpMatDescr_t acquire_blocked_ell_f16_descriptor(
    cusparse_cache *cache,
    const execution_context &ctx,
    const void *matrix_token,
    uint32_t rows,
    uint32_t cols,
    uint32_t block_size,
    uint32_t ell_cols,
    const uint32_t *block_col_idx,
    const __half *values);
size_t &cached_blocked_ell_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op);

} // namespace cellerator::runtime
