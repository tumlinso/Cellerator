#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

namespace cellerator::compute::operation::model_ops {

// Pair/row counts and bucket labels are bounded by signed 32-bit counters.
// Bucket scratch is dense in max(bucket label)+1; this is a representational
// bound, not an allocation policy. Oversized memory requests may fail normally.
inline constexpr std::int64_t maximum_indexed_count =
    static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());

// All buffers are device pointers associated with `stream`'s device. Callers
// provide full, correctly sized buffers and in-range indices. Inputs, scratch,
// and outputs must not alias; all pointers must remain alive through async
// completion. Functions initialize scratch/output buffers on `stream` and
// return the first CUDA launch/memory error.
cudaError_t dense_reduce_pair_forward(
    const std::int64_t* pair_rows,
    const std::int64_t* pair_cols,
    const float* latent_unit,
    const float* developmental_time,
    std::int64_t pair_count,
    std::int64_t row_count,
    std::int64_t latent_dim,
    float local_time_window,
    float far_time_window,
    float margin,
    float* local_sum_scratch,
    std::int32_t* local_count_scratch,
    float* far_sum_scratch,
    std::int32_t* far_count_scratch,
    float* local_loss,
    float* far_loss,
    cudaStream_t stream);

cudaError_t dense_reduce_pair_backward(
    const std::int64_t* pair_rows,
    const std::int64_t* pair_cols,
    const float* latent_unit,
    const float* developmental_time,
    const std::int32_t* local_count,
    const std::int32_t* far_count,
    std::int64_t pair_count,
    std::int64_t row_count,
    std::int64_t latent_dim,
    float local_time_window,
    float far_time_window,
    float margin,
    float grad_local,
    float grad_far,
    float* grad_latent,
    cudaStream_t stream);

cudaError_t developmental_stage_bucket_forward(
    const float* stage,
    const std::int64_t* day_buckets,
    std::int64_t row_count,
    std::int64_t bucket_count,
    float ranking_margin,
    float min_within_day_std,
    bool use_neighbor_day_pairs_only,
    std::int64_t num_day_buckets,
    float* bucket_sum_scratch,
    float* bucket_sumsq_scratch,
    std::int32_t* bucket_rows_scratch,
    float* bucket_mean,
    float* row_anchor_scale,
    float* row_rank_scale,
    float* spread_row_scale,
    float* ranking_loss,
    float* anchor_loss,
    float* spread_loss,
    cudaStream_t stream);

cudaError_t developmental_stage_bucket_backward(
    const float* stage,
    const std::int64_t* day_buckets,
    const float* bucket_mean,
    const float* row_anchor_scale,
    const float* row_rank_scale,
    const float* spread_row_scale,
    std::int64_t row_count,
    std::int64_t bucket_count,
    float grad_ranking,
    float grad_anchor,
    float grad_spread,
    float* grad_stage,
    cudaStream_t stream);

cudaError_t weighted_future_target(
    const float* reference_dense,
    const std::int64_t* neighbor_row_indices,
    const float* neighbor_weights,
    std::int64_t reference_rows,
    std::int64_t query_rows,
    std::int64_t top_k,
    std::int64_t genes,
    float* target,
    cudaStream_t stream);

} // namespace cellerator::compute::operation::model_ops
