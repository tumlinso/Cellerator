#include <Cellerator/compute/operation/model_ops/model_ops.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::operation::model_ops {
namespace {

constexpr int kThreads = 256;
constexpr std::int64_t kMaxGridBlocks = 2147483647;
constexpr float kPairEps = 1.0e-12f;
constexpr float kStdEps = 1.0e-12f;

#include "kernels/dense_reduce_pair_forward_kernel_.cuh"
#include "kernels/dense_reduce_pair_backward_kernel_.cuh"
#include "kernels/stage_bucket_accumulate_kernel_.cuh"
#include "kernels/stage_bucket_finalize_kernel_.cuh"
#include "kernels/stage_bucket_backward_kernel_.cuh"
#include "kernels/weighted_future_target_kernel_.cuh"

__global__ void finalize_pair_losses(
    const float* local_sum,
    const std::int32_t* local_count,
    const float* far_sum,
    const std::int32_t* far_count,
    float* local_loss,
    float* far_loss) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const auto lc = *local_count;
    const auto fc = *far_count;
    *local_loss = lc > 0 ? *local_sum / static_cast<float>(lc) : 0.0f;
    *far_loss = fc > 0 ? *far_sum / static_cast<float>(fc) : 0.0f;
}

bool valid_dimensions(std::int64_t count, std::int64_t rows, std::int64_t width) {
    if (count < 0 || count > maximum_indexed_count
        || rows < 0 || rows > maximum_indexed_count || width < 0)
        return false;
    if (static_cast<std::uint64_t>(count)
            > std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t)
        || static_cast<std::uint64_t>(rows)
            > std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t))
        return false;
    if (rows == 0 || width == 0) return true;
    const auto row_count = static_cast<std::uint64_t>(rows);
    const auto dimension = static_cast<std::uint64_t>(width);
    const auto max_elements = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float)));
    return dimension <= max_elements / row_count;
}

bool valid_row_count(std::int64_t rows) {
    return rows >= 0 && rows <= maximum_indexed_count
        && static_cast<std::uint64_t>(rows)
            <= std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t);
}

cudaError_t clear(void* pointer, std::size_t bytes, cudaStream_t stream) {
    return cudaMemsetAsync(pointer, 0, bytes, stream);
}

} // namespace

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
    cudaStream_t stream) {
    if (!valid_dimensions(pair_count, row_count, latent_dim)) return cudaErrorInvalidValue;
    if (!local_sum_scratch || !local_count_scratch || !far_sum_scratch || !far_count_scratch
        || !local_loss || !far_loss
        || (pair_count > 0 && (!pair_rows || !pair_cols || !latent_unit || !developmental_time)))
        return cudaErrorInvalidValue;
    cudaError_t status = clear(local_sum_scratch, sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(local_count_scratch, sizeof(std::int32_t), stream);
    if (status != cudaSuccess) return status;
    status = clear(far_sum_scratch, sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(far_count_scratch, sizeof(std::int32_t), stream);
    if (status != cudaSuccess) return status;
    status = clear(local_loss, sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(far_loss, sizeof(float), stream);
    if (status != cudaSuccess) return status;

    if (pair_count > 0) {
        const int blocks = static_cast<int>((pair_count + kThreads - 1) / kThreads);
        dense_reduce_pair_forward_kernel_<<<blocks, kThreads, 0, stream>>>(
            pair_rows, pair_cols, latent_unit, developmental_time, pair_count, row_count,
            latent_dim, local_time_window, far_time_window, margin,
            local_sum_scratch, reinterpret_cast<int*>(local_count_scratch),
            far_sum_scratch, reinterpret_cast<int*>(far_count_scratch));
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    finalize_pair_losses<<<1, 1, 0, stream>>>(
        local_sum_scratch, local_count_scratch, far_sum_scratch, far_count_scratch,
        local_loss, far_loss);
    return cudaGetLastError();
}

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
    cudaStream_t stream) {
    if (!valid_dimensions(pair_count, row_count, latent_dim)) return cudaErrorInvalidValue;
    const auto elements = static_cast<std::size_t>(row_count) * static_cast<std::size_t>(latent_dim);
    if ((elements > 0 && !grad_latent) || (pair_count > 0 && (!pair_rows || !pair_cols || !latent_unit
            || !developmental_time || !local_count || !far_count)))
        return cudaErrorInvalidValue;
    cudaError_t status = elements > 0 ? clear(grad_latent, elements * sizeof(float), stream) : cudaSuccess;
    if (status != cudaSuccess || pair_count == 0 || (grad_local == 0.0f && grad_far == 0.0f)) return status;
    const int blocks = static_cast<int>((pair_count + kThreads - 1) / kThreads);
    dense_reduce_pair_backward_kernel_<<<blocks, kThreads, 0, stream>>>(
        pair_rows, pair_cols, latent_unit, developmental_time,
        reinterpret_cast<const int*>(local_count), reinterpret_cast<const int*>(far_count),
        pair_count, row_count, latent_dim, local_time_window, far_time_window, margin,
        grad_local, grad_far, grad_latent);
    return cudaGetLastError();
}

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
    cudaStream_t stream) {
    if (!valid_row_count(row_count) || bucket_count < 0 || num_day_buckets < 0
        || static_cast<std::uint64_t>(bucket_count)
            > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return cudaErrorInvalidValue;
    if ((row_count > 0 && (!stage || !day_buckets))
        || (bucket_count > 0 && (!bucket_sum_scratch || !bucket_sumsq_scratch
            || !bucket_rows_scratch || !bucket_mean || !row_anchor_scale
            || !row_rank_scale || !spread_row_scale))
        || !ranking_loss || !anchor_loss || !spread_loss)
        return cudaErrorInvalidValue;

    const auto bucket_bytes = static_cast<std::size_t>(bucket_count);
    cudaError_t status = clear(ranking_loss, sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(anchor_loss, sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(spread_loss, sizeof(float), stream);
    if (status != cudaSuccess) return status;
    if (bucket_count == 0) return cudaSuccess;
    status = clear(bucket_sum_scratch, bucket_bytes * sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(bucket_sumsq_scratch, bucket_bytes * sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(bucket_rows_scratch, bucket_bytes * sizeof(std::int32_t), stream);
    if (status != cudaSuccess) return status;
    status = clear(bucket_mean, bucket_bytes * sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(row_anchor_scale, bucket_bytes * sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(row_rank_scale, bucket_bytes * sizeof(float), stream);
    if (status != cudaSuccess) return status;
    status = clear(spread_row_scale, bucket_bytes * sizeof(float), stream);
    if (status != cudaSuccess) return status;

    if (row_count == 0) return cudaSuccess;
    const int blocks = static_cast<int>((row_count + kThreads - 1) / kThreads);
    stage_bucket_accumulate_kernel_<<<blocks, kThreads, 0, stream>>>(
        stage, day_buckets, row_count, bucket_count,
        bucket_sum_scratch, bucket_sumsq_scratch,
        reinterpret_cast<int*>(bucket_rows_scratch));
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    stage_bucket_finalize_kernel_<<<1, 1, 0, stream>>>(
        bucket_sum_scratch, bucket_sumsq_scratch,
        reinterpret_cast<const int*>(bucket_rows_scratch), bucket_count,
        ranking_margin, min_within_day_std, use_neighbor_day_pairs_only ? 1 : 0,
        num_day_buckets, bucket_mean, row_anchor_scale, row_rank_scale,
        spread_row_scale, ranking_loss, anchor_loss, spread_loss);
    return cudaGetLastError();
}

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
    cudaStream_t stream) {
    if (!valid_row_count(row_count) || bucket_count < 0
        || static_cast<std::uint64_t>(bucket_count)
            > std::numeric_limits<std::size_t>::max() / sizeof(float)
        || (row_count > 0 && !grad_stage)
        || (row_count > 0 && (!stage || !day_buckets || !bucket_mean
            || !row_anchor_scale || !row_rank_scale || !spread_row_scale)))
        return cudaErrorInvalidValue;
    cudaError_t status = row_count > 0
        ? clear(grad_stage, static_cast<std::size_t>(row_count) * sizeof(float), stream)
        : cudaSuccess;
    if (status != cudaSuccess || row_count == 0
        || (grad_ranking == 0.0f && grad_anchor == 0.0f && grad_spread == 0.0f)) return status;
    const int blocks = static_cast<int>((row_count + kThreads - 1) / kThreads);
    stage_bucket_backward_kernel_<<<blocks, kThreads, 0, stream>>>(
        stage, day_buckets, bucket_mean, row_anchor_scale, row_rank_scale,
        spread_row_scale, row_count, bucket_count, grad_ranking, grad_anchor,
        grad_spread, grad_stage);
    return cudaGetLastError();
}

cudaError_t weighted_future_target(
    const float* reference_dense,
    const std::int64_t* neighbor_row_indices,
    const float* neighbor_weights,
    std::int64_t reference_rows,
    std::int64_t query_rows,
    std::int64_t top_k,
    std::int64_t genes,
    float* target,
    cudaStream_t stream) {
    if (reference_rows < 0 || query_rows < 0 || top_k < 0 || genes < 0)
        return cudaErrorInvalidValue;
    const auto max_signed = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    const auto max_size = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (genes > 0 && static_cast<std::uint64_t>(query_rows) > max_signed / static_cast<std::uint64_t>(genes))
        return cudaErrorInvalidValue;
    if (genes > 0 && static_cast<std::uint64_t>(query_rows) > (max_size / sizeof(float)) / static_cast<std::uint64_t>(genes))
        return cudaErrorInvalidValue;
    if (top_k > 0 && static_cast<std::uint64_t>(query_rows) > max_signed / static_cast<std::uint64_t>(top_k))
        return cudaErrorInvalidValue;
    const auto neighbor_count = static_cast<std::uint64_t>(query_rows) * static_cast<std::uint64_t>(top_k);
    if (neighbor_count > (max_size / sizeof(std::int64_t))
        || neighbor_count > (max_size / sizeof(float)))
        return cudaErrorInvalidValue;
    if (genes > 0 && static_cast<std::uint64_t>(reference_rows) > max_signed / static_cast<std::uint64_t>(genes))
        return cudaErrorInvalidValue;
    if (genes > 0 && static_cast<std::uint64_t>(reference_rows)
            > (max_size / sizeof(float)) / static_cast<std::uint64_t>(genes))
        return cudaErrorInvalidValue;
    const auto total = static_cast<std::uint64_t>(query_rows) * static_cast<std::uint64_t>(genes);
    if (total > static_cast<std::size_t>(kMaxGridBlocks) * kThreads) return cudaErrorInvalidValue;
    if (total == 0) return cudaSuccess;
    if (!target || (neighbor_count > 0 && (!neighbor_row_indices || !neighbor_weights))
        || (reference_rows > 0 && genes > 0 && !reference_dense))
        return cudaErrorInvalidValue;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    weighted_future_target_kernel_<<<blocks, kThreads, 0, stream>>>(
        reference_dense, neighbor_row_indices, neighbor_weights,
        query_rows, top_k, genes, reference_rows, target);
    return cudaGetLastError();
}

} // namespace cellerator::compute::operation::model_ops
