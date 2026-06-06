__global__ void stage_bucket_backward_kernel_(
    const float *stage,
    const std::int64_t *day_buckets,
    const float *bucket_mean,
    const float *row_anchor_scale,
    const float *row_rank_scale,
    const float *spread_row_scale,
    std::int64_t row_count,
    float grad_ranking,
    float grad_anchor,
    float grad_spread,
    float *grad_stage) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const std::int64_t bucket = day_buckets[row];
    float grad = 0.0f;
    grad += grad_ranking * row_rank_scale[bucket];
    grad += grad_anchor * row_anchor_scale[bucket];
    const float spread_scale = spread_row_scale[bucket];
    if (spread_scale != 0.0f) {
        grad += grad_spread * spread_scale * (stage[row] - bucket_mean[bucket]);
    }
    grad_stage[row] = grad;
}
