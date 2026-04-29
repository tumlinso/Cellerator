__global__ void dense_reduce_pair_forward_kernel_(
    const std::int64_t *pair_rows,
    const std::int64_t *pair_cols,
    const float *latent,
    const float *time,
    std::int64_t pair_count,
    std::int64_t latent_dim,
    float local_time_window,
    float far_time_window,
    float margin,
    float *local_sum,
    int *local_count,
    float *far_sum,
    int *far_count) {
    const std::int64_t pair_idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= pair_count) return;

    const std::int64_t row_i = pair_rows[pair_idx];
    const std::int64_t row_j = pair_cols[pair_idx];
    const float delta_t = fabsf(time[row_i] - time[row_j]);
    const float *lhs = latent + row_i * latent_dim;
    const float *rhs = latent + row_j * latent_dim;

    float dot = 0.0f;
    for (std::int64_t d = 0; d < latent_dim; ++d) dot += lhs[d] * rhs[d];

    const float raw_sqdist = 2.0f - 2.0f * dot;
    const float pair_sqdist = raw_sqdist > 0.0f ? raw_sqdist : 0.0f;

    if (delta_t <= local_time_window) {
        atomicAdd(local_sum, pair_sqdist);
        atomicAdd(local_count, 1);
    }

    if (delta_t >= far_time_window) {
        const float pair_dist = sqrtf(pair_sqdist + kPairEps);
        const float far_value = margin > pair_dist ? margin - pair_dist : 0.0f;
        atomicAdd(far_sum, far_value);
        atomicAdd(far_count, 1);
    }
}
