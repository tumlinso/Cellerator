__global__ void dense_reduce_pair_backward_kernel_(
    const std::int64_t *pair_rows,
    const std::int64_t *pair_cols,
    const float *latent,
    const float *time,
    std::int64_t pair_count,
    std::int64_t latent_dim,
    float local_time_window,
    float far_time_window,
    float margin,
    float grad_local_scale,
    float grad_far_scale,
    float *grad_latent) {
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
    const bool unclamped = raw_sqdist > 0.0f;
    const float pair_sqdist = unclamped ? raw_sqdist : 0.0f;
    const float pair_dist = sqrtf(pair_sqdist + kPairEps);

    float scale = 0.0f;
    if (delta_t <= local_time_window && unclamped) {
        scale += -2.0f * grad_local_scale;
    }
    if (delta_t >= far_time_window && unclamped && margin > pair_dist) {
        scale += grad_far_scale / pair_dist;
    }
    if (scale == 0.0f) return;

    float *grad_i = grad_latent + row_i * latent_dim;
    float *grad_j = grad_latent + row_j * latent_dim;
    for (std::int64_t d = 0; d < latent_dim; ++d) {
        atomicAdd(grad_i + d, scale * rhs[d]);
        atomicAdd(grad_j + d, scale * lhs[d]);
    }
}
