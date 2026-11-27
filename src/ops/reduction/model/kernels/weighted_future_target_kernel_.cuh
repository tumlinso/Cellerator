__global__ void weighted_future_target_kernel_(
    const float *reference_dense,
    const std::int64_t *neighbor_row_indices,
    const float *neighbor_weights,
    std::int64_t query_rows,
    std::int64_t top_k,
    std::int64_t genes,
    std::int64_t reference_rows,
    float *target) {
    const std::int64_t element = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = query_rows * genes;
    if (element >= total) return;

    const std::int64_t row = element / genes;
    const std::int64_t gene = element - row * genes;
    float accum = 0.0f;
    for (std::int64_t slot = 0; slot < top_k; ++slot) {
        const std::int64_t neighbor_row = neighbor_row_indices[row * top_k + slot];
        const float weight = neighbor_weights[row * top_k + slot];
        if (neighbor_row < 0 || neighbor_row >= reference_rows || weight == 0.0f) continue;
        accum += weight * reference_dense[neighbor_row * genes + gene];
    }
    target[element] = accum;
}
