__global__ void init_best_kernel_(
    std::int64_t *best_cell,
    std::int64_t *best_shard,
    float *best_time,
    std::int64_t *best_embryo,
    float *best_similarity,
    std::int64_t query_rows,
    int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = query_rows * static_cast<std::int64_t>(k);
    if (index >= total) return;
    best_cell[index] = -1;
    best_shard[index] = -1;
    best_time[index] = INFINITY;
    best_embryo[index] = -1;
    best_similarity[index] = -INFINITY;
}
