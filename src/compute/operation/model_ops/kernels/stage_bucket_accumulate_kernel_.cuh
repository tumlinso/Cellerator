__global__ void stage_bucket_accumulate_kernel_(
    const float *stage,
    const std::int64_t *day_buckets,
    std::int64_t row_count,
    std::int64_t bucket_count,
    float *bucket_sum,
    float *bucket_sumsq,
    int *bucket_rows) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const std::int64_t bucket = day_buckets[row];
    if (bucket < 0 || bucket >= bucket_count) return;
    const float value = stage[row];
    atomicAdd(bucket_sum + bucket, value);
    atomicAdd(bucket_sumsq + bucket, value * value);
    atomicAdd(bucket_rows + bucket, 1);
}
