__global__ void ann_probe_kernel_(
    const __half *query_latent,
    const std::int64_t *query_embryo,
    const float *centroids,
    const std::int64_t *list_embryo,
    std::int64_t query_rows,
    int latent_dim,
    std::int64_t list_count,
    int hard_same_embryo,
    int probe_count,
    std::int32_t *selected_list_offsets) {
    const std::int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= query_rows || lane >= kWarpThreads) return;

    ProbeCandidate local_best[kMaxProbe];
    init_probe_candidates_device_(local_best, probe_count);

    const __half *query_row = query_latent + row * static_cast<std::int64_t>(latent_dim);
    const std::int64_t embryo = query_embryo[row];
    for (std::int64_t list_idx = lane; list_idx < list_count; list_idx += kWarpThreads) {
        if (hard_same_embryo && embryo >= 0 && list_embryo[list_idx] != embryo) continue;
        const float *centroid = centroids + list_idx * static_cast<std::int64_t>(latent_dim);
        insert_probe_candidate_device_(ProbeCandidate{
            dot_half_float_rows_(query_row, centroid, latent_dim),
            static_cast<std::int32_t>(list_idx)
        }, local_best, probe_count);
    }

    __shared__ ProbeCandidate shared_candidates[kWarpThreads * kMaxProbe];
    ProbeCandidate *thread_shared = shared_candidates + lane * kMaxProbe;
    for (int i = 0; i < probe_count; ++i) thread_shared[i] = local_best[i];
    __syncthreads();

    if (lane == 0) {
        ProbeCandidate merged[kMaxProbe];
        init_probe_candidates_device_(merged, probe_count);
        for (int thread = 0; thread < kWarpThreads; ++thread) {
            const ProbeCandidate *thread_candidates = shared_candidates + thread * kMaxProbe;
            for (int i = 0; i < probe_count; ++i) insert_probe_candidate_device_(thread_candidates[i], merged, probe_count);
        }
        for (int i = 0; i < probe_count; ++i) {
            selected_list_offsets[row * static_cast<std::int64_t>(probe_count) + i] = merged[i].list_offset;
        }
    }
}
