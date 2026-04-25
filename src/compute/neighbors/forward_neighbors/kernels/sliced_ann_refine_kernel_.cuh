__global__ void sliced_ann_refine_kernel_(
    const __half *query_latent,
    const float *query_lower,
    const float *query_upper,
    const std::int64_t *query_embryo,
    const std::uint32_t *row_slot_offsets,
    const std::uint32_t *row_widths,
    const std::uint32_t *col_idx,
    const __half *values,
    const float *index_time,
    const std::int64_t *index_embryo,
    const std::int64_t *index_cell,
    const std::int32_t *selected_list_offsets,
    const std::int64_t *list_row_begin,
    const std::int64_t *list_row_end,
    std::int64_t query_rows,
    int latent_dim,
    std::int64_t shard_index,
    int hard_same_embryo,
    int probe_count,
    int k,
    std::int64_t *best_cell,
    std::int64_t *best_shard,
    float *best_time,
    std::int64_t *best_embryo,
    float *best_similarity) {
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) *
            static_cast<std::int64_t>(kForwardNeighborWarpsPerBlock) +
        static_cast<std::int64_t>(warp);
    if (row >= query_rows) return;
    const unsigned warp_mask = __activemask();
    __shared__ Candidate merged_shared[kForwardNeighborWarpsPerBlock * kMaxTopK];
    Candidate *merged = merged_shared + warp * kMaxTopK;

    Candidate local_best[kMaxTopK];
    init_candidates_device_(local_best, k);

    const __half *query_row = query_latent + row * static_cast<std::int64_t>(latent_dim);
    const float lower = query_lower[row];
    const float upper = query_upper[row];
    const std::int64_t embryo = query_embryo[row];

    for (int slot = 0; slot < probe_count; ++slot) {
        const std::int32_t list_offset = selected_list_offsets[row * static_cast<std::int64_t>(probe_count) + slot];
        if (list_offset < 0) continue;
        const std::int64_t begin = list_row_begin[list_offset];
        const std::int64_t end = list_row_end[list_offset];
        for (std::int64_t index_row = begin + lane; index_row < end; index_row += kWarpThreads) {
            const float time = index_time[index_row];
            if (!(time > lower)) continue;
            if (std::isfinite(upper) && time > upper) continue;
            if (hard_same_embryo && embryo >= 0 && index_embryo[index_row] != embryo) continue;
            insert_candidate_device_(Candidate{
                dot_query_sliced_row_(query_row, row_slot_offsets, row_widths, col_idx, values, index_row),
                time,
                index_embryo[index_row],
                index_cell[index_row],
                shard_index
            }, local_best, k);
        }
    }

    if (lane == 0) {
        init_candidates_device_(merged, k);
        const std::int64_t row_base = row * static_cast<std::int64_t>(k);
        for (int i = 0; i < k; ++i) {
            merged[i] = Candidate{
                best_similarity[row_base + i],
                best_time[row_base + i],
                best_embryo[row_base + i],
                best_cell[row_base + i],
                best_shard[row_base + i]
            };
        }
    }
    for (int src_lane = 0; src_lane < kWarpThreads; ++src_lane) {
        for (int i = 0; i < k; ++i) {
            const Candidate candidate = shfl_candidate_device_(warp_mask, local_best[i], src_lane);
            if (lane == 0) insert_candidate_device_(candidate, merged, k);
        }
    }
    if (lane == 0) {
        const std::int64_t row_base = row * static_cast<std::int64_t>(k);
        for (int i = 0; i < k; ++i) {
            best_similarity[row_base + i] = merged[i].similarity;
            best_time[row_base + i] = merged[i].developmental_time;
            best_embryo[row_base + i] = merged[i].embryo_id;
            best_cell[row_base + i] = merged[i].cell_index;
            best_shard[row_base + i] = merged[i].shard_index;
        }
    }
}
