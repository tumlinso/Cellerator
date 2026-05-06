__device__ inline float dot_query_blocked_row_(
    const __half *query_row,
    int latent_dim,
    const std::uint32_t *block_col_idx,
    const __half *blocked_values,
    std::int64_t index_row,
    int block_size,
    int ell_cols) {
    if (block_size <= 0 || ell_cols <= 0) return 0.0f;
    const int row_block = static_cast<int>(index_row / static_cast<std::int64_t>(block_size));
    const int width = ell_cols / block_size;
    float sum = 0.0f;
    for (int slot = 0; slot < width; ++slot) {
        const std::uint32_t block_col = block_col_idx[static_cast<std::size_t>(row_block) * static_cast<std::size_t>(width)
            + static_cast<std::size_t>(slot)];
        if (block_col == css::blocked_ell_invalid_col) continue;
        const std::int64_t base_col = static_cast<std::int64_t>(block_col) * static_cast<std::int64_t>(block_size);
        const std::int64_t value_base = index_row * static_cast<std::int64_t>(ell_cols)
            + static_cast<std::int64_t>(slot) * static_cast<std::int64_t>(block_size);
        for (int local_col = 0; local_col < block_size; ++local_col) {
            const std::int64_t col = base_col + static_cast<std::int64_t>(local_col);
            if (col >= latent_dim) break;
            sum += __half2float(query_row[col]) * __half2float(blocked_values[value_base + local_col]);
        }
    }
    return sum;
}

__global__ void blocked_ann_refine_kernel_(
    const __half *query_latent,
    const float *query_lower,
    const float *query_upper,
    const std::int64_t *query_embryo,
    const std::uint32_t *block_col_idx,
    const __half *blocked_values,
    const float *index_time,
    const std::int64_t *index_embryo,
    const std::int64_t *index_cell,
    const std::int32_t *selected_list_offsets,
    const std::int64_t *list_row_begin,
    const std::int64_t *list_row_end,
    std::int64_t query_rows,
    int latent_dim,
    std::int64_t shard_index,
    int block_size,
    int ell_cols,
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
                dot_query_blocked_row_(query_row, latent_dim, block_col_idx, blocked_values, index_row, block_size, ell_cols),
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
