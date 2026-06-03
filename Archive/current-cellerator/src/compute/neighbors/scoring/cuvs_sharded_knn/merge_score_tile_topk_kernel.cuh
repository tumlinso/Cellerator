__global__ void merge_score_tile_topk_kernel(const float *scores,
                                             int query_rows,
                                             int index_rows,
                                             int k,
                                             int select_min,
                                             unsigned long global_query_begin,
                                             unsigned long global_index_begin,
                                             int exclude_self,
                                             float *best_values,
                                             std::int64_t *best_indices)
{
    const int row = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= query_rows) return;

    const std::size_t base = (std::size_t) row * (std::size_t) k;
    int col = 0;
    for (col = 0; col < index_rows; ++col) {
        const std::int64_t global_id = (std::int64_t) global_index_begin + (std::int64_t) col;
        const float value = scores[(std::size_t) row * (std::size_t) index_rows + (std::size_t) col];
        int insert = 0;
        if (exclude_self && global_id == (std::int64_t) global_query_begin + (std::int64_t) row) continue;
        if (!isfinite(value)) continue;
        if (!better_value(value, best_values[base + (std::size_t) k - 1u], select_min)) continue;
        insert = k - 1;
        while (insert > 0 && better_value(value, best_values[base + (std::size_t) insert - 1u], select_min)) {
            best_values[base + (std::size_t) insert] = best_values[base + (std::size_t) insert - 1u];
            best_indices[base + (std::size_t) insert] = best_indices[base + (std::size_t) insert - 1u];
            --insert;
        }
        best_values[base + (std::size_t) insert] = value;
        best_indices[base + (std::size_t) insert] = global_id;
    }
}
