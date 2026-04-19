__global__ void extract_sample_tile_blocked_ell_kernel(csv::blocked_ell_view src,
                                                       const unsigned int * __restrict__ sample_rows,
                                                       unsigned int sample_count,
                                                       const unsigned int * __restrict__ selected,
                                                       unsigned int selected_count,
                                                       float * __restrict__ out) {
    const unsigned int sample_idx = (unsigned int) blockIdx.x;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    if (sample_idx >= sample_count) return;

    const unsigned int row = sample_rows[sample_idx];
    if (row >= src.rows) return;

    for (unsigned int k = threadIdx.x; k < selected_count; k += blockDim.x) {
        out[(std::size_t) sample_idx * selected_count + k] = 0.0f;
    }
    __syncthreads();

    const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
    for (unsigned int ell_col = (unsigned int) threadIdx.x; ell_col < src.ell_cols; ell_col += (unsigned int) blockDim.x) {
        const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
        const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
        const unsigned int block_col = ell_width_blocks != 0u
            ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
            : cs::sparse::blocked_ell_invalid_col;
        const unsigned int gene = block_col != cs::sparse::blocked_ell_invalid_col
            ? block_col * block_size + lane
            : src.cols;
        if (gene >= src.cols) continue;
        const float value = __half2float(src.val[(unsigned long) row * src.ell_cols + ell_col]);
        if (value == 0.0f) continue;
        for (unsigned int k = 0; k < selected_count; ++k) {
            if (selected[k] != gene) continue;
            out[(std::size_t) sample_idx * selected_count + k] = value;
            break;
        }
    }
}
