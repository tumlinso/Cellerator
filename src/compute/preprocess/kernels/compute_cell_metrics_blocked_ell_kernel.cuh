__global__ static void compute_cell_metrics_blocked_ell_kernel(
    csv::blocked_ell_view src,
    const unsigned char * __restrict__ gene_flags,
    cell_filter_params filter,
    float * __restrict__ total_counts,
    float * __restrict__ mito_counts,
    float * __restrict__ max_counts,
    unsigned int * __restrict__ detected_genes,
    unsigned char * __restrict__ keep_cells
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned int row = warp_global;

    while (row < src.rows) {
        float sum = 0.0f;
        float mito = 0.0f;
        float vmax = 0.0f;
        float detected = 0.0f;
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        unsigned int ell_col = lane;

        while (ell_col < src.ell_cols) {
            const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
            const unsigned int lane_in_block = block_size != 0u ? ell_col % block_size : 0u;
            const unsigned int block_col = ell_width_blocks != 0u
                ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                : cs::sparse::blocked_ell_invalid_col;
            const unsigned int gene = block_col != cs::sparse::blocked_ell_invalid_col
                ? block_col * block_size + lane_in_block
                : src.cols;
            const float value = __half2float(src.val[(unsigned long) row * src.ell_cols + ell_col]);
            if (gene < src.cols && value != 0.0f) {
                sum += value;
                if (gene_flags != 0 && (gene_flags[gene] & gene_flag_mito) != 0u) mito += value;
                vmax = fmaxf(vmax, value);
                detected += 1.0f;
            }
            ell_col += 32u;
        }

        sum = reduce::warp_sum(sum);
        mito = reduce::warp_sum(mito);
        vmax = reduce::warp_max(vmax);
        detected = reduce::warp_sum(detected);

        if (lane == 0u) {
            total_counts[row] = sum;
            mito_counts[row] = mito;
            max_counts[row] = vmax;
            detected_genes[row] = (unsigned int) detected;
            if (keep_cells != 0) {
                const float mito_fraction = sum > 0.0f ? mito / sum : 0.0f;
                keep_cells[row] = (unsigned char) (sum >= filter.min_counts &&
                                                   (unsigned int) detected >= filter.min_genes &&
                                                   mito_fraction <= filter.max_mito_fraction);
            }
        }

        row += warp_stride;
    }
}
