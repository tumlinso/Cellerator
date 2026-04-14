__global__ static void compute_cell_metrics_sliced_ell_kernel(
    csv::sliced_ell_view src,
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
    unsigned int row = warp_global;

    while (row < src.rows) {
        unsigned int slice = 0u;
        unsigned int row_begin = 0u;
        unsigned int width = 0u;
        unsigned long slot_base = 0ul;
        float sum = 0.0f;
        float mito = 0.0f;
        float vmax = 0.0f;
        float detected = 0.0f;

        if (src.slice_count != 0u) {
            if (src.slice_rows == 32u) {
                slice = row >> 5;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else if (src.slice_rows != 0u) {
                slice = row / src.slice_rows;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else {
                while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
            }
            row_begin = src.slice_row_offsets[slice];
            width = src.slice_widths[slice];
            slot_base = (unsigned long) src.slice_slot_offsets[slice]
                + (unsigned long) (row - row_begin) * (unsigned long) width;
        }

        for (unsigned int slot = lane; slot < width; slot += 32u) {
            const unsigned int col = src.col_idx[slot_base + slot];
            const float value = __half2float(src.val[slot_base + slot]);
            if (col < src.cols && value != 0.0f) {
                sum += value;
                if (gene_flags != 0 && (gene_flags[col] & gene_flag_mito) != 0u) mito += value;
                vmax = fmaxf(vmax, value);
                detected += 1.0f;
            }
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
                keep_cells[row] = (unsigned char) (sum >= filter.min_counts
                    && (unsigned int) detected >= filter.min_genes
                    && mito_fraction <= filter.max_mito_fraction);
            }
        }

        row += warp_stride;
    }
}
