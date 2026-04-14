__global__ static void accumulate_gene_metrics_sliced_ell_kernel(
    csv::sliced_ell_view src,
    const unsigned char * __restrict__ keep_cells,
    float * __restrict__ gene_sum,
    float * __restrict__ gene_detected,
    float * __restrict__ gene_sq_sum
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

        if (keep_cells == 0 || keep_cells[row] != 0u) {
            for (unsigned int slot = lane; slot < width; slot += 32u) {
                const unsigned int col = src.col_idx[slot_base + slot];
                const float value = __half2float(src.val[slot_base + slot]);
                if (col < src.cols && value != 0.0f) {
                    atomicAdd(gene_sum + col, value);
                    atomicAdd(gene_detected + col, 1.0f);
                    atomicAdd(gene_sq_sum + col, value * value);
                }
            }
        }

        row += warp_stride;
    }
}
