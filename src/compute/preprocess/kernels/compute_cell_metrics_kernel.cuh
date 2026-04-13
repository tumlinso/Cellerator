__global__ static void compute_cell_metrics_kernel(
    csv::compressed_view src,
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
        const unsigned int begin = src.majorPtr[row];
        const unsigned int end = src.majorPtr[row + 1u];
        const unsigned int count = end - begin;
        float sum = 0.0f;
        float mito = 0.0f;
        float vmax = 0.0f;
        unsigned int idx = begin + lane;

        while (idx < end) {
            const float value = __half2float(src.val[idx]);
            const unsigned int gene = src.minorIdx[idx];
            sum += value;
            if (gene_flags != 0 && (gene_flags[gene] & gene_flag_mito) != 0u) mito += value;
            vmax = fmaxf(vmax, value);
            idx += 32u;
        }

        sum = reduce::warp_sum(sum);
        mito = reduce::warp_sum(mito);
        vmax = reduce::warp_max(vmax);

        if (lane == 0u) {
            total_counts[row] = sum;
            mito_counts[row] = mito;
            max_counts[row] = vmax;
            detected_genes[row] = count;
            if (keep_cells != 0) {
                const float mito_fraction = sum > 0.0f ? mito / sum : 0.0f;
                keep_cells[row] = (unsigned char) (sum >= filter.min_counts &&
                                                   count >= filter.min_genes &&
                                                   mito_fraction <= filter.max_mito_fraction);
            }
        }

        row += warp_stride;
    }
}
