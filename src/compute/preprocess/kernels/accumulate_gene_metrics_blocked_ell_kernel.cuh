__global__ static void accumulate_gene_metrics_blocked_ell_kernel(
    csv::blocked_ell_view src,
    const unsigned char * __restrict__ keep_cells,
    float * __restrict__ gene_sum,
    float * __restrict__ gene_detected,
    float * __restrict__ gene_sq_sum
) {
    const unsigned long tid = (unsigned long) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned long stride = (unsigned long) (gridDim.x * blockDim.x);
    const unsigned long total = (unsigned long) src.rows * (unsigned long) src.ell_cols;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned long linear = tid;

    while (linear < total) {
        const unsigned int row = (unsigned int) (linear / src.ell_cols);
        const unsigned int ell_col = (unsigned int) (linear % src.ell_cols);
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
        const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
        const unsigned int block_col = ell_width_blocks != 0u
            ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
            : cs::sparse::blocked_ell_invalid_col;
        const unsigned int col = block_col != cs::sparse::blocked_ell_invalid_col
            ? block_col * block_size + lane
            : src.cols;
        if ((keep_cells == 0 || keep_cells[row] != 0u) && col < src.cols) {
            const float value = __half2float(src.val[linear]);
            if (value != 0.0f) {
                atomicAdd(gene_sum + col, value);
                atomicAdd(gene_detected + col, 1.0f);
                atomicAdd(gene_sq_sum + col, value * value);
            }
        }
        linear += stride;
    }
}
