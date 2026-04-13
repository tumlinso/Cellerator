__global__ void accumulate_selected_feature_sums_blocked_ell_kernel(csv::blocked_ell_view src,
                                                                    const unsigned int * __restrict__ selected,
                                                                    unsigned int selected_count,
                                                                    float * __restrict__ dst_a,
                                                                    float * __restrict__ dst_b) {
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
        const unsigned int gene = block_col != cs::sparse::blocked_ell_invalid_col
            ? block_col * block_size + lane
            : src.cols;
        if (gene < src.cols) {
            const float value = __half2float(src.val[linear]);
            if (value != 0.0f) {
                for (unsigned int k = 0; k < selected_count; ++k) {
                    if (selected[k] != gene) continue;
                    if (dst_a != nullptr) atomicAdd(dst_a + k, value);
                    if (dst_b != nullptr) atomicAdd(dst_b + k, value);
                    break;
                }
            }
        }
        linear += stride;
    }
}
