__global__ static void normalize_log1p_blocked_ell_kernel(
    csv::blocked_ell_view src,
    const float * __restrict__ total_counts,
    const unsigned char * __restrict__ keep_cells,
    float target_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const float denom = total_counts[row];
        const float scale = denom > 0.0f ? target_sum / denom : 0.0f;
        const int keep = keep_cells == 0 ? 1 : keep_cells[row] != 0;
        unsigned int ell_col = lane;

        if (keep) {
            while (ell_col < src.ell_cols) {
                const unsigned long offset = (unsigned long) row * src.ell_cols + ell_col;
                const float value = __half2float(src.val[offset]);
                src.val[offset] = value != 0.0f ? __float2half(log1pf(value * scale)) : __float2half(0.0f);
                ell_col += 32u;
            }
        }

        row += warp_stride;
    }
}
