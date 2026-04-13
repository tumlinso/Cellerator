__global__ void blocked_ell_spmm_fwd_kernel_(
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::int64_t col = static_cast<std::int64_t>(threadIdx.x) + static_cast<std::int64_t>(blockIdx.y) * blockDim.x;
    if (row >= rows || col >= out_cols || block_size == 0u || ell_cols == 0u) return;

    const std::uint32_t row_block = row / block_size;
    const std::uint32_t ell_width = ell_cols / block_size;
    float accum = 0.0f;

    for (std::uint32_t slot = 0u; slot < ell_width; ++slot) {
        const std::uint32_t block_col = block_col_idx[static_cast<std::size_t>(row_block) * ell_width + slot];
        if (block_col == cs::sparse::blocked_ell_invalid_col) continue;
        const std::int64_t rhs_base = static_cast<std::int64_t>(block_col) * static_cast<std::int64_t>(block_size);
        const std::size_t values_base = static_cast<std::size_t>(row) * ell_cols + static_cast<std::size_t>(slot) * block_size;
        for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
            const std::int64_t rhs_col = rhs_base + static_cast<std::int64_t>(col_in_block);
            if (rhs_col >= cols) break;
            accum += primitives::load_f16_as_f32(values, values_base + col_in_block) * rhs[rhs_col * rhs_ld + col];
        }
    }
    out[static_cast<std::int64_t>(row) * out_ld + col] = accum;
}
