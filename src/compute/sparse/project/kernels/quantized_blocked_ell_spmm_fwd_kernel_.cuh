template<int Bits, typename Metadata>
__global__ void quantized_blocked_ell_spmm_fwd_kernel_(
    ::cellerator::quantized::blocked_ell::matrix<Bits, float, Metadata> matrix,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::int64_t col = static_cast<std::int64_t>(threadIdx.x) + static_cast<std::int64_t>(blockIdx.y) * blockDim.x;
    if (row >= static_cast<std::uint32_t>(matrix.rows) || col >= out_cols || matrix.block_size <= 0 || matrix.ell_cols <= 0) return;

    const std::uint32_t row_block = row / static_cast<std::uint32_t>(matrix.block_size);
    const std::uint32_t ell_width = static_cast<std::uint32_t>(matrix.ell_cols / matrix.block_size);
    const auto row_cache = matrix.metadata.prepare_row(static_cast<int>(row));
    float accum = 0.0f;

    for (std::uint32_t slot = 0u; slot < ell_width; ++slot) {
        const std::uint32_t block_col = matrix.block_col_idx[static_cast<std::size_t>(row_block) * ell_width + slot];
        if (block_col == ::cellerator::quantized::blocked_ell::invalid_block_col) continue;
        const std::int64_t rhs_base = static_cast<std::int64_t>(block_col) * static_cast<std::int64_t>(matrix.block_size);
        const std::uint32_t slot_base = slot * static_cast<std::uint32_t>(matrix.block_size);
        for (std::uint32_t col_in_block = 0u; col_in_block < static_cast<std::uint32_t>(matrix.block_size); ++col_in_block) {
            const std::int64_t rhs_col = rhs_base + static_cast<std::int64_t>(col_in_block);
            if (rhs_col >= static_cast<std::int64_t>(matrix.cols)) break;
            const unsigned int code = ::cellerator::quantized::blocked_ell::get_code(
                &matrix,
                static_cast<int>(row),
                static_cast<int>(slot_base + col_in_block));
            const float value = ::cellerator::quantized::dequantize_code<Bits>(
                code,
                matrix.metadata,
                row_cache,
                static_cast<int>(rhs_col));
            accum += value * rhs[rhs_col * rhs_ld + col];
        }
    }
    out[static_cast<std::int64_t>(row) * out_ld + col] = accum;
}
