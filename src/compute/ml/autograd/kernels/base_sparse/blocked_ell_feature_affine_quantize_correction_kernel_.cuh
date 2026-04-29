__global__ void blocked_ell_feature_affine_quantize_correction_kernel_(
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const float *log_scale,
    const float *offset,
    float scale_floor,
    float max_code,
    float reconstruction_weight,
    float inv_elements,
    float *reconstruction_loss,
    float *grad_log_scale,
    float *grad_offset) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || block_size == 0u || ell_cols == 0u) return;

    const std::uint32_t row_block = row / block_size;
    const std::uint32_t ell_width = ell_cols / block_size;
    const std::size_t row_values_base = static_cast<std::size_t>(row) * ell_cols;
    const std::size_t row_block_base = static_cast<std::size_t>(row_block) * ell_width;

    for (std::uint32_t slot = 0u; slot < ell_width; ++slot) {
        const std::uint32_t block_col = block_col_idx[row_block_base + slot];
        if (block_col == cs::sparse::blocked_ell_invalid_col) continue;
        const std::uint32_t col_base = block_col * block_size;
        const std::size_t values_base = row_values_base + static_cast<std::size_t>(slot) * block_size;
        for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
            const std::uint32_t col = col_base + col_in_block;
            if (col >= cols) break;

            const float log_scale_value = log_scale[col];
            const float offset_value = offset[col];
            const float scale = softplus_f32_(log_scale_value) + scale_floor;
            const float scale_grad_factor = sigmoid_f32_(log_scale_value);
            const quantize_entry_eval_ zero_eval = eval_quantize_entry_(0.0f, scale, offset_value, max_code);
            const quantize_entry_eval_ value_eval = eval_quantize_entry_(
                primitives::load_f16_as_f32(values, values_base + col_in_block),
                scale,
                offset_value,
                max_code);

            atomicAdd(
                reconstruction_loss,
                (value_eval.squared_error - zero_eval.squared_error) * inv_elements);
            atomicAdd(
                grad_offset + col,
                (value_eval.grad_offset - zero_eval.grad_offset) * inv_elements * reconstruction_weight);
            atomicAdd(
                grad_log_scale + col,
                (value_eval.grad_scale - zero_eval.grad_scale) * inv_elements * reconstruction_weight * scale_grad_factor);
        }
    }
}
