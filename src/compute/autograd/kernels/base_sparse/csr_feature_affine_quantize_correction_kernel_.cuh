__global__ void csr_feature_affine_quantize_correction_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
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
    if (row >= rows) return;

    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        const std::uint32_t col = minor_idx[idx];
        const float log_scale_value = log_scale[col];
        const float offset_value = offset[col];
        const float scale = softplus_f32_(log_scale_value) + scale_floor;
        const float scale_grad_factor = sigmoid_f32_(log_scale_value);
        const quantize_entry_eval_ zero_eval = eval_quantize_entry_(0.0f, scale, offset_value, max_code);
        const quantize_entry_eval_ value_eval = eval_quantize_entry_(
            primitives::load_f16_as_f32(values, idx),
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
