__global__ void feature_affine_quantize_baseline_kernel_(
    std::uint32_t rows,
    std::uint32_t cols,
    const float *log_scale,
    const float *offset,
    float scale_floor,
    float max_code,
    float reconstruction_weight,
    float range_weight,
    float min_dynamic_range,
    float inv_elements,
    float inv_features,
    float *reconstruction_loss,
    float *range_loss,
    float *grad_log_scale,
    float *grad_offset) {
    const std::uint32_t col = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    const float log_scale_value = log_scale[col];
    const float offset_value = offset[col];
    const float scale = softplus_f32_(log_scale_value) + scale_floor;
    const float scale_grad_factor = sigmoid_f32_(log_scale_value);
    const quantize_entry_eval_ zero_eval = eval_quantize_entry_(0.0f, scale, offset_value, max_code);

    const float rows_f = static_cast<float>(rows);
    const float reconstruction_loss_value = rows_f * zero_eval.squared_error * inv_elements;

    float range_loss_value = offset_value * offset_value * inv_features;
    float grad_offset_value = rows_f * zero_eval.grad_offset * inv_elements * reconstruction_weight
        + 2.0f * offset_value * inv_features * range_weight;
    float grad_scale_value = rows_f * zero_eval.grad_scale * inv_elements * reconstruction_weight;

    const float dynamic_range = scale * max_code;
    if (dynamic_range < min_dynamic_range) {
        const float diff = min_dynamic_range - dynamic_range;
        range_loss_value += diff * diff * inv_features;
        grad_scale_value += -2.0f * diff * max_code * inv_features * range_weight;
    }

    grad_log_scale[col] = grad_scale_value * scale_grad_factor;
    grad_offset[col] = grad_offset_value;
    atomicAdd(reconstruction_loss, reconstruction_loss_value);
    atomicAdd(range_loss, range_loss_value);
}
