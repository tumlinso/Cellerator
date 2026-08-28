__global__ void csr_row_scale_bwd_scales_kernel_(
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    float *grad_scales) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    float accum = 0.0f;
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        accum += primitives::load_f16_as_f32(values, idx) * grad_out[idx];
    }
    grad_scales[row] = accum;
}
