__global__ void csr_row_scale_fwd_kernel_(
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *row_scales,
    std::uint32_t rows,
    float *out_values) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float scale = row_scales[row];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        out_values[idx] = primitives::load_f16_as_f32(values, idx) * scale;
    }
}
