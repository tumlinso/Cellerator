__global__ void csr_spmv_fwd_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    const float *vector,
    float *out) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float accum = 0.0f;
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        accum += primitives::load_f16_as_f32(values, idx) * vector[minor_idx[idx]];
    }
    out[row] = accum;
}
