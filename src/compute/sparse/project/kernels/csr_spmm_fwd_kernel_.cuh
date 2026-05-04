__global__ void csr_spmm_fwd_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::int64_t col = static_cast<std::int64_t>(threadIdx.x) + static_cast<std::int64_t>(blockIdx.y) * blockDim.x;
    if (row >= rows || col >= out_cols) return;

    float accum = 0.0f;
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        accum += primitives::load_f16_as_f32(values, idx) * rhs[static_cast<std::int64_t>(minor_idx[idx]) * rhs_ld + col];
    }
    out[static_cast<std::int64_t>(row) * out_ld + col] = accum;
}
