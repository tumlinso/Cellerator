__global__ void csr_spmm_bwd_values_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *rhs,
    std::uint32_t rows,
    std::int64_t grad_out_ld,
    std::int64_t out_cols,
    float *grad_values) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= rows) return;
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    const float *grad_row = grad_out + static_cast<std::int64_t>(row) * grad_out_ld;
    for (std::uint32_t idx = begin + static_cast<std::uint32_t>(threadIdx.x); idx < end; idx += static_cast<std::uint32_t>(blockDim.x)) {
        const float *rhs_row = rhs + static_cast<std::int64_t>(minor_idx[idx]) * grad_out_ld;
        float accum = 0.0f;
        for (std::int64_t col = 0; col < out_cols; ++col) accum += grad_row[col] * rhs_row[col];
        grad_values[idx] = accum;
    }
}
