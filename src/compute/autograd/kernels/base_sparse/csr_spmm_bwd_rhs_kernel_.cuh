__global__ void csr_spmm_bwd_rhs_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    std::int64_t grad_out_ld,
    std::int64_t out_cols,
    float *grad_rhs,
    std::int64_t grad_rhs_ld) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::int64_t col = static_cast<std::int64_t>(threadIdx.x) + static_cast<std::int64_t>(blockIdx.y) * blockDim.x;
    if (row >= rows || col >= out_cols) return;
    const float grad_value = grad_out[static_cast<std::int64_t>(row) * grad_out_ld + col];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        atomicAdd(
            grad_rhs + static_cast<std::int64_t>(minor_idx[idx]) * grad_rhs_ld + col,
            primitives::load_f16_as_f32(values, idx) * grad_value);
    }
}
