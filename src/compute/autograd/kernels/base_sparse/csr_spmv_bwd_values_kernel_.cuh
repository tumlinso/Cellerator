__global__ void csr_spmv_bwd_values_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *vector,
    std::uint32_t rows,
    float *grad_values) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float grad_row = grad_out[row];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        grad_values[idx] = grad_row * vector[minor_idx[idx]];
    }
}
