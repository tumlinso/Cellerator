__global__ void csr_spmv_bwd_vector_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    float *grad_vector) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float grad_row = grad_out[row];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        atomicAdd(grad_vector + minor_idx[idx], primitives::load_f16_as_f32(values, idx) * grad_row);
    }
}
