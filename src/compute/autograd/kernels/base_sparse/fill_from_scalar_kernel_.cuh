__global__ void fill_from_scalar_kernel_(const float *grad_scalar, std::uint32_t count, float *dst) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = *grad_scalar;
}
