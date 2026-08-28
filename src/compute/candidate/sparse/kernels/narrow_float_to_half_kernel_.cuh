__global__ void narrow_float_to_half_kernel_(const float *src, std::uint32_t count, __half *dst) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = __float2half(src[idx]);
}
