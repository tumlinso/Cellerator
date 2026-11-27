__global__ void widen_half_to_float_kernel_(const __half *src, std::uint32_t count, float *dst) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = primitives::load_f16_as_f32(src, idx);
}
