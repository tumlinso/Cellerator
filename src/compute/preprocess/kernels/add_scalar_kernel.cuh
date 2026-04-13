__global__ static void add_scalar_kernel(float * __restrict__ dst, const float * __restrict__ src) {
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[0] += src[0];
}
