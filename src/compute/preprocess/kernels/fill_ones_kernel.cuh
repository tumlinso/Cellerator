__global__ static void fill_ones_kernel(unsigned int n, float * __restrict__ dst) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < n) {
        dst[i] = 1.0f;
        i += stride;
    }
}
