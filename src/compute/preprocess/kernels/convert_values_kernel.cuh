__global__ static void convert_values_kernel(
    unsigned int nnz,
    const __half * __restrict__ src,
    float * __restrict__ dst
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < nnz) {
        dst[i] = __half2float(src[i]);
        i += stride;
    }
}
