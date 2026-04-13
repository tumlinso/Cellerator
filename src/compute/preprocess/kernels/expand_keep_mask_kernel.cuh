__global__ static void expand_keep_mask_kernel(unsigned int n,
                                              const unsigned char * __restrict__ keep,
                                              float * __restrict__ dst) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < n) {
        dst[i] = keep != 0 && keep[i] != 0 ? 1.0f : 0.0f;
        i += stride;
    }
}
