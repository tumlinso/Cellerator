__global__ void csr_init_heads_kernel(matrix::Index rows,
                                      const matrix::Index * __restrict__ row_ptr,
                                      matrix::Index * __restrict__ heads) {
    const matrix::Index tid = (matrix::Index) (blockIdx.x * blockDim.x + threadIdx.x);
    const matrix::Index stride = (matrix::Index) (gridDim.x * blockDim.x);
    matrix::Index i = tid;
    while (i < rows) {
        heads[i] = row_ptr[i];
        i += stride;
    }
}
