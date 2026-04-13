__global__ void csr_count_rows_kernel(matrix::Index nnz,
                                      const matrix::Index * __restrict__ row_idx,
                                      matrix::Index * __restrict__ row_ptr_shifted) {
    const matrix::Index tid = (matrix::Index) (blockIdx.x * blockDim.x + threadIdx.x);
    const matrix::Index stride = (matrix::Index) (gridDim.x * blockDim.x);
    matrix::Index i = tid;
    while (i < nnz) {
        atomicAdd(row_ptr_shifted + row_idx[i] + 1, 1u);
        i += stride;
    }
}
