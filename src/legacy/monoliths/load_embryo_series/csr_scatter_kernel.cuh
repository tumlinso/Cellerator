__global__ void csr_scatter_kernel(matrix::Index nnz,
                                   const matrix::Index * __restrict__ row_idx,
                                   const matrix::Index * __restrict__ col_idx,
                                   const matrix::Real * __restrict__ val,
                                   matrix::Index * __restrict__ heads,
                                   matrix::Index * __restrict__ out_col,
                                   matrix::Real * __restrict__ out_val) {
    const matrix::Index tid = (matrix::Index) (blockIdx.x * blockDim.x + threadIdx.x);
    const matrix::Index stride = (matrix::Index) (gridDim.x * blockDim.x);
    matrix::Index i = tid;
    while (i < nnz) {
        const matrix::Index dst = atomicAdd(heads + row_idx[i], 1u);
        out_col[dst] = col_idx[i];
        out_val[dst] = val[i];
        i += stride;
    }
}
