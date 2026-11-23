__global__ static void shift_ptr_idx_count(
    const unsigned int nnz,
    const unsigned int * __restrict__ axIdx,
    unsigned int * __restrict__ axPtr_shifted
) {
    const unsigned int tid = (unsigned int) ::cellshard::cuda_index::global_tid_1d();
    const unsigned int stride = (unsigned int) ::cellshard::cuda_index::global_stride_1d();
    unsigned int i = tid;
    while (i < nnz) {
        ::cellshard::cuda_index::atomic_add_u32(axPtr_shifted + ::cellshard::cuda_index::add_u32(axIdx[i], 1u), 1u);
        i += stride;
    }
}
