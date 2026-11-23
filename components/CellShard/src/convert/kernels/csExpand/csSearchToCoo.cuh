__global__ static void csSearchToCoo(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int * __restrict__ cAxPtr,
    const unsigned int * __restrict__ uAxIdx,
    const __half * __restrict__ val,
    unsigned int * __restrict__ out_cAxIdx,
    unsigned int * __restrict__ out_uAxIdx,
    __half * __restrict__ out_val
) {
    const unsigned int tid = (unsigned int) ::cellshard::cuda_index::global_tid_1d();
    const unsigned int stride = (unsigned int) ::cellshard::cuda_index::global_stride_1d();
    unsigned int i = tid;

    while (i < nnz) {
        unsigned int lo = 0;
        unsigned int hi = cDim;

        while (lo < hi) {
            const unsigned int mid = ::cellshard::cuda_index::add_u32(lo, ::cellshard::cuda_index::shr_u32(::cellshard::cuda_index::sub_u32(hi, lo), 1u));
            if (i >= cAxPtr[mid + 1u]) lo = ::cellshard::cuda_index::add_u32(mid, 1u);
            else hi = mid;
        }

        out_cAxIdx[i] = lo;
        out_uAxIdx[i] = uAxIdx[i];
        out_val[i] = val[i];
        i += stride;
    }
}
