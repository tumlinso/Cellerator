__global__ void accumulate_selected_feature_sums_kernel(csv::compressed_view src,
                                                        const unsigned int * __restrict__ selected,
                                                        unsigned int selected_count,
                                                        float * __restrict__ dst_a,
                                                        float * __restrict__ dst_b) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int nz = tid;

    while (nz < src.nnz) {
        const unsigned int gene = src.minorIdx[nz];
        const float value = __half2float(src.val[nz]);
        for (unsigned int k = 0; k < selected_count; ++k) {
            if (selected[k] != gene) continue;
            if (dst_a != nullptr) atomicAdd(dst_a + k, value);
            if (dst_b != nullptr) atomicAdd(dst_b + k, value);
            break;
        }
        nz += stride;
    }
}
