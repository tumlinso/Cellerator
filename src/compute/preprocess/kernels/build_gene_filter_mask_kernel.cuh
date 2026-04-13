__global__ static void build_gene_filter_mask_kernel(
    unsigned int cols,
    float inv_cells,
    gene_filter_params filter,
    const float * __restrict__ sum,
    const float * __restrict__ sq_sum,
    const float * __restrict__ detected_cells,
    unsigned char * __restrict__ keep
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int gene = tid;

    while (gene < cols) {
        const float mean = sum[gene] * inv_cells;
        const float var = fmaxf(sq_sum[gene] * inv_cells - mean * mean, 0.0f);
        keep[gene] = (unsigned char) (sum[gene] >= filter.min_sum &&
                                      detected_cells[gene] >= filter.min_detected_cells &&
                                      var >= filter.min_variance);
        gene += stride;
    }
}
