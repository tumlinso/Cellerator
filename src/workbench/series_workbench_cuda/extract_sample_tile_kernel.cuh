__global__ void extract_sample_tile_kernel(csv::compressed_view src,
                                           const unsigned int * __restrict__ sample_rows,
                                           unsigned int sample_count,
                                           const unsigned int * __restrict__ selected,
                                           unsigned int selected_count,
                                           float * __restrict__ out) {
    const unsigned int sample_idx = (unsigned int) blockIdx.x;
    if (sample_idx >= sample_count) return;

    const unsigned int row = sample_rows[sample_idx];
    if (row >= src.rows) return;

    for (unsigned int k = threadIdx.x; k < selected_count; k += blockDim.x) {
        out[(std::size_t) sample_idx * selected_count + k] = 0.0f;
    }
    __syncthreads();

    const unsigned int begin = src.majorPtr[row];
    const unsigned int end = src.majorPtr[row + 1u];
    for (unsigned int nz = begin + (unsigned int) threadIdx.x; nz < end; nz += (unsigned int) blockDim.x) {
        const unsigned int gene = src.minorIdx[nz];
        const float value = __half2float(src.val[nz]);
        for (unsigned int k = 0; k < selected_count; ++k) {
            if (selected[k] != gene) continue;
            out[(std::size_t) sample_idx * selected_count + k] = value;
            break;
        }
    }
}
