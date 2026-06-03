__global__ void postprocess_score_tile_kernel(float *scores,
                                              const float *query_norms,
                                              const float *index_norms,
                                              int query_rows,
                                              int index_rows,
                                              int metric)
{
    const int tid = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = query_rows * index_rows;
    int idx = tid;
    while (idx < total) {
        const int row = idx / index_rows;
        const int col = idx - row * index_rows;
        const float dot = scores[idx];
        float value = dot;
        if (metric == proprietary_metric_l2) {
            value = query_norms[row] + index_norms[col] - 2.0f * dot;
            if (value < 0.0f) value = 0.0f;
        } else if (metric == proprietary_metric_cosine) {
            value = 1.0f - dot;
        }
        scores[idx] = value;
        idx += (int) (gridDim.x * blockDim.x);
    }
}
