__global__ void finalize_sparse_partition_kernel(int rows,
                                                 int k,
                                                 unsigned long global_row_begin,
                                                 int filter_self,
                                                 int negate_distance,
                                                 const int *neighbors_in,
                                                 float *distances_inout,
                                                 std::int64_t *neighbors_out)
{
    const int tid = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = rows * k;
    int idx = tid;
    while (idx < total) {
        const int row = idx / k;
        int col = neighbors_in[idx];
        float distance = distances_inout[idx];
        if (filter_self && col == row) {
            col = -1;
            distance = CUDART_INF_F;
        }
        if (negate_distance && isfinite(distance)) distance = -distance;
        neighbors_out[idx] = col >= 0 ? (std::int64_t) col + (std::int64_t) global_row_begin : (std::int64_t) -1;
        distances_inout[idx] = distance;
        idx += (int) (gridDim.x * blockDim.x);
    }
}
