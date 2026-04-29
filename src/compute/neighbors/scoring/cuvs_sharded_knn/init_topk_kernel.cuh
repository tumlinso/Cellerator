__global__ void init_topk_kernel(int total, int select_min, float *values, std::int64_t *indices)
{
    const int tid = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const float fill = select_min ? CUDART_INF_F : -CUDART_INF_F;
    int idx = tid;
    while (idx < total) {
        values[idx] = fill;
        indices[idx] = (std::int64_t) -1;
        idx += (int) (gridDim.x * blockDim.x);
    }
}
