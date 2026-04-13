__global__ void compute_row_norms_kernel(const __half *data, int rows, int cols, int ld, float *norms)
{
    const int row = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    float accum = 0.0f;
    int col = 0;
    const __half *row_ptr = data + (std::size_t) row * (std::size_t) ld;
    for (col = 0; col < cols; ++col) {
        const float v = __half2float(row_ptr[col]);
        accum += v * v;
    }
    norms[row] = accum;
}
