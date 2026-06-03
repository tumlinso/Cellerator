__global__ void normalize_rows_in_place_kernel(__half *data, int rows, int cols, int ld)
{
    const int row = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    float accum = 0.0f;
    float inv_norm = 0.0f;
    int col = 0;
    __half *row_ptr = data + (std::size_t) row * (std::size_t) ld;
    for (col = 0; col < cols; ++col) {
        const float v = __half2float(row_ptr[col]);
        accum += v * v;
    }
    inv_norm = accum > 1.0e-20f ? rsqrtf(accum) : 0.0f;
    for (col = 0; col < cols; ++col) {
        row_ptr[col] = __float2half(__half2float(row_ptr[col]) * inv_norm);
    }
}
