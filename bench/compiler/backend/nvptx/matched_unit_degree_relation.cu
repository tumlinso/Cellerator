#include <cuda_runtime.h>

extern "C" __global__ void ce_unit_degree_relation_apply_sm70(
    const unsigned* columns,
    const float* weights,
    const float* input,
    float* output,
    unsigned rows) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) output[row] = weights[row] * input[columns[row]];
}
