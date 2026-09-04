#include <cuda_runtime.h>

__global__ void add_one(const float* input, float* output) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    output[index] = input[index] + 1.0F;
}
