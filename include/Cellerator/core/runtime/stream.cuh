#pragma once

#include <cuda_runtime.h>

namespace cellerator::core::runtime {

struct execution_context {
    int device = -1;
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
};

void init(execution_context *ctx, int device = -1, cudaStream_t stream = nullptr);
void clear(execution_context *ctx);

} // namespace cellerator::core::runtime
