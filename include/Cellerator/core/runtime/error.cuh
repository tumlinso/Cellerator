#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cellerator::core::runtime {

inline void cuda_require(cudaError_t status, const char *label) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
}

} // namespace cellerator::core::runtime
