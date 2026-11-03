#pragma once
#include "contract.hh"
namespace cgs {
// In-place SGD; grad is canonical FP32. Validation occurs before commit and a
// nonzero shared status suppresses all writes. FP16 parameters round once at
// storage after FP32 multiply/subtract. Caller invalidates value revisions.
cudaError_t model_sgd(void* parameters, const float* gradient, std::uint64_t count,
                      DType dtype, float learning_rate, std::uint32_t* status,
                      cudaStream_t stream) noexcept;
}
