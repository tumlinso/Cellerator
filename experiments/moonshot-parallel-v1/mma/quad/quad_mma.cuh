#pragma once
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace cellerator::experimental::moonshot {
// Borrowed contiguous row-major A[P,8,4], B[P,4,8], Y[P,8,8].
// FP16 stored operands, FP32 accumulation/output. No allocation or synchronization.
// Caller owns lifetimes and binds generations. Capacities count elements.
struct QuadMmaRequest {
    const __half* a; const __half* b; float* y;
    std::uint64_t a_elements,b_elements,y_elements;
    std::uint32_t panels;
};
// Pure host checks: capacity, alignment, index overflow, output aliasing.
cudaError_t validate_quad_mma(const QuadMmaRequest& request);
// Also checks CUDA architecture, grid, current-device allocation and stream device.
// Device-only buffers on current device; caller stream must belong to that device.
cudaError_t launch_quad_mma_checked(const QuadMmaRequest& request,cudaStream_t stream);
// Compatibility wrapper: caller asserts exact minimum buffer capacities.
cudaError_t launch_quad_mma(const __half* a,const __half* b,float* y,
                           std::uint32_t panels,cudaStream_t stream);
}
