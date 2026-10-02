#pragma once
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace cellerator::experimental::moonshot {
// Borrowed device arrays on caller_stream's device. No allocation or sync.
// Public wrappers only check basic arguments; the prepared adapter must validate
// capacities, device placement, supported architecture, aliases and epochs.
// A[panels,8,4], B[panels,4,8], Y[panels,8,8], all row-major.
cudaError_t launch_quad_mma(const __half* a,const __half* b,float* y,
                           std::uint32_t panels,cudaStream_t stream);
// Independent one-cell matrix patches. L,X,R[patches,16,16] -> Y[...].
// Numeric contract: T=fp32(L_half X_half), V=half(tanh(T)), Y=fp32(V R_half).
cudaError_t launch_patch16(const __half* l,const __half* x,const __half* r,
                          float* y,std::uint32_t patches,cudaStream_t stream);
// One operation per mechanism: k*x[src0]*x[src1], with one direction.
// Outputs are mechanism-local scratch, NOT already assembled biological output.
cudaError_t launch_product2_jvp(const float* x,const float* v,const float* k,
       const std::uint32_t* src0,const std::uint32_t* src1,
       float* product,float* jvp,std::uint32_t count,cudaStream_t stream);
} // namespace
