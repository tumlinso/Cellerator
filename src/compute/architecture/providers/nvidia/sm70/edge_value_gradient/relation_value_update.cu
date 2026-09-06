#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
namespace {
// One physical slot per thread; f16 load/store plus f32 delta/gradient traffic
// dominates. No master value plane, atomics, topology walk or synchronization.
__global__ void update_kernel(__half* values,const float* operand,std::uint32_t count,
    bool gradient_step,float alpha) {
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const float current=__half2float(values[i]);
    const float next=gradient_step?fmaf(-alpha,operand[i],current):__fadd_rn(current,operand[i]);
    values[i]=__float2half_rn(next);
}
}
cudaError_t enqueue_relation_value_update(void* values,const float* operand,
    std::uint32_t count,bool gradient_step,float alpha,cudaStream_t stream) noexcept {
    if(!count)return cudaSuccess;
    update_kernel<<<(std::uint64_t(count)+255)/256,256,0,stream>>>(
        static_cast<__half*>(values),operand,count,gradient_step,alpha);
    return cudaPeekAtLastError();
}
}
