#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
namespace {
// One physical slot per thread. The prepared value storage is authoritative;
// an optional f16 projection is explicitly derived from each updated f32 value.
// No hidden master plane, atomics, topology walk or synchronization.
__device__ void store_value(__half& slot,float value){slot=__float2half_rn(value);}
__device__ void store_value(float& slot,float value){slot=value;}
template<class Value>
__global__ void update_kernel(Value* values,const float* operand,std::uint32_t count,
    bool gradient_step,float alpha,__half* derived=nullptr) {
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const float current=static_cast<float>(values[i]);
    const float next=gradient_step?fmaf(-alpha,operand[i],current):__fadd_rn(current,operand[i]);
    store_value(values[i],next);if(derived)derived[i]=__float2half_rn(next);
}
}
cudaError_t enqueue_relation_value_update(void* values,const float* operand,
    std::uint32_t count,bool gradient_step,float alpha,cudaStream_t stream) noexcept {
    if(!count)return cudaSuccess;
    update_kernel<<<(std::uint64_t(count)+255)/256,256,0,stream>>>(
        static_cast<__half*>(values),operand,count,gradient_step,alpha);
    return cudaPeekAtLastError();
}
cudaError_t enqueue_relation_value_update_f32(float* values,void* derived,const float* operand,
    std::uint32_t count,bool gradient_step,float alpha,cudaStream_t stream) noexcept {
    if(!count)return cudaSuccess;
    update_kernel<<<(std::uint64_t(count)+255)/256,256,0,stream>>>(
        values,operand,count,gradient_step,alpha,static_cast<__half*>(derived));
    return cudaPeekAtLastError();
}

}
