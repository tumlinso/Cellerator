#include <cstddef>
#include <moonshot/kernels.cuh>
#include <cuda_runtime.h>
namespace cellerator::experimental::moonshot {
namespace {
__device__ __forceinline__ float shared_read(const float* x,unsigned id) {
    constexpr unsigned mask=0xffffffffu;
    const unsigned peers=__match_any_sync(mask,id);
    const int leader=__ffs(peers)-1;
    const unsigned lane=threadIdx.x&31u;
    float value=0.f;
    if(lane==unsigned(leader) && id!=0xffffffffu) value=x[id];
    return __shfl_sync(mask,value,leader);
}
__global__ void product2_kernel(const float* x,const float* v,const float* k,
        const unsigned* s0,const unsigned* s1,float* y,float* dy,unsigned count) {
    unsigned idx=blockIdx.x*blockDim.x+threadIdx.x;
    const bool valid=idx<count;
    const unsigned a=valid?s0[idx]:0xffffffffu,b=valid?s1[idx]:0xffffffffu;
    // All 32 lanes participate, including padded lanes. Inputs read one snapshot.
    float xa=shared_read(x,a),xb=shared_read(x,b);
    float va=shared_read(v,a),vb=shared_read(v,b);
    if(valid) {
        y[idx]=k[idx]*xa*xb;
        dy[idx]=k[idx]*fmaf(va,xb,xa*vb);
    }
}
}
cudaError_t launch_product2_jvp(const float* x,const float* v,const float* k,
        const std::uint32_t* s0,const std::uint32_t* s1,float* y,float* dy,
        std::uint32_t count,cudaStream_t stream) {
    if(!count) return cudaSuccess;
    if(!x||!v||!k||!s0||!s1||!y||!dy) return cudaErrorInvalidValue;
    product2_kernel<<<static_cast<unsigned>((std::uint64_t(count)+127)/128),128,0,stream>>>(x,v,k,s0,s1,y,dy,count);
    return cudaGetLastError();
}
}
