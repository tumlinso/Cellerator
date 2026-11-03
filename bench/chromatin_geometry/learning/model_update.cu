#include "learning/model_update.hh"
#include <cuda_fp16.h>
#include <cmath>
#include <limits>
namespace cgs {
namespace {
__device__ float read_param(const void* p,DType type,std::uint64_t i){return type==DType::fp16?__half2float(static_cast<const __half*>(p)[i]):static_cast<const float*>(p)[i];}
__global__ void validate(const void* p,const float* g,std::uint64_t n,DType type,float rate,std::uint32_t* status){
 for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x){
  float x=read_param(p,type,i),v=x-rate*g[i];
  if(!isfinite(x)||!isfinite(g[i])||!isfinite(v)||(type==DType::fp16 && !isfinite(__half2float(__float2half_rn(v)))))atomicOr(status,status_nonfinite);
 }
}
__global__ void commit(void* p,const float* g,std::uint64_t n,DType type,float rate,const std::uint32_t* status){
 if(*status)return;
 for(auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x){
  float v=read_param(p,type,i)-rate*g[i];
  if(type==DType::fp16)static_cast<__half*>(p)[i]=__float2half_rn(v);else static_cast<float*>(p)[i]=v;
 }
}
}
cudaError_t model_sgd(void* p,const float* g,std::uint64_t n,DType type,float rate,std::uint32_t* status,cudaStream_t stream) noexcept {
 if((type!=DType::fp16&&type!=DType::fp32)||!std::isfinite(rate)||rate<0||n>std::numeric_limits<std::size_t>::max()/sizeof(float)||
    (n&&(!p||!g||!status)))return cudaErrorInvalidValue;
 if(!n)return cudaSuccess;
 auto blocks=static_cast<unsigned>(n/256+(n%256!=0)); if(blocks>65535)blocks=65535;
 validate<<<blocks,256,0,stream>>>(p,g,n,type,rate,status); auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
 commit<<<blocks,256,0,stream>>>(p,g,n,type,rate,status);return cudaGetLastError();
}
}
