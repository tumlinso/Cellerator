#include "product.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <new>
namespace cellerator::experimental::moonshot::product {
namespace {
struct Span { std::uintptr_t begin{},end{}; };
template<class T> bool span(View<T> v,Span& s) {
    s.begin=reinterpret_cast<std::uintptr_t>(v.data);
    if(v.size && (!v.data || s.begin%alignof(T))) return false;
    if(v.size>(UINTPTR_MAX-s.begin)/sizeof(T)) return false;
    s.end=s.begin+v.size*sizeof(T); return true;
}
bool overlap(Span a,Span b) { return a.begin<a.end && b.begin<b.end && a.begin<b.end && b.begin<a.end; }
template<class T> cudaError_t check_device(View<T> v,int device) {
    if(!v.size) return cudaSuccess;
    cudaPointerAttributes attr{};
    auto e=cudaPointerGetAttributes(&attr,v.data);
    if(e!=cudaSuccess) return e;
    if(attr.type!=cudaMemoryTypeDevice || attr.device!=device) return cudaErrorInvalidDevicePointer;
    CUdeviceptr base{}; std::size_t bytes{};
    const auto ptr=reinterpret_cast<CUdeviceptr>(v.data);
    if(cuMemGetAddressRange(&base,&bytes,ptr)!=CUDA_SUCCESS || ptr<base ||
       ptr-base>bytes || v.size>(bytes-(ptr-base))/sizeof(T))
        return cudaErrorInvalidDevicePointer;
    return cudaSuccess;
}
__device__ __forceinline__ float shared_read(const float* x,unsigned id) {
    const unsigned peers=__match_any_sync(0xffffffffu,id);
    const int leader=__ffs(peers)-1;
    float value=0.f;
    if((threadIdx.x&31u)==unsigned(leader) && id!=padding_id) value=x[id];
    return __shfl_sync(0xffffffffu,value,leader);
}
__global__ void kernel(Inputs in,Outputs out) {
    const std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const bool valid=i<in.count;
    unsigned a=valid?in.src0.data[i]:padding_id,b=valid?in.src1.data[i]:padding_id;
    const float xa=shared_read(in.x.data,a),xb=shared_read(in.x.data,b);
    const float va=shared_read(in.direction.data,a),vb=shared_read(in.direction.data,b);
    if(valid) {
        out.value.data[i]=(in.coefficient.data[i]*xa)*xb;
        out.jvp.data[i]=in.coefficient.data[i]*fmaf(va,xb,xa*vb);
    }
}
}
cudaError_t validate_metadata(const Inputs& in,const Outputs& out) {
    if(in.coefficient.size<in.count || in.src0.size<in.count || in.src1.size<in.count ||
       out.value.size<in.count || out.jvp.size<in.count || in.x.size!=in.direction.size ||
       (in.count && !in.x.size)) return cudaErrorInvalidValue;
    Span reads[5],writes[2];
    if(!span(in.x,reads[0]) || !span(in.direction,reads[1]) || !span(in.coefficient,reads[2]) ||
       !span(in.src0,reads[3]) || !span(in.src1,reads[4]) || !span(out.value,writes[0]) || !span(out.jvp,writes[1]))
       return cudaErrorInvalidValue;
    if(overlap(writes[0],writes[1])) return cudaErrorInvalidValue;
    for(auto w:writes) for(auto r:reads) if(overlap(w,r)) return cudaErrorInvalidValue;
    return cudaSuccess;
}
cudaError_t prepare_product2(const Inputs& in,const Outputs& out,Generations gen,cudaStream_t stream,Prepared& p) {
    p=Prepared{};
    auto e=validate_metadata(in,out); if(e!=cudaSuccess) return e;
    // Empty work never touches CUDA, accepts a fully empty descriptor.
    if(!in.count) { p.inputs_=in;p.outputs_=out;p.generations_=gen;p.stream_=stream;p.ready_=true;return cudaSuccess; }
    int device=-1,stream_device=-1;
    if((e=cudaGetDevice(&device))!=cudaSuccess) return e;
    if((e=cudaStreamGetDevice(stream,&stream_device))!=cudaSuccess) return e;
    if(device!=stream_device) return cudaErrorInvalidResourceHandle;
    cudaDeviceProp prop{};
    if((e=cudaGetDeviceProperties(&prop,device))!=cudaSuccess) return e;
    if(prop.major!=7 || prop.minor!=0) return cudaErrorNotSupported;
    if((e=check_device(in.x,device))!=cudaSuccess || (e=check_device(in.direction,device))!=cudaSuccess ||
       (e=check_device(in.coefficient,device))!=cudaSuccess || (e=check_device(in.src0,device))!=cudaSuccess ||
       (e=check_device(in.src1,device))!=cudaSuccess || (e=check_device(out.value,device))!=cudaSuccess ||
       (e=check_device(out.jvp,device))!=cudaSuccess) return e;
    try {
        std::vector<std::uint32_t> a(in.count),b(in.count);
        if((e=cudaMemcpyAsync(a.data(),in.src0.data,in.count*sizeof(std::uint32_t),cudaMemcpyDeviceToHost,stream))!=cudaSuccess) return e;
        if((e=cudaMemcpyAsync(b.data(),in.src1.data,in.count*sizeof(std::uint32_t),cudaMemcpyDeviceToHost,stream))!=cudaSuccess) {
            cudaStreamSynchronize(stream); return e;
        }
        if((e=cudaStreamSynchronize(stream))!=cudaSuccess) return e;
        for(std::size_t i=0;i<in.count;++i)
            if(a[i]==padding_id || b[i]==padding_id || a[i]>=in.x.size || b[i]>=in.x.size) return cudaErrorInvalidValue;
    } catch(const std::bad_alloc&) { return cudaErrorMemoryAllocation; }
    p.inputs_=in;p.outputs_=out;p.generations_=gen;p.device_=device;p.stream_=stream;p.ready_=true;
    return cudaSuccess;
}
cudaError_t launch_product2(const Prepared& p,Generations gen,cudaStream_t stream) {
    if(!p.ready_ || !(gen==p.generations_) || stream!=p.stream_) return cudaErrorInvalidValue;
    if(!p.inputs_.count) return cudaSuccess;
    int device=-1; auto e=cudaGetDevice(&device); if(e!=cudaSuccess) return e;
    if(device!=p.device_) return cudaErrorInvalidDevice;
    kernel<<<static_cast<unsigned>((std::uint64_t(p.inputs_.count)+127)/128),128,0,stream>>>(p.inputs_,p.outputs_);
    return cudaGetLastError();
}
}
