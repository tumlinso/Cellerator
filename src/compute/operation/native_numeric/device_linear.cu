#include <Cellerator/compute/operation/native_numeric/device_linear.hh>
#include <limits>
#include <cstdint>
namespace cellerator::compute::native_numeric { namespace {
std::size_t bytes(const resident_vector& v){return v.elements*(v.representation==device_representation::f16?2:4);}
constexpr unsigned arithmetic_threads=256;
constexpr unsigned arithmetic_max_blocks=65535;
__global__ void axpby(float* o,const float* x,const float* y,std::uint64_t n,float a,float b){auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;auto step=std::uint64_t(gridDim.x)*blockDim.x;for(;i<n;i+=step)o[i]=a*x[i]+b*y[i];}
__global__ void weighted_sum4(float* o,const float* base,const float* k1,const float* k2,const float* k3,const float* k4,std::uint64_t n,float h){auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)o[i]=base[i]+h*(k1[i]+2.f*k2[i]+2.f*k3[i]+k4[i])/6.f;}
execution::program::program_status admit(const void* state,const execution::program::launch_binding_v2& b,void*) noexcept {auto* s=static_cast<const linear_stage*>(state);if(!s||!b.input||!b.output)return execution::program::program_status::invalid_argument;if(s->kind==linear_kind::copy)return execution::program::program_status::success;if(s->representation!=device_representation::f32)return execution::program::program_status::invalid_argument;if(s->kind==linear_kind::axpby)return b.values?execution::program::program_status::success:execution::program::program_status::invalid_argument;return b.workspace&&b.workspace_bytes>=5*s->elements*sizeof(float)?execution::program::program_status::success:execution::program::program_status::invalid_argument;}
execution::program::program_status launch(const void* state,const execution::program::launch_binding_v2& b,void* sp) noexcept {if(admit(state,b,sp)!=execution::program::program_status::success)return execution::program::program_status::invalid_argument;auto* s=static_cast<const linear_stage*>(state);auto stream=static_cast<cudaStream_t>(sp);if(s->kind==linear_kind::copy){auto bytes=s->elements*(s->representation==device_representation::f16?2:4);if(cudaMemcpyAsync(b.output,b.input,bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return execution::program::program_status::launch_failed;return execution::program::program_status::success;}auto blocks=unsigned((s->elements+255)/256);if(s->kind==linear_kind::axpby){axpby<<<blocks,256,0,stream>>>(static_cast<float*>(b.output),static_cast<const float*>(b.input),static_cast<const float*>(b.values),s->elements,s->alpha,s->beta);}else {auto* p=static_cast<const float*>(b.workspace);weighted_sum4<<<blocks,256,0,stream>>>(static_cast<float*>(b.output),p,p+s->elements,p+2*s->elements,p+3*s->elements,p+4*s->elements,s->elements,s->alpha);}return cudaGetLastError()==cudaSuccess?execution::program::program_status::success:execution::program::program_status::launch_failed;}
}
cudaError_t allocate(resident_vector* v,std::uint64_t n,device_representation r,int d) noexcept {if(!v||v->data||!n)return cudaErrorInvalidValue;v->elements=n;v->representation=r;v->device_ordinal=d;if(cudaSetDevice(d)!=cudaSuccess)return cudaErrorInvalidDevice;return cudaMalloc(&v->data,bytes(*v));}
cudaError_t release(resident_vector* v) noexcept {if(!v)return cudaErrorInvalidValue;auto e=v->data?cudaFree(v->data):cudaSuccess;*v={};return e;}
cudaError_t upload(resident_vector&v,const void*h,std::uint64_t n,execution::value_generation g,cudaStream_t s) noexcept {if(!h||n!=v.elements||!g.value)return cudaErrorInvalidValue;auto e=cudaMemcpyAsync(v.data,h,bytes(v),cudaMemcpyHostToDevice,s);if(e==cudaSuccess)v.generation=g;return e;}
cudaError_t download(const resident_vector&v,void*h,std::uint64_t n,cudaStream_t s) noexcept{return !h||n!=v.elements?cudaErrorInvalidValue:cudaMemcpyAsync(h,v.data,bytes(v),cudaMemcpyDeviceToHost,s);}
cudaError_t reset(resident_vector&v,float x,cudaStream_t s) noexcept{return x!=0?cudaErrorNotSupported:cudaMemsetAsync(v.data,0,bytes(v),s);}
execution::program::prepared_stage_v2 make_linear_stage(std::uint64_t id,std::uint64_t c,const linear_stage*s) noexcept{return{id,c,s,launch,0,0,0,0,admit};}
}

namespace cellerator::compute::native_numeric { namespace {
__global__ void elementwise_multiply_kernel(float* output,const float* left,
                                            const float* right,std::uint64_t n) {
    auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    auto step=std::uint64_t(gridDim.x)*blockDim.x;
    for(;i<n;i+=step) output[i]=left[i]*right[i];
}

bool arithmetic_ranges_overlap(const void* left,const void* right,
                               std::uint64_t bytes) noexcept {
    const auto a=reinterpret_cast<std::uintptr_t>(left);
    const auto b=reinterpret_cast<std::uintptr_t>(right);
    return a<b+bytes && b<a+bytes;
}

cudaError_t validate_binary_arithmetic(const resident_vector& left,
        const resident_vector& right,resident_vector& output,cudaStream_t stream) noexcept {
    if(left.elements!=right.elements || left.elements!=output.elements ||
       left.representation!=device_representation::f32 ||
       right.representation!=device_representation::f32 ||
       output.representation!=device_representation::f32)
        return cudaErrorInvalidValue;
    int active_device=-1;
    auto error=cudaGetDevice(&active_device);
    if(error!=cudaSuccess) return error;
    if(left.device_ordinal!=active_device || right.device_ordinal!=active_device ||
       output.device_ordinal!=active_device) return cudaErrorInvalidDevice;
    unsigned flags=0;
    error=cudaStreamGetFlags(stream,&flags);
    if(error!=cudaSuccess) return error;
    int stream_device=-1;
    error=cudaStreamGetDevice(stream,&stream_device);
    if(error!=cudaSuccess) return error;
    if(stream_device!=active_device) return cudaErrorInvalidDevice;
    if(left.elements==0) return cudaSuccess;
    if(left.elements>std::numeric_limits<std::uint64_t>::max()/sizeof(float) ||
       left.elements>std::numeric_limits<std::uintptr_t>::max()/sizeof(float))
        return cudaErrorInvalidValue;
    const auto bytes=left.elements*sizeof(float);
    const resident_vector* vectors[]={&left,&right,&output};
    for(const auto* vector:vectors) {
        if(!vector->data) return cudaErrorInvalidDevicePointer;
        const auto address=reinterpret_cast<std::uintptr_t>(vector->data);
        if(address%alignof(float) || address>std::numeric_limits<std::uintptr_t>::max()-bytes)
            return cudaErrorInvalidDevicePointer;
    }
    if(arithmetic_ranges_overlap(output.data,left.data,bytes) ||
       arithmetic_ranges_overlap(output.data,right.data,bytes)) return cudaErrorInvalidValue;
    for(const auto* vector:vectors) {
        cudaPointerAttributes attributes{};
        error=cudaPointerGetAttributes(&attributes,vector->data);
        if(error!=cudaSuccess) return error;
        if((attributes.type!=cudaMemoryTypeDevice && attributes.type!=cudaMemoryTypeManaged) ||
           attributes.device!=active_device) return cudaErrorInvalidDevicePointer;
    }
    return cudaSuccess;
}

unsigned arithmetic_grid(std::uint64_t n) noexcept {
    const auto blocks=n/arithmetic_threads+(n%arithmetic_threads!=0);
    return static_cast<unsigned>(blocks<arithmetic_max_blocks?blocks:arithmetic_max_blocks);
}
}

cudaError_t enqueue_elementwise_multiply(const resident_vector& left,
        const resident_vector& right,resident_vector& output,cudaStream_t stream) noexcept {
    auto error=validate_binary_arithmetic(left,right,output,stream);
    if(error!=cudaSuccess || left.elements==0) return error;
    elementwise_multiply_kernel<<<arithmetic_grid(left.elements),arithmetic_threads,0,stream>>>(
        static_cast<float*>(output.data),static_cast<const float*>(left.data),
        static_cast<const float*>(right.data),left.elements);
    return cudaPeekAtLastError();
}

cudaError_t enqueue_axpby(float alpha,const resident_vector& x,float beta,
        const resident_vector& y,resident_vector& output,cudaStream_t stream) noexcept {
    auto error=validate_binary_arithmetic(x,y,output,stream);
    if(error!=cudaSuccess || x.elements==0) return error;
    axpby<<<arithmetic_grid(x.elements),arithmetic_threads,0,stream>>>(
        static_cast<float*>(output.data),static_cast<const float*>(x.data),
        static_cast<const float*>(y.data),x.elements,alpha,beta);
    return cudaPeekAtLastError();
}
}
