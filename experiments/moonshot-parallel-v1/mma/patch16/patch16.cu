#include "patch16.cuh"
#include <limits>
#include <initializer_list>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstddef>
namespace cellerator::experimental::moonshot {
namespace {
namespace wmma=nvcuda::wmma;
__global__ void patch16_kernel(const __half* l,const __half* x,const __half* r,
                              float* out,unsigned count) {
#if __CUDA_ARCH__ >= 700
    const unsigned local_warp=threadIdx.x>>5,lane=threadIdx.x&31u;
    const unsigned patch=blockIdx.x*4+local_warp;
    // Whole warp returns. No CTA barrier; each warp owns disjoint shared slices.
    if(patch>=count) return;
    __shared__ __align__(32) float intermediate[4][256];
    __shared__ __align__(32) __half hidden[4][256];
    const std::size_t base=std::size_t(patch)*256;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    wmma::fill_fragment(c,0.f);
    wmma::load_matrix_sync(a,l+base,16);
    wmma::load_matrix_sync(b,x+base,16);
    wmma::mma_sync(c,a,b,c);
    wmma::store_matrix_sync(intermediate[local_warp],c,16,wmma::mem_row_major);
    __syncwarp(0xffffffffu);
    for(unsigned i=lane;i<256;i+=32)
        hidden[local_warp][i]=__float2half_rn(tanhf(intermediate[local_warp][i]));
    __syncwarp(0xffffffffu);
    wmma::load_matrix_sync(a,hidden[local_warp],16);
    wmma::load_matrix_sync(b,r+base,16);
    wmma::fill_fragment(c,0.f);
    wmma::mma_sync(c,a,b,c);
    wmma::store_matrix_sync(out+base,c,16,wmma::mem_row_major);
#endif
}
}
cudaError_t launch_patch16(const __half* l,const __half* x,const __half* r,
                          float* y,std::uint32_t count,cudaStream_t stream) {
    if(!count) return cudaSuccess;
    if(!l||!x||!r||!y) return cudaErrorInvalidValue;
    // load/store_matrix_sync requires 32-byte alignment; every patch preserves it.
    auto aligned=[](const void* p){return (reinterpret_cast<std::uintptr_t>(p)&31u)==0;};
    if(!aligned(l)||!aligned(x)||!aligned(r)||!aligned(y)) return cudaErrorInvalidValue;
    patch16_kernel<<<static_cast<unsigned>((std::uint64_t(count)+3)/4),128,0,stream>>>(l,x,r,y,count);
    return cudaGetLastError();
}
}

namespace cellerator::experimental::moonshot::patch16 {
namespace {
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto aa = reinterpret_cast<std::uintptr_t>(a);
    const auto bb = reinterpret_cast<std::uintptr_t>(b);
    return aa < bb + bn && bb < aa + an;
}
bool valid_range(const void* p, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    return p && (address & 31u) == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}
}
cudaError_t validate(const Request& q) {
    if (q.device < 0) return cudaErrorInvalidDevice;
    // This artifact is deliberately compiled and qualified only for sm70.
    if (q.compute_major != 7 || q.compute_minor != 0) return cudaErrorNotSupported;
    if (!q.patches) return cudaSuccess;
    if (std::size_t(q.patches) > std::numeric_limits<std::size_t>::max() / 256)
        return cudaErrorInvalidValue;
    const auto n = std::size_t(q.patches) * 256;
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return cudaErrorInvalidValue;
    if (q.l_capacity < n || q.x_capacity < n || q.r_capacity < n || q.y_capacity < n)
        return cudaErrorInvalidValue;
    const auto input_bytes = n * sizeof(__half), output_bytes = n * sizeof(float);
    if (!valid_range(q.l, input_bytes) || !valid_range(q.x, input_bytes) ||
        !valid_range(q.r, input_bytes) || !valid_range(q.y, output_bytes))
        return cudaErrorInvalidValue;
    // Read-only inputs may overlap each other; output cannot overlap any input.
    if (overlap(q.y, output_bytes, q.l, input_bytes) ||
        overlap(q.y, output_bytes, q.x, input_bytes) ||
        overlap(q.y, output_bytes, q.r, input_bytes)) return cudaErrorInvalidValue;
    return cudaSuccess;
}
cudaError_t launch_prepared(const Request& q, cudaStream_t stream) {
    auto error = validate(q);
    if (error != cudaSuccess || !q.patches) return error;
    int current = -1;
    if ((error = cudaGetDevice(&current)) != cudaSuccess) return error;
    if (current != q.device) return cudaErrorInvalidDevice;
    cudaDeviceProp properties{};
    if ((error = cudaGetDeviceProperties(&properties, current)) != cudaSuccess) return error;
    if (properties.major != q.compute_major || properties.minor != q.compute_minor)
        return cudaErrorNotSupported;
    int stream_device = -1;
    if ((error = cudaStreamGetDevice(stream, &stream_device)) != cudaSuccess) return error;
    if (stream_device != current) return cudaErrorInvalidDevice;
    for (const void* pointer : {static_cast<const void*>(q.l), static_cast<const void*>(q.x),
                              static_cast<const void*>(q.r), static_cast<const void*>(q.y)}) {
        cudaPointerAttributes attributes{};
        if ((error = cudaPointerGetAttributes(&attributes, pointer)) != cudaSuccess) return error;
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != current)
            return cudaErrorInvalidDevicePointer;
    }
    return launch_patch16(q.l, q.x, q.r, q.y, q.patches, stream);
}
}
