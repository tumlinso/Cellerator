#include <cstddef>
#include "quad_mma.cuh"
#include <limits>
#include <initializer_list>
#include <cuda_runtime.h>
namespace cellerator::experimental::moonshot {
namespace {
__device__ __forceinline__ unsigned pack_half(__half a,__half b) {
    return unsigned(__half_as_ushort(a)) | (unsigned(__half_as_ushort(b))<<16);
}
// Four INDEPENDENT 8x4 by 4x8 jobs. Every lane executes the same PTX instruction.
// The documented PTX register mapping below is NOT the opaque C++ WMMA mapping.
__global__ void quad_kernel(const __half* a,const __half* b,float* y,unsigned panels) {
#if __CUDA_ARCH__ >= 700
    const std::uint64_t warp=(std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)>>5;
    const unsigned lane=threadIdx.x&31u;
    const unsigned group=(lane>>2)&3u;
    const std::uint64_t panel=warp*4+group;
    const unsigned rc=(lane&3u)+(lane>=16u?4u:0u);
    __half aa[4],bb[4];
    #pragma unroll
    for(int k=0;k<4;++k) {
        aa[k]=panel<panels ? a[std::size_t(panel)*32+rc*4+k] : __float2half(0.f);
        bb[k]=panel<panels ? b[std::size_t(panel)*32+k*8+rc] : __float2half(0.f);
    }
    const unsigned a0=pack_half(aa[0],aa[1]),a1=pack_half(aa[2],aa[3]);
    const unsigned b0=pack_half(bb[0],bb[1]),b1=pack_half(bb[2],bb[3]);
    float d[8]={0,0,0,0,0,0,0,0};
    asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),
        "+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
      : "r"(a0),"r"(a1),"r"(b0),"r"(b1));
    if(panel<panels) {
        #pragma unroll
        for(unsigned i=0;i<8;++i) {
            const unsigned row=(lane&1u)+(i&2u)+(lane>=16u?4u:0u);
            const unsigned col=(i&4u)+(lane&2u)+(i&1u);
            y[std::size_t(panel)*64+row*8+col]=d[i];
        }
    }
#endif
}
}
cudaError_t validate_quad_mma(const QuadMmaRequest& q) {
    if (!q.panels) return cudaSuccess;
    const std::uint64_t ab=std::uint64_t(q.panels)*32*sizeof(__half);
    const std::uint64_t yy=std::uint64_t(q.panels)*64*sizeof(float);
    if (ab>std::numeric_limits<std::size_t>::max() || yy>std::numeric_limits<std::size_t>::max())
        return cudaErrorInvalidValue;
    if (!q.a || !q.b || !q.y || q.a_elements<ab/sizeof(__half) ||
        q.b_elements<ab/sizeof(__half) || q.y_elements<yy/sizeof(float)) return cudaErrorInvalidValue;
    const auto a=reinterpret_cast<std::uintptr_t>(q.a), b=reinterpret_cast<std::uintptr_t>(q.b),
               y=reinterpret_cast<std::uintptr_t>(q.y);
    if (a%alignof(__half) || b%alignof(__half) || y%alignof(float) ||
        a>UINTPTR_MAX-ab || b>UINTPTR_MAX-ab || y>UINTPTR_MAX-yy) return cudaErrorInvalidValue;
    if ((y<a+ab && a<y+yy) || (y<b+ab && b<y+yy)) return cudaErrorInvalidValue;
    return cudaSuccess;
}
cudaError_t launch_quad_mma_checked(const QuadMmaRequest& q,cudaStream_t stream) {
    auto status=validate_quad_mma(q);
    if(status!=cudaSuccess || !q.panels) return status;
    int device=0; cudaDeviceProp prop{};
    if((status=cudaGetDevice(&device))!=cudaSuccess) return status;
    if((status=cudaGetDeviceProperties(&prop,device))!=cudaSuccess) return status;
    if(prop.major<7) return cudaErrorNotSupported;
    const std::uint64_t blocks=(std::uint64_t(q.panels)+15)/16;
    if(blocks>static_cast<std::uint64_t>(prop.maxGridSize[0])) return cudaErrorInvalidConfiguration;
    for(const void* pointer : {static_cast<const void*>(q.a),static_cast<const void*>(q.b),static_cast<const void*>(q.y)}) {
        cudaPointerAttributes attr{};
        if((status=cudaPointerGetAttributes(&attr,pointer))!=cudaSuccess) return status;
        if(attr.type!=cudaMemoryTypeDevice || attr.device!=device) return cudaErrorInvalidDevicePointer;
    }
    int stream_device=0;
    if((status=cudaStreamGetDevice(stream,&stream_device))!=cudaSuccess) return status;
    if(stream_device!=device) return cudaErrorInvalidResourceHandle;
    unsigned flags=0;
    if((status=cudaStreamGetFlags(stream,&flags))!=cudaSuccess) return status;
    quad_kernel<<<static_cast<unsigned>(blocks),128,0,stream>>>(q.a,q.b,q.y,q.panels);
    return cudaGetLastError();
}
cudaError_t launch_quad_mma(const __half* a,const __half* b,float* y,
                           std::uint32_t panels,cudaStream_t stream) {
    // Compatibility route: capacities are caller assertions. Prefer checked request.
    return launch_quad_mma_checked({a,b,y,std::uint64_t(panels)*32,
        std::uint64_t(panels)*32,std::uint64_t(panels)*64,panels},stream);
}
}
