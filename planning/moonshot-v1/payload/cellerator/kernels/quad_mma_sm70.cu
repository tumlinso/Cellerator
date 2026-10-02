#include <cstddef>
#include <moonshot/kernels.cuh>
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
    const unsigned warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    const unsigned lane=threadIdx.x&31u;
    const unsigned group=(lane>>2)&3u;
    const unsigned panel=warp*4+group;
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
cudaError_t launch_quad_mma(const __half* a,const __half* b,float* y,
                           std::uint32_t panels,cudaStream_t stream) {
    if(!panels) return cudaSuccess;
    if(!a||!b||!y) return cudaErrorInvalidValue;
    // One block holds four warps -> sixteen independent products.
    const unsigned blocks=static_cast<unsigned>((std::uint64_t(panels)+15)/16);
    quad_kernel<<<blocks,128,0,stream>>>(a,b,y,panels);
    return cudaGetLastError();
}
}
