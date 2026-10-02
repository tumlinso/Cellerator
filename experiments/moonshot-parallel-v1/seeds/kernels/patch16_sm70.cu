#include <moonshot/kernels.cuh>
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
