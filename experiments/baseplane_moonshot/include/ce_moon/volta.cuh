#pragma once
// Experimental sm_70 numerical kernels. Block width must be a multiple of 32.
// Caller owns nonaliasing storage, validated domains/capacities and stream.
// Linear indices and 32*regions must remain below 2^31. No allocation or sync.
// Counted composition requires a caller proof of no uint64 overflow.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstddef>
namespace ce_moon_cuda {
constexpr unsigned full=0xffffffffu;
using u64=unsigned long long;
struct MonoLane {unsigned p;float d,b;};
// Layout: 32 adjacent unsigned states per region. Every entry is in [0,32).
// lane s owns T(s). Composition is R(L(s)), one shuffle per result field.
static __global__ void dfa_compose(const unsigned* left,const unsigned* right,unsigned* out,unsigned regions){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid&31,region=tid>>5;
    if(region>=regions)return; // uniform return for the entire warp
    unsigned l=left[region*32+lane],r=right[region*32+lane];
    out[region*32+lane]=__shfl_sync(full,r,l);
}
static __global__ void counted_dfa_compose(const unsigned* left,const unsigned* right,
    const u64* left_count,const u64* right_count,unsigned* out,u64* count,unsigned regions){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid&31,region=tid>>5;
    if(region>=regions)return;
    unsigned i=region*32+lane,l=left[i],r=right[i];u64 lc=left_count[i],rc=right_count[i];
    out[i]=__shfl_sync(full,r,l);count[i]=lc+__shfl_sync(full,rc,l);
    // Preparation must prove no uint64 overflow. Counts do NOT preserve event positions.
}
// h_out[i] = d[i] * h_in[p[i]] + b[i]; p is a validated permutation.
static __global__ void monomial_compose(const MonoLane* left,const MonoLane* right,MonoLane* out,unsigned regions){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid&31,region=tid>>5;
    if(region>=regions)return;
    unsigned i=region*32+lane;MonoLane l=left[i],r=right[i];
    unsigned p=__shfl_sync(full,l.p,r.p);
    float d=__shfl_sync(full,l.d,r.p),b=__shfl_sync(full,l.b,r.p);
    out[i]={p,r.d*d,r.d*b+r.b}; // Mathematical closure; floating evaluation is approximate.
}
static __global__ void lift_pairs(const float* fine,float* coarse,float* detail,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=(n+1u)/2u)return;
    const float even=fine[2*i],odd=2*i+1<n?fine[2*i+1]:even;
    float d=odd-even;coarse[i]=even+.5f*d;detail[i]=d;
}
// One full warp per 16x16 tile. Both multiplicands are row-major FP16; accumulator
// FP32. Pointers must be 32-byte aligned; stride=16; zero-pad unused rows/columns.
// Fragments stay opaque. This uses ordinary arithmetic, NOT a Boolean MMA.
static __global__ void wmma16(const __half* a,const __half* b,float* c,unsigned tiles){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,tile=tid>>5;
    if(tile>=tiles)return;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> fb;
    wmma::fragment<wmma::accumulator,16,16,16,float> fc;
    wmma::load_matrix_sync(fa,a+std::size_t(tile)*256,16);
    wmma::load_matrix_sync(fb,b+std::size_t(tile)*256,16);
    wmma::fill_fragment(fc,0.f);wmma::mma_sync(fc,fa,fb,fc);
    wmma::store_matrix_sync(c+std::size_t(tile)*256,fc,16,wmma::mem_row_major);
#endif
}
static __global__ void positive_relation(const float* count,unsigned char* relation,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)relation[i]=count[i]>0.f;
}
static __global__ void packed_dot(const int* a,const int* b,int* out,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__dp4a(a[i],b[i],0);
}
static __global__ void butterfly32(const float* input,float* output,unsigned regions){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid&31,region=tid>>5;
    if(region>=regions)return;
    float x=input[region*32+lane];
    for(unsigned step=1;step<32;step<<=1){float peer=__shfl_xor_sync(full,x,step);x=(lane&step)?peer-x:x+peer;}
    output[region*32+lane]=x;
}
// Experimental floating lookup only. Caller configures the texture explicitly.
// Linear interpolation has finite coordinate precision; it is not an exact gate.
static __global__ void texture_response(cudaTextureObject_t table,const float2* coordinates,float* out,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float2 p=coordinates[i];out[i]=tex2D<float>(table,p.x,p.y);}
}
} // namespace ce_moon_cuda
