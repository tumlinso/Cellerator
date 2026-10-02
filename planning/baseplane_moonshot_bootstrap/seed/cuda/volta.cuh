#pragma once
// sm_70 research seeds. Authored, NOT compiled or GPU-tested in this package's
// creation environment. See docs/03_VOLTA_PALETTE.md and evidence/validation.json.
// All kernels: blockDim.x multiple of 32; caller-owned nonaliasing storage and
// stream. No allocation, device selection, host sync, or unbounded spin queues.
// Fixture domain: all launched linear indices and 32*regions stay below 2^31.
// Host preparation must validate capacities, input state/permutation domains and
// nonaliasing. A generalized launch API is intentionally not claimed here.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstddef>
namespace bp_moon_cuda {
constexpr unsigned full=0xffffffffu;
using u64=unsigned long long;
struct MonoLane {unsigned p;float d,b;};
struct EmitCounts {u64 produced,stored,dropped;}; // required_capacity = produced

template<unsigned LUT> __device__ __forceinline__ unsigned lop3(unsigned a,unsigned b,unsigned c){
    static_assert(LUT<256,"eight-bit truth table");unsigned out;
    asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(out) : "r"(a),"r"(b),"r"(c),"n"(LUT));
    return out;
}
__global__ void predicate_circuit(const unsigned* a,const unsigned* b,const unsigned* c,
                                  const unsigned* valid,unsigned* out,std::size_t words){
    std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<words)out[i]=lop3<0x96>(a[i],b[i],c[i])&valid[i];
}
// Layout: 32 adjacent unsigned states per region. Every entry is in [0,32).
// lane s owns T(s). Composition is R(L(s)), one shuffle per result field.
__global__ void dfa_compose(const unsigned* left,const unsigned* right,unsigned* out,unsigned regions){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid&31,region=tid>>5;
    if(region>=regions)return; // uniform return for the entire warp
    unsigned l=left[region*32+lane],r=right[region*32+lane];
    out[region*32+lane]=__shfl_sync(full,r,l);
}
__global__ void counted_dfa_compose(const unsigned* left,const unsigned* right,
    const u64* left_count,const u64* right_count,unsigned* out,u64* count,unsigned regions){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid&31,region=tid>>5;
    if(region>=regions)return;
    unsigned i=region*32+lane,l=left[i],r=right[i];u64 lc=left_count[i],rc=right_count[i];
    out[i]=__shfl_sync(full,r,l);count[i]=lc+__shfl_sync(full,rc,l);
    // Preparation must prove no uint64 overflow. Counts do NOT preserve event positions.
}
// h_out[i] = d[i] * h_in[p[i]] + b[i]; p is a validated permutation.
__global__ void monomial_compose(const MonoLane* left,const MonoLane* right,MonoLane* out,unsigned regions){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid&31,region=tid>>5;
    if(region>=regions)return;
    unsigned i=region*32+lane;MonoLane l=left[i],r=right[i];
    unsigned p=__shfl_sync(full,l.p,r.p);
    float d=__shfl_sync(full,l.d,r.p),b=__shfl_sync(full,l.b,r.p);
    out[i]={p,r.d*d,r.d*b+r.b}; // Mathematical closure; floating evaluation is approximate.
}
__global__ void lift_pairs(const float* fine,float* coarse,float* detail,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=(n+1u)/2u)return;
    const float even=fine[2*i],odd=2*i+1<n?fine[2*i+1]:even;
    float d=odd-even;coarse[i]=even+.5f*d;detail[i]=d;
}
// Sparse output order is unspecified between warps. No consumer reads the
// reservation counter until this producer kernel has completed on its stream.
__global__ void emit_selected(const unsigned char* selected,unsigned n,unsigned* ids,
                              u64 capacity,EmitCounts* counts){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x&31;
    bool yes=i<n&&selected[i]!=0;unsigned mask=__ballot_sync(full,yes),k=__popc(mask);
    u64 base=0;
    if(lane==0&&k)base=atomicAdd(&counts->produced,u64(k));
    base=__shfl_sync(full,base,0);
    const unsigned rank=__popc(mask&((1u<<lane)-1u));
    if(yes && base<capacity && u64(rank)<capacity-base)ids[base+rank]=i;
    if(lane==0&&k){const u64 room=base<capacity?capacity-base:0;
        const u64 stored=room<u64(k)?room:u64(k);
        atomicAdd(&counts->stored,stored);atomicAdd(&counts->dropped,u64(k)-stored);}
}
// Only local peers. These masks must NEVER be labelled a global candidate directory.
__global__ void local_equal_peers(const unsigned* keys,unsigned* peers,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned members=__ballot_sync(full,i<n);
    if(i<n)peers[i]=__match_any_sync(members,keys[i]);
}
__global__ void bitplane_count4(const unsigned* supports,unsigned* planes,unsigned* overflow,
                              unsigned words,unsigned predicates){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=words)return;
    unsigned b0=0,b1=0,b2=0,b3=0,ov=0;
    for(unsigned p=0;p<predicates;++p){unsigned x=supports[std::size_t(p)*words+i],c=b0&x;b0^=x;
        x=c;c=b1&x;b1^=x;x=c;c=b2&x;b2^=x;x=c;c=b3&x;b3^=x;ov|=c;}
    planes[i]=b0;planes[std::size_t(words)+i]=b1;planes[std::size_t(2)*words+i]=b2;
    planes[std::size_t(3)*words+i]=b3;overflow[i]=ov;
}
// One full warp per 16x16 tile. Both multiplicands are row-major FP16; accumulator
// FP32. Pointers must be 32-byte aligned; stride=16; zero-pad unused rows/columns.
// Fragments stay opaque. This uses ordinary arithmetic, NOT a Boolean MMA.
__global__ void wmma16(const __half* a,const __half* b,float* c,unsigned tiles){
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
__global__ void positive_relation(const float* count,unsigned char* relation,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)relation[i]=count[i]>0.f;
}
__global__ void packed_dot(const int* a,const int* b,int* out,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__dp4a(a[i],b[i],0);
}
__global__ void butterfly32(const float* input,float* output,unsigned regions){
    unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid&31,region=tid>>5;
    if(region>=regions)return;
    float x=input[region*32+lane];
    for(unsigned step=1;step<32;step<<=1){float peer=__shfl_xor_sync(full,x,step);x=(lane&step)?peer-x:x+peer;}
    output[region*32+lane]=x;
}
// Experimental floating lookup only. Caller configures the texture explicitly.
// Linear interpolation has finite coordinate precision; it is not an exact gate.
__global__ void texture_response(cudaTextureObject_t table,const float2* coordinates,float* out,unsigned n){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float2 p=coordinates[i];out[i]=tex2D<float>(table,p.x,p.y);}
}
} // namespace bp_moon_cuda
