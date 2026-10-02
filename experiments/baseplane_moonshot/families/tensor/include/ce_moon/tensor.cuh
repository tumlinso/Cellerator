#pragma once
#include <ce_moon/volta.cuh>
#include <ce_moon/tensor.hpp>
namespace ce_moon::tensor::cuda {
// All pointers are device pointers, nonaliasing caller-owned storage. WMMA
// a/b pointers require 32-byte alignment. Exactly one tile per invocation.
// Pack/transpose/WMMA/postprocess share the caller stream; no allocation/sync.
static __global__ void pack(const float* a,const float* b,__half* pa,__half* pb,unsigned rows,unsigned features,unsigned outputs,bool transpose){
 unsigned n=threadIdx.x;if(n>=256)return;unsigned i=n/16,j=n%16;
 pa[n]=__float2half_rn(i<rows&&j<features?a[n]:0.f);
 pb[n]=__float2half_rn(i<features&&j<outputs?(transpose?b[j*16+i]:b[n]):0.f);
}
static __global__ void activate(float* result,unsigned rows,unsigned outputs){unsigned n=threadIdx.x;if(n<256){unsigned i=n/16,j=n%16;result[n]=i<rows&&j<outputs?tanhf(result[n]):0.f;}}
struct DevicePair {unsigned long long from,to;unsigned row,column;float score;};
// Deterministic bounded sparse emit. required and overflow use caller scalars.
// A one-thread postpass is intentional for this 256-entry experimental tile.
static __global__ void emit(const float* scores,const unsigned long long* ids,const unsigned char* mask,unsigned rows,float threshold,DevicePair* pairs,unsigned capacity,unsigned* required,unsigned* overflow){
 if(threadIdx.x||blockIdx.x)return;unsigned count=0;
 for(unsigned i=0;i<rows;++i)for(unsigned j=0;j<rows;++j){unsigned n=i*16+j;if(mask[n]&&scores[n]>threshold){if(count<capacity)pairs[count]={ids[i],ids[j],i,j,scores[n]};++count;}}
 *required=count;*overflow=count>capacity;
}
inline cudaError_t object_features(const float* x,const float* w,__half* packed_x,__half* packed_w,float* result,TileShape shape,cudaStream_t stream){
 validate(shape);pack<<<1,256,0,stream>>>(x,w,packed_x,packed_w,shape.rows,shape.features,shape.outputs,false);auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
 ce_moon_cuda::wmma16<<<1,32,0,stream>>>(packed_x,packed_w,result,1);return cudaGetLastError();
}
inline cudaError_t relation_scores(const float* q,const float* k,__half* packed_q,__half* packed_kt,float* scores,unsigned rows,unsigned features,cudaStream_t stream){
 validate({rows,features,rows});pack<<<1,256,0,stream>>>(q,k,packed_q,packed_kt,rows,features,rows,true);auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
 ce_moon_cuda::wmma16<<<1,32,0,stream>>>(packed_q,packed_kt,scores,1);return cudaGetLastError();
}
inline cudaError_t compact_relations(const float* scores,const unsigned long long* ids,const unsigned char* mask,unsigned rows,float threshold,DevicePair* pairs,unsigned capacity,unsigned* required,unsigned* overflow,cudaStream_t stream){
 validate({rows,1,1});finite(threshold);emit<<<1,1,0,stream>>>(scores,ids,mask,rows,threshold,pairs,capacity,required,overflow);return cudaGetLastError();
}
// Binary float input validated by caller. Counts [0,16] are exact under FP16
// conversion/FP32 accumulation; arbitrary weighted or repeated paths are not.
inline cudaError_t finite_relation_counts(const float* a,const float* b,__half* packed_a,__half* packed_b,float* counts,unsigned char* exists,cudaStream_t stream){
 auto error=object_features(a,b,packed_a,packed_b,counts,{},stream);if(error!=cudaSuccess)return error;
 ce_moon_cuda::positive_relation<<<1,256,0,stream>>>(counts,exists,256);return cudaGetLastError();
}
inline cudaError_t possible_states(const float* states,const float* weights,__half* packed_states,__half* packed_weights,float* table,TileShape shape,cudaStream_t stream){
 auto error=object_features(states,weights,packed_states,packed_weights,table,shape,stream);if(error!=cudaSuccess)return error;
 activate<<<1,256,0,stream>>>(table,shape.rows,shape.outputs);return cudaGetLastError();
}
} // namespace ce_moon::tensor::cuda
