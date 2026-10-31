#include <Cellerator/modeling/cuda/attention.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {
constexpr int kThreads = 128;
constexpr std::int64_t kFusedSmallDegree = 32;
constexpr std::int64_t kMaxGrid = 65535;

bool product(std::int64_t a, std::int64_t b, std::size_t& out) noexcept {
    if (a < 0 || b < 0) return false;
    const auto x = static_cast<std::uint64_t>(a), y = static_cast<std::uint64_t>(b);
    if (y && x > std::numeric_limits<std::size_t>::max() / y) return false;
    const auto n = x * y;
    if (n > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        n > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) return false;
    out = static_cast<std::size_t>(n);
    return true;
}
bool capacity(const void* p, std::size_t have, std::size_t need) noexcept {
    return need == 0 || (p && have >= need);
}
bool float_count(std::int64_t count, std::size_t& out) noexcept {
    if (count < 0 || static_cast<std::uint64_t>(count) >
                         std::numeric_limits<std::size_t>::max() / sizeof(float)) return false;
    out = static_cast<std::size_t>(count);
    return true;
}
int blocks(std::int64_t count) noexcept {
    const auto n = count / kThreads + (count % kThreads != 0);
    return static_cast<int>(std::min<std::int64_t>(kMaxGrid, std::max<std::int64_t>(1, n)));
}
int query_blocks(std::int64_t count) noexcept {
    return static_cast<int>(std::min<std::int64_t>(kMaxGrid, std::max<std::int64_t>(1, count)));
}
bool valid_plan(SparseAttentionPlan p) noexcept {
    if (p.query_count < 0 || p.key_count < 0 || p.edge_count < 0 ||
        p.query_key_dim <= 0 || p.value_dim <= 0) return false;
    if (p.query_count > 0 && p.offsets == nullptr) return false;
    if (p.edge_count > 0 && (p.key_ids == nullptr || p.query_count == 0 || p.key_count == 0)) return false;
    return true;
}
template<class T> __device__ float as_float(T x);
template<> __device__ float as_float(float x) { return x; }
template<> __device__ float as_float(__half x) { return __half2float(x); }
template<class T> __device__ T from_float(float x);
template<> __device__ float from_float(float x) { return x; }
template<> __device__ __half from_float(float x) { return __float2half_rn(x); }

template<class T>
__global__ void edge_logits_kernel(const T* q, const T* k,
                                   const std::int64_t* offsets,
                                   const std::int64_t* keys,
                                   std::int64_t queries, std::int64_t dim,
                                   float scale, float* logits, bool skip_small) {
    for (std::int64_t row = blockIdx.x; row < queries; row += gridDim.x) {
        if (threadIdx.x == 0) {
            const auto begin = offsets[row], end = offsets[row + 1];
            if (!skip_small || end - begin > kFusedSmallDegree) {
                for (auto edge = begin; edge < end; ++edge) {
                    const auto key = keys[edge];
                    float dot = 0.0f;
                    for (std::int64_t d = 0; d < dim; ++d)
                        dot += as_float(q[row * dim + d]) * as_float(k[key * dim + d]);
                    logits[edge] = dot * scale;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void segmented_softmax_kernel(const std::int64_t* offsets,
                                         std::int64_t queries, const float* logits,
                                         float* probabilities, bool skip_small) {
    for (std::int64_t row = blockIdx.x; row < queries; row += gridDim.x) {
        if (threadIdx.x == 0) {
            const auto begin = offsets[row], end = offsets[row + 1];
            if ((!skip_small || end - begin > kFusedSmallDegree) && begin < end) {
                float maximum = -INFINITY;
                for (auto e = begin; e < end; ++e) maximum = fmaxf(maximum, logits[e]);
                float denominator = 0.0f;
                for (auto e = begin; e < end; ++e) {
                    const auto weight = expf(logits[e] - maximum);
                    probabilities[e] = weight;
                    denominator += weight;
                }
                const float inverse = 1.0f / denominator;
                for (auto e = begin; e < end; ++e) probabilities[e] *= inverse;
            }
        }
        __syncthreads();
    }
}

template<class T>
__global__ void aggregate_kernel(const T* v, T* output,
                                 const std::int64_t* offsets,
                                 const std::int64_t* keys,
                                 const float* probabilities,
                                 std::int64_t queries, std::int64_t value_dim,
                                 bool skip_small) {
    for (std::int64_t row = blockIdx.x; row < queries; row += gridDim.x) {
        const auto begin = offsets[row], end = offsets[row + 1];
        if (skip_small && end - begin <= kFusedSmallDegree) { __syncthreads(); continue; }
        for (std::int64_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
            float sum = 0.0f;
            for (auto edge = begin; edge < end; ++edge)
                sum += probabilities[edge] * as_float(v[keys[edge] * value_dim + d]);
            output[row * value_dim + d] = static_cast<T>(sum);
        }
        __syncthreads();
    }
}

template<class T>
__global__ void fused_small_kernel(const T* q, const T* k, const T* v, T* output,
                                   const std::int64_t* offsets, const std::int64_t* keys,
                                   std::int64_t queries, std::int64_t dim,
                                   std::int64_t value_dim, float scale, float* probabilities) {
    constexpr unsigned mask = 0xffffffffu;
    __shared__ float aggregation_probabilities[32];
    __shared__ std::int64_t aggregation_keys[32];
    for (std::int64_t row = blockIdx.x; row < queries; row += gridDim.x) {
        const auto begin = offsets[row], end = offsets[row + 1];
        const auto degree = end - begin;
        if (degree <= kFusedSmallDegree) {
            float score = -INFINITY;
            if (threadIdx.x < degree) {
                const auto edge = begin + threadIdx.x;
                const auto key = keys[edge];
                float dot = 0.0f;
                for (std::int64_t d = 0; d < dim; ++d)
                    dot += as_float(q[row * dim + d]) * as_float(k[key * dim + d]);
                score = dot * scale;
            }
            float maximum = score;
            for (int delta = 16; delta > 0; delta >>= 1)
                maximum = fmaxf(maximum, __shfl_down_sync(mask, maximum, delta));
            maximum = __shfl_sync(mask, maximum, 0);
            float weight = threadIdx.x < degree ? expf(score - maximum) : 0.0f;
            float denominator = weight;
            for (int delta = 16; delta > 0; delta >>= 1)
                denominator += __shfl_down_sync(mask, denominator, delta);
            denominator = __shfl_sync(mask, denominator, 0);
            const float probability = degree > 0 ? weight / denominator : 0.0f;
            if (threadIdx.x < degree) {
                probabilities[begin + threadIdx.x] = probability;
                aggregation_probabilities[threadIdx.x] = probability;
                aggregation_keys[threadIdx.x] = keys[begin + threadIdx.x];
            }
            __syncthreads();
            for (std::int64_t d = threadIdx.x; d < value_dim; d += 32) {
                float sum = 0.0f;
                for (std::int64_t edge = 0; edge < degree; ++edge) {
                    const float p = aggregation_probabilities[edge];
                    const auto key = aggregation_keys[edge];
                    sum += p * as_float(v[key * value_dim + d]);
                }
                output[row * value_dim + d] = from_float<T>(sum);
            }
            __syncthreads();
        }
    }
}
template<> __global__ void aggregate_kernel<__half>(const __half* v, __half* output,
                                 const std::int64_t* offsets, const std::int64_t* keys,
                                 const float* probabilities, std::int64_t queries,
                                 std::int64_t value_dim, bool skip_small) {
    for (std::int64_t row = blockIdx.x; row < queries; row += gridDim.x) {
        const auto begin = offsets[row], end = offsets[row + 1];
        if (skip_small && end - begin <= kFusedSmallDegree) { __syncthreads(); continue; }
        for (std::int64_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
            float sum = 0.0f;
            for (auto edge = begin; edge < end; ++edge)
                sum += probabilities[edge] * __half2float(v[keys[edge] * value_dim + d]);
            output[row * value_dim + d] = __float2half_rn(sum);
        }
        __syncthreads();
    }
}

template<class T>
__global__ void edge_value_dot_kernel(const T* v, const T* grad_output,
                                      const std::int64_t* offsets,
                                      const std::int64_t* keys,
                                      std::int64_t queries, std::int64_t value_dim,
                                      float* edge_dot) {
    for (std::int64_t row = blockIdx.x; row < queries; row += gridDim.x) {
        if (threadIdx.x == 0)
            for (auto e = offsets[row]; e < offsets[row + 1]; ++e) {
                    float dot = 0.0f;
                    const auto key = keys[e];
                    for (std::int64_t d = 0; d < value_dim; ++d)
                        dot += as_float(grad_output[row * value_dim + d]) *
                               as_float(v[key * value_dim + d]);
                    edge_dot[e] = dot;
            }
        __syncthreads();
    }
}

__global__ void correction_kernel(const std::int64_t* offsets, std::int64_t queries,
                                  const float* probabilities, const float* edge_dot,
                                  float* correction) {
    for (std::int64_t row = blockIdx.x; row < queries; row += gridDim.x) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (auto e = offsets[row]; e < offsets[row + 1]; ++e)
                sum += probabilities[e] * edge_dot[e];
            correction[row] = sum;
        }
        __syncthreads();
    }
}

template<class T>
__global__ void backward_rows_kernel(const T* q, const T* k, const T* grad_output,
                                     const T* v, const std::int64_t* offsets,
                                     const std::int64_t* keys, const float* probabilities,
                                     const float* edge_dot, const float* correction,
                                     std::int64_t queries, std::int64_t dim,
                                     std::int64_t value_dim, float scale,
                                     float* grad_q, float* grad_k, float* grad_v,
                                     bool accumulate_q) {
    for (std::int64_t row = blockIdx.x; row < queries; row += gridDim.x) {
        const auto begin = offsets[row], end = offsets[row + 1];
        __shared__ float corr;
        if (threadIdx.x == 0) corr = correction[row];
        __syncthreads();
        const auto max_dim = dim > value_dim ? dim : value_dim;
        for (std::int64_t d = threadIdx.x; d < max_dim; d += blockDim.x) {
            float qsum = 0.0f;
            for (auto e = begin; e < end; ++e) {
                const auto key = keys[e];
                const float p = probabilities[e];
                const float dlogit = p * (edge_dot[e] - corr);
                if (d < dim) {
                    qsum += scale * dlogit * as_float(k[key * dim + d]);
                    atomicAdd(&grad_k[key * dim + d], scale * dlogit * as_float(q[row * dim + d]));
                }
                if (d < value_dim)
                    atomicAdd(&grad_v[key * value_dim + d], p * as_float(grad_output[row * value_dim + d]));
            }
            if (d < dim) {
                const auto index = row * dim + d;
                if (accumulate_q) grad_q[index] += qsum;
                else grad_q[index] = qsum;
            }
        }
        __syncthreads();
    }
}

template<class T>
cudaError_t forward(const T* q, std::size_t qc, const T* k, std::size_t kc,
                    const T* v, std::size_t vc, T* out, std::size_t oc,
                    SparseAttentionPlan p, float scale, SparseAttentionWorkspace w,
                    cudaStream_t stream, bool fuse_small) noexcept {
    if (!valid_plan(p) || !std::isfinite(scale)) return cudaErrorInvalidValue;
    std::size_t qn=0, kn=0, vn=0, on=0, edges=0;
    if (!product(p.query_count,p.query_key_dim,qn) || !product(p.key_count,p.query_key_dim,kn) ||
        !product(p.key_count,p.value_dim,vn) || !product(p.query_count,p.value_dim,on) ||
        !float_count(p.edge_count,edges) ||
        !capacity(q,qc,qn) || !capacity(k,kc,kn) || !capacity(v,vc,vn) || !capacity(out,oc,on) ||
        !capacity(w.logits,w.logits_capacity,edges) ||
        !capacity(w.probabilities,w.probabilities_capacity,edges))
        return cudaErrorInvalidValue;
    if (p.query_count == 0) return cudaSuccess;
    if (fuse_small) {
        fused_small_kernel<<<query_blocks(p.query_count), 32, 0, stream>>>(
            q,k,v,out,p.offsets,p.key_ids,p.query_count,p.query_key_dim,p.value_dim,
            scale,w.probabilities);
        auto error=cudaGetLastError(); if(error!=cudaSuccess)return error;
    }
    if (p.edge_count > 0) {
        edge_logits_kernel<<<query_blocks(p.query_count), kThreads, 0, stream>>>(q,k,p.offsets,p.key_ids,p.query_count,p.query_key_dim,scale,w.logits,fuse_small);
        auto error=cudaGetLastError(); if(error!=cudaSuccess)return error;
        segmented_softmax_kernel<<<query_blocks(p.query_count), kThreads, 0, stream>>>(p.offsets,p.query_count,w.logits,w.probabilities,fuse_small);
        error=cudaGetLastError(); if(error!=cudaSuccess)return error;
    }
    aggregate_kernel<<<query_blocks(p.query_count), kThreads, 0, stream>>>(v,out,p.offsets,p.key_ids,w.probabilities,p.query_count,p.value_dim,fuse_small);
    return cudaGetLastError();
}

template<class T>
cudaError_t backward(const T* q,std::size_t qc,const T* k,std::size_t kc,const T* v,std::size_t vc,
                     const T* go,std::size_t goc,SparseAttentionPlan p,float scale,
                     SparseAttentionWorkspace w,float* gq,std::size_t gqc,float* gk,std::size_t gkc,
                     float* gv,std::size_t gvc,GradientWriteMode mode,cudaStream_t stream) noexcept {
    if (!valid_plan(p) || !std::isfinite(scale) ||
        (mode!=GradientWriteMode::zero && mode!=GradientWriteMode::accumulate)) return cudaErrorInvalidValue;
    std::size_t qn=0,kn=0,vn=0,gn=0,edges=0;
    if (!product(p.query_count,p.query_key_dim,qn)||!product(p.key_count,p.query_key_dim,kn)||
        !product(p.key_count,p.value_dim,vn)||!product(p.query_count,p.value_dim,gn)||
        !float_count(p.edge_count,edges)||
        !capacity(q,qc,qn)||!capacity(k,kc,kn)||!capacity(v,vc,vn)||!capacity(go,goc,gn)||
        !capacity(gq,gqc,qn)||!capacity(gk,gkc,kn)||!capacity(gv,gvc,vn)||
        !capacity(w.probabilities,w.probabilities_capacity,edges)||
        !capacity(w.edge_value_dot,w.edge_value_dot_capacity,edges)||
        !capacity(w.query_correction,w.query_correction_capacity,
                  p.edge_count > 0 ? static_cast<std::size_t>(p.query_count) : 0))
        return cudaErrorInvalidValue;
    if (mode==GradientWriteMode::zero) {
        cudaError_t e = cudaSuccess;
        if (qn) { e=cudaMemsetAsync(gq,0,qn*sizeof(float),stream); if(e!=cudaSuccess)return e; }
        if (kn) { e=cudaMemsetAsync(gk,0,kn*sizeof(float),stream); if(e!=cudaSuccess)return e; }
        if (vn) { e=cudaMemsetAsync(gv,0,vn*sizeof(float),stream); if(e!=cudaSuccess)return e; }
    }
    if (p.query_count==0)return cudaSuccess;
    if (p.edge_count>0) {
        edge_value_dot_kernel<<<query_blocks(p.query_count),kThreads,0,stream>>>(v,go,p.offsets,p.key_ids,p.query_count,p.value_dim,w.edge_value_dot);
        auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
        correction_kernel<<<query_blocks(p.query_count),kThreads,0,stream>>>(p.offsets,p.query_count,w.probabilities,w.edge_value_dot,w.query_correction);
        e=cudaGetLastError();if(e!=cudaSuccess)return e;
    } else {
        // With no edges, all derivatives are zero. Zero mode was cleared above;
        // accumulation mode must preserve the caller's existing buffers.
        return cudaSuccess;
    }
    backward_rows_kernel<<<query_blocks(p.query_count),kThreads,0,stream>>>(q,k,go,v,p.offsets,p.key_ids,w.probabilities,w.edge_value_dot,w.query_correction,
        p.query_count,p.query_key_dim,p.value_dim,scale,gq,gk,gv,mode==GradientWriteMode::accumulate);
    return cudaGetLastError();
}
} // namespace

cudaError_t sparse_attention_fp16(const __half*q,std::size_t qc,const __half*k,std::size_t kc,const __half*v,std::size_t vc,__half*o,std::size_t oc,SparseAttentionPlan p,float s,SparseAttentionWorkspace w,cudaStream_t st) noexcept { return forward(q,qc,k,kc,v,vc,o,oc,p,s,w,st,false); }
cudaError_t sparse_attention_fp32(const float*q,std::size_t qc,const float*k,std::size_t kc,const float*v,std::size_t vc,float*o,std::size_t oc,SparseAttentionPlan p,float s,SparseAttentionWorkspace w,cudaStream_t st) noexcept { return forward(q,qc,k,kc,v,vc,o,oc,p,s,w,st,false); }
cudaError_t sparse_attention_fused_small_fp16(const __half*q,std::size_t qc,const __half*k,std::size_t kc,const __half*v,std::size_t vc,__half*o,std::size_t oc,SparseAttentionPlan p,float s,SparseAttentionWorkspace w,cudaStream_t st) noexcept { return forward(q,qc,k,kc,v,vc,o,oc,p,s,w,st,true); }
cudaError_t sparse_attention_fused_small_fp32(const float*q,std::size_t qc,const float*k,std::size_t kc,const float*v,std::size_t vc,float*o,std::size_t oc,SparseAttentionPlan p,float s,SparseAttentionWorkspace w,cudaStream_t st) noexcept { return forward(q,qc,k,kc,v,vc,o,oc,p,s,w,st,true); }
cudaError_t sparse_attention_backward_fp16(const __half*q,std::size_t qc,const __half*k,std::size_t kc,const __half*v,std::size_t vc,const __half*go,std::size_t goc,SparseAttentionPlan p,float s,SparseAttentionWorkspace w,float*gq,std::size_t gqc,float*gk,std::size_t gkc,float*gv,std::size_t gvc,GradientWriteMode m,cudaStream_t st) noexcept { return backward(q,qc,k,kc,v,vc,go,goc,p,s,w,gq,gqc,gk,gkc,gv,gvc,m,st); }
cudaError_t sparse_attention_backward_fp32(const float*q,std::size_t qc,const float*k,std::size_t kc,const float*v,std::size_t vc,const float*go,std::size_t goc,SparseAttentionPlan p,float s,SparseAttentionWorkspace w,float*gq,std::size_t gqc,float*gk,std::size_t gkc,float*gv,std::size_t gvc,GradientWriteMode m,cudaStream_t st) noexcept { return backward(q,qc,k,kc,v,vc,go,goc,p,s,w,gq,gqc,gk,gkc,gv,gvc,m,st); }
} // namespace cellerator::modeling::cuda
