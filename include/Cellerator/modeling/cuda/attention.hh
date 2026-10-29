#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::modeling::cuda {

struct SparseAttentionPlan {
    const std::int64_t* offsets = nullptr; // device CSR offsets [query_count+1]
    const std::int64_t* key_ids = nullptr; // device key row for each active edge
    std::int64_t query_count = 0;
    std::int64_t key_count = 0;
    std::int64_t edge_count = 0;
    std::int64_t query_key_dim = 0;
    std::int64_t value_dim = 0;
};

// Caller-owned FP32 intermediates; edge arrays have edge_count elements,
// query_correction has query_count elements. Reusable between invocations
// after the prior stream work completes.
struct SparseAttentionWorkspace {
    float* logits = nullptr;
    std::size_t logits_capacity = 0;
    float* probabilities = nullptr;
    std::size_t probabilities_capacity = 0;
    float* edge_value_dot = nullptr;
    std::size_t edge_value_dot_capacity = 0;
    float* query_correction = nullptr;
    std::size_t query_correction_capacity = 0;
};

enum class GradientWriteMode : std::uint8_t { zero, accumulate };

// Relation-restricted scaled dot-product attention. q is [Q,D], k [N,D],
// v [N,V], output [Q,V]. CSR contains only active query->key edges: no dense
// adjacency or implicit all-pairs work is performed. The caller supplies the
// scale (commonly 1/sqrt(D)). Empty neighborhoods produce all-zero output.
// CSR offsets/keys must be monotone/in-bounds and arrays live on the payload
// device. No allocation or synchronization is performed.
cudaError_t sparse_attention_fp16(
    const __half* q, std::size_t q_capacity, const __half* k, std::size_t k_capacity,
    const __half* v, std::size_t v_capacity, __half* output, std::size_t output_capacity,
    SparseAttentionPlan plan, float scale, SparseAttentionWorkspace workspace,
    cudaStream_t stream) noexcept;
cudaError_t sparse_attention_fp32(
    const float* q, std::size_t q_capacity, const float* k, std::size_t k_capacity,
    const float* v, std::size_t v_capacity, float* output, std::size_t output_capacity,
    SparseAttentionPlan plan, float scale, SparseAttentionWorkspace workspace,
    cudaStream_t stream) noexcept;

// Optional producer-consumer composition for rows of at most 32 edges. A
// single warp scores, normalizes and aggregates such a row without materializing
// logits; it still writes probabilities because backward consumes them. Larger
// rows use the staged path above. The staged entrypoints remain the control.
cudaError_t sparse_attention_fused_small_fp16(
    const __half* q, std::size_t q_capacity, const __half* k, std::size_t k_capacity,
    const __half* v, std::size_t v_capacity, __half* output, std::size_t output_capacity,
    SparseAttentionPlan plan, float scale, SparseAttentionWorkspace workspace,
    cudaStream_t stream) noexcept;
cudaError_t sparse_attention_fused_small_fp32(
    const float* q, std::size_t q_capacity, const float* k, std::size_t k_capacity,
    const float* v, std::size_t v_capacity, float* output, std::size_t output_capacity,
    SparseAttentionPlan plan, float scale, SparseAttentionWorkspace workspace,
    cudaStream_t stream) noexcept;

// First-order VJP. All gradients are FP32 [Q,D], [N,D], [N,V]. In `zero`
// mode they are cleared asynchronously before accumulation; `accumulate`
// adds to caller-provided gradients. Repeated key IDs use FP32 atomic adds, so
// key/value gradients can vary slightly with scheduling. Query gradients have
// one owning CTA and are written deterministically. Gradient of CSR indices is
// undefined. Probabilities must come from a preceding forward call with the
// same plan, scale and input state. Workspace must be separate from payloads
// and gradients.
cudaError_t sparse_attention_backward_fp16(
    const __half* q, std::size_t q_capacity, const __half* k, std::size_t k_capacity,
    const __half* v, std::size_t v_capacity, const __half* grad_output,
    std::size_t grad_output_capacity, SparseAttentionPlan plan, float scale,
    SparseAttentionWorkspace workspace, float* grad_q, std::size_t grad_q_capacity,
    float* grad_k, std::size_t grad_k_capacity, float* grad_v,
    std::size_t grad_v_capacity, GradientWriteMode mode, cudaStream_t stream) noexcept;
cudaError_t sparse_attention_backward_fp32(
    const float* q, std::size_t q_capacity, const float* k, std::size_t k_capacity,
    const float* v, std::size_t v_capacity, const float* grad_output,
    std::size_t grad_output_capacity, SparseAttentionPlan plan, float scale,
    SparseAttentionWorkspace workspace, float* grad_q, std::size_t grad_q_capacity,
    float* grad_k, std::size_t grad_k_capacity, float* grad_v,
    std::size_t grad_v_capacity, GradientWriteMode mode, cudaStream_t stream) noexcept;

} // namespace cellerator::modeling::cuda
