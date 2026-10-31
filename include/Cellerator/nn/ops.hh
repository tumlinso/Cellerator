#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <vector>

namespace cellerator::nn::ops {
// FP32 CPU, 1D x[N], coefficient[M], lhs/rhs[M] int64. Output[M].
// y[i] = coefficient[i] * x[lhs[i]] * x[rhs[i]]. First-order VJP only.
at::Tensor product2_cpu(const at::Tensor& x, const at::Tensor& coefficient,
                        const at::Tensor& lhs, const at::Tensor& rhs);

// Matching CUDA FP16/FP32 input[R,I], weights[E,O,I]. Immutable host
// row_offsets[E+1] partition rows by expert. Empty expert segments are valid.
// A separate native executor is retained per autograd invocation.
at::Tensor grouped_linear_cuda(const at::Tensor& input, const at::Tensor& weights,
                               const std::vector<std::int64_t>& row_offsets);

// Explicit staged native first-order attention. q[Q,H,D], k[N,H,D],
// v[N,H,V], CUDA int64 offsets[Q+1], key_ids[edges]. Matching FP16/FP32
// payload storage; FP32 internal accumulation/VJP, cast back to input dtype.
// Private immutable CSR snapshots; duplicates retained and empty rows zero.
// dropout_p must be zero. Preparation may synchronize to validate raw CSR.
at::Tensor csr_scaled_dot_product_attention_cuda(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& offsets, const at::Tensor& key_ids,
    double scale, double dropout_p = 0.0);
// Trusted structure route for identity-bound PreparedRelation. Caller guarantees
// validated immutable contiguous CSR. No content reads, snapshots, or host
// transfers are performed; original structure tensors are saved/version-checked.
at::Tensor csr_scaled_dot_product_attention_cuda_prepared(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& offsets, const at::Tensor& key_ids,
    double scale, double dropout_p = 0.0);
} // namespace cellerator::nn::ops
