#pragma once
#include <Cellerator/layout/factorized_interaction.hh>
#include <ATen/ATen.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::modeling::cuda {
class FactorizedInteractionDeviceLayout;
}

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

// Native gated joint-interaction adapter. Inputs must be contiguous CUDA
// tensors on one device with shapes x[input,channel,cell], z[context,cell],
// alpha[parameter_row,channel], bias[parameter], gamma[parameter]. Activation
// and parameter groups independently use matching FP16 or FP32 storage. The
// output dtype and packet/direct route are explicit. This overload prepares
// and uploads structure for this call; use the prepared-owner overload below
// to reuse immutable structure across repeated calls.
at::Tensor factorized_interaction_cuda(
    const layout::FactorizedInteractionLayout& plan,
    const at::Tensor& x, const at::Tensor& z, const at::Tensor& alpha,
    const at::Tensor& bias, const at::Tensor& gamma,
    at::ScalarType output_dtype = at::kHalf,
    bool direct = false);

// Reuses an immutable uploaded structural owner. The shared owner is retained
// by each autograd invocation through forward/backward stream completion. It
// must match the tensor device; the adapter orders its upload against the
// current Torch stream and allocates private tape/scratch for each call.
at::Tensor factorized_interaction_cuda_prepared(
    const std::shared_ptr<modeling::cuda::FactorizedInteractionDeviceLayout>& plan,
    const at::Tensor& x, const at::Tensor& z, const at::Tensor& alpha,
    const at::Tensor& bias, const at::Tensor& gamma,
    at::ScalarType output_dtype = at::kHalf,
    bool direct = false);
} // namespace cellerator::nn::ops
