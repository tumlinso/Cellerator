#pragma once

#include <Cellerator/modeling/cuda/factorized_interaction.hh>

namespace cellerator::modeling::cuda::stable_detail {

// Private launch boundary: the owning dispatcher validates host-visible
// extents, pointers, dtype, alignment, device, lifetime and non-overlap first.
// Gates and slopes contain one FP32 value per canonical mechanism/cell, and
// gate_adjoint has the same extent. All calls are asynchronous on stream.
cudaError_t launch_forward(
    const layout::FactorizedInteractionView& plan,
    FactorizedInteractionProblem problem, FactorizedInteractionPrecision precision,
    FactorizedInteractionValues values, memory::cuda::DeviceView output,
    float* gates, float* slopes, cudaStream_t stream) noexcept;

// Computes dL/d(preactivation), reducing all channels with compensated FP32.
cudaError_t launch_gate_adjoint(
    const layout::FactorizedInteractionView& plan,
    FactorizedInteractionProblem problem, FactorizedInteractionPrecision precision,
    FactorizedInteractionValues values, const float* output_gradient,
    const float* gates, const float* slopes, float* gate_adjoint,
    cudaStream_t stream) noexcept;

cudaError_t launch_backward(
    const layout::FactorizedInteractionView& plan,
    FactorizedInteractionProblem problem, FactorizedInteractionPrecision precision,
    FactorizedInteractionValues values, const float* output_gradient,
    const float* gates, const float* slopes,
    FactorizedInteractionGradients gradients, float* gate_adjoint,
    FactorizedInteractionGradientMode mode, cudaStream_t stream) noexcept;

} // namespace cellerator::modeling::cuda::stable_detail
