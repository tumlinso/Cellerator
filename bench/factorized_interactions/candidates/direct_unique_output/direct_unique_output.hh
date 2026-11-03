#pragma once

#include <Cellerator/modeling/cuda/factorized_interaction.hh>

#include <cstdint>

namespace cellerator::bench::factorized_interactions::candidates {

// This candidate is selected only when a host-prepared plan proves a bijection
// from nonempty canonical mechanisms to output targets and the frozen direct
// backend is numerically admissible. Other invocations delegate unchanged to
// the native direct implementation.
enum class DirectUniqueOutputBackend : std::uint8_t {
    direct_unique_output,
    direct,
    stable
};

cudaError_t factorized_interaction_unique_output_backend(
    const layout::FactorizedInteractionView& plan,
    modeling::cuda::FactorizedInteractionProblem problem,
    DirectUniqueOutputBackend* result) noexcept;

cudaError_t factorized_interaction_unique_output_workspace(
    const layout::FactorizedInteractionView& plan,
    modeling::cuda::FactorizedInteractionProblem problem,
    modeling::cuda::FactorizedInteractionWorkspace* result) noexcept;

cudaError_t factorized_interaction_unique_output_forward(
    const layout::FactorizedInteractionView& plan,
    modeling::cuda::FactorizedInteractionProblem problem,
    modeling::cuda::FactorizedInteractionPrecision precision,
    modeling::cuda::FactorizedInteractionValues values,
    memory::cuda::DeviceView output, memory::cuda::DeviceView tape,
    memory::cuda::DeviceView scratch, cudaStream_t stream) noexcept;

cudaError_t factorized_interaction_unique_output_backward(
    const layout::FactorizedInteractionView& plan,
    modeling::cuda::FactorizedInteractionProblem problem,
    modeling::cuda::FactorizedInteractionPrecision precision,
    modeling::cuda::FactorizedInteractionValues values,
    memory::cuda::ConstDeviceView output_gradient,
    memory::cuda::ConstDeviceView tape,
    modeling::cuda::FactorizedInteractionGradients gradients,
    memory::cuda::DeviceView scratch,
    modeling::cuda::FactorizedInteractionGradientMode mode,
    cudaStream_t stream) noexcept;

} // namespace cellerator::bench::factorized_interactions::candidates
