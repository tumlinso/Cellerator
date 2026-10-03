#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace cellerator::compiler::substrate {
// Enumerated instruction families, rather than a target-name string shortcut.
enum class instruction { scalar_f32, popcount, mma_f16_f32, binary_mma, tf32_mma, cp_async };
constexpr bool sm70_supports(instruction i) noexcept {
    return i == instruction::scalar_f32 || i == instruction::popcount || i == instruction::mma_f16_f32;
}
enum class realization { direct, fused_scaled_tanh };
enum class numerical_policy { nearest_ieee_f32, approximate };
struct specialization_guard {
    std::size_t extent = 0;
    std::uint64_t support_generation = 0;
    numerical_policy policy = numerical_policy::nearest_ieee_f32;
};
// Guard only structural inputs. Trainable parameter values remain live operands.
struct scaled_tanh_plan {
    specialization_guard guard;
    realization choose(specialization_guard live) const noexcept {
        return live.extent == guard.extent && live.support_generation == guard.support_generation
            && live.policy == numerical_policy::nearest_ieee_f32
            && guard.policy == live.policy ? realization::fused_scaled_tanh : realization::direct;
    }
};
// Finite support rows use the declared relation universe; response support is
// never inferred from primal zeros. This computes Boolean OR-of-AND exactly.
struct support_row_result { bool valid; std::uint64_t row; };
constexpr support_row_result compose_support_row(std::uint64_t left,
        const std::uint64_t* right_rows, unsigned inner_extent) noexcept {
    if (inner_extent > 64 || (inner_extent < 64 && (left >> inner_extent))) return {false,0};
    if (inner_extent && !right_rows) return {false,0};
    std::uint64_t out = 0;
    for (unsigned k = 0; k < inner_extent; ++k)
        if ((left >> k) & 1) out |= right_rows[k];
    return {true,out};
}
// Borrowed distinct device buffers live through stream completion. Both routes
// store z=state*parameters and y=tanh(z), with no reassociation or fast math.
// The existing local derivative owner can consume z/y; this provider advertises
// forward only. No sampled/approximate derivative or detached parameter cache.
struct device_scaled_tanh_binding {
    const float* state = nullptr;
    const float* parameters = nullptr;
    float* intermediate = nullptr;
    float* output = nullptr;
    std::size_t extent = 0;
};
cudaError_t launch_scaled_tanh(realization, device_scaled_tanh_binding, cudaStream_t) noexcept;
cudaError_t launch_guarded_scaled_tanh(const scaled_tanh_plan&, specialization_guard,
    device_scaled_tanh_binding, cudaStream_t, realization* selected = nullptr) noexcept;
} // namespace cellerator::compiler::substrate
