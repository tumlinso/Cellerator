#pragma once
#include <Cellerator/math/matrix/contracts.hh>
namespace cellerator::math::matrix {
struct patch_descriptor {
    rel::axis_descriptor input_rows{},input_columns{},output_rows{},output_columns{};
    std::uint64_t row_stride=0; // zero selects contiguous width; other strides unsupported
    rel::arithmetic_policy policy=host_policy();
};
struct patch_primal {
    std::span<const float> state,left,right;
    const generations* current=nullptr; // operand roles: state,L,R,external context
};
struct patch_workspace { std::span<float> preactivation,activation; };
// Borrowed tape: owner retains primal/workspace bytes unchanged until response.
// Generation metadata is also borrowed; external owner serializes publication.
// No canonical parameter owner, snapshot allocation or optimizer is introduced.
struct patch_tape {
    patch_descriptor descriptor{};
    patch_primal primal{};
    generations saved{};
    std::span<const float> preactivation,activation;
};
inline constexpr capabilities patch_capabilities{};
// Y=tanh(LX)R. LXR is a separate restricted family, not this operation and not
// an equivalent replacement for an arbitrary sparse matrix operator.
status patch_forward(const patch_descriptor&,const patch_primal&,patch_workspace,
                     std::span<float> output,patch_tape&) noexcept;
status patch_vjp(const patch_tape&,std::span<const float> cotangent,
    std::span<float> state_gradient,std::span<float> left_gradient,std::span<float> right_gradient) noexcept;
status patch_jvp(const patch_tape&,std::span<const float> state_direction,
    std::span<const float> left_direction,std::span<const float> right_direction,std::span<float> response) noexcept;
} // namespace cellerator::math::matrix
