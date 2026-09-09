#pragma once
#include <Cellerator/compute/operation/native_numeric/local_arithmetic.hh>
#include <Cellerator/compute/operation/native_foundation_contract.hh>

namespace cellerator::compute::differential {
namespace numeric = native_numeric;
namespace nf1 = operation::nf1;
// Mathematical actions at supplied stored primal values. Independent output
// contributions are overwrite-only; callers explicitly assemble repeated roles.
// No pointer or generation caching. Spans remain borrowed through execution.
template<class T> struct local_binding {
    std::span<const T> left, right;
    std::span<const T> left_direction, right_direction, cotangent;
    std::span<T> output, left_adjoint, right_adjoint;
};
numeric::local_status local_jvp(numeric::local_operation, const local_binding<float>&) noexcept;
numeric::local_status local_jvp(numeric::local_operation, const local_binding<double>&) noexcept;
numeric::local_status local_vjp(numeric::local_operation, const local_binding<float>&) noexcept;
numeric::local_status local_vjp(numeric::local_operation, const local_binding<double>&) noexcept;
struct local_block {
    numeric::local_operation operation{};
    nf1::compiled_block block{};
};
// Produces callbacks for the existing runner; no graph/evaluator is added.
// Borrowed contract spans must outlive the block. Pass &local_block as stage
// prepared_state and &local_binding<T> in legacy launch_binding_v2::input.
// Accepted policies: uniform f32/f64, nearest-even, propagate NaN/Inf, no
// saturation, overwrite/nonaliasing. Unsupported capabilities fail explicitly.
nf1::status make_local_block(numeric::local_operation, const nf1::operation_contract&,
                            local_block&) noexcept;
} // namespace cellerator::compute::differential
