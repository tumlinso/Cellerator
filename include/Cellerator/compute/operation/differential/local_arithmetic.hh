#pragma once
#include <Cellerator/compute/operation/native_numeric/device_linear.hh>
#include <Cellerator/compute/operation/native_numeric/local_arithmetic.hh>
#include <Cellerator/compute/operation/native_foundation_contract.hh>

namespace cellerator::compute::differential {
namespace numeric = native_numeric;
namespace nf1 = operation::nf1;
// Mathematical actions at supplied stored primal values. Independent output
// contributions are overwrite-only; callers explicitly assemble repeated roles.
// Host spans remain borrowed through execution. CUDA primitives retain only
// borrowed resident-owner references and derive their live generations on each
// admission.
template<class T> struct local_binding {
    std::span<const T> left, right;
    std::span<const T> left_direction, right_direction, cotangent;
    std::span<T> output, left_adjoint, right_adjoint;
};
numeric::local_status local_jvp(numeric::local_operation, const local_binding<float>&) noexcept;
numeric::local_status local_jvp(numeric::local_operation, const local_binding<double>&) noexcept;
numeric::local_status local_vjp(numeric::local_operation, const local_binding<float>&) noexcept;
numeric::local_status local_vjp(numeric::local_operation, const local_binding<double>&) noexcept;
// Device actions use the same stored primal convention as the host actions.
// The caller owns all device memory through stream completion.
template<class T> struct local_device_binding {
    const T* left = nullptr;
    const T* right = nullptr;
    const T* left_direction = nullptr;
    const T* right_direction = nullptr;
    const T* cotangent = nullptr;
    T* output = nullptr;
    T* left_adjoint = nullptr;
    T* right_adjoint = nullptr;
    std::uint64_t count = 0;
};
// Device primal owners are captured when the primitive is prepared.  They are
// borrowed: callers retain the vectors and stream until program completion.
// The request carries semantic primal metadata, while admission derives the
// mutable state/parameter generations from these resident CE values.
struct local_primal_owners {
    const numeric::resident_vector* left = nullptr;
    const numeric::resident_vector* right = nullptr;
    const numeric::resident_vector* state = nullptr;
    const numeric::resident_vector* parameters = nullptr;
    cudaStream_t stream = nullptr;
    nf1::primal_record identity{};
};
template<class T> struct response_binding {
    const numeric::resident_vector* left_direction = nullptr;
    const numeric::resident_vector* right_direction = nullptr;
    const numeric::resident_vector* cotangent = nullptr;
    const numeric::resident_vector* output = nullptr;
    const numeric::resident_vector* left_adjoint = nullptr;
    const numeric::resident_vector* right_adjoint = nullptr;
    std::uint64_t count = 0;
    nf1::derivative_request request{};
    nf1::operand_signature direction{};
    nf1::operand_signature response{};
};
struct local_block {
    numeric::local_operation operation{};
    nf1::compiled_block block{};
    local_primal_owners device_primal{};
};
// Produces callbacks for the existing runner; no graph/evaluator is added.
// Borrowed contract spans must outlive the block. Pass &local_block as stage
// prepared_state and &local_binding<T> in legacy launch_binding_v2::input.
// Accepted policies: uniform f32/f64, nearest-even, propagate NaN/Inf, no
// saturation, overwrite/nonaliasing. Unsupported capabilities fail explicitly.
nf1::status make_local_block(numeric::local_operation, const nf1::operation_contract&,
                            local_block&) noexcept;
// CUDA sibling of local_block. It remains a single primitive callback; program
// sequencing and dependency admission are still owned by prepared_program_v2.
nf1::status make_local_device_block(numeric::local_operation,
                                    const nf1::operation_contract&, const local_primal_owners&,
                                    local_block&) noexcept;
} // namespace cellerator::compute::differential
