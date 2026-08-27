#pragma once

#include <Cellerator/execution/program.hh>

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace celleratorch {

enum class program_op_status_code : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    tensor_not_cuda = 2u,
    device_mismatch = 3u,
    dtype_mismatch = 4u,
    rank_mismatch = 5u,
    shape_mismatch = 6u,
    stride_mismatch = 7u,
    torch_failure = 8u,
    native_failure = 9u
};

struct program_op_status {
    program_op_status_code code = program_op_status_code::ok;
    cellerator::execution::executable_program_status native{};
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == program_op_status_code::ok;
    }
};

// Thin, allocation-free binding seam for the Wave D custom operation.
//
// The launch template carries all native biological identities, current value
// bindings/readiness, scalar bindings, and workspace. Exactly one dense input
// and one dense output are rebound to the supplied Torch tensors for this call.
// The tensors and native program remain caller-owned. Registration and package
// wiring belong to the CE-LIVE-43 fan-in.
program_op_status run_program_forward(
    cellerator::execution::executable_program *program,
    const at::Tensor &input,
    const at::Tensor &output,
    cellerator::execution::executable_program_launch launch,
    cellerator::execution::executable_program_result *result) noexcept;

} // namespace celleratorch
