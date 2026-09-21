#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <cuda_runtime_api.h>

namespace cellerator::compute::differential {
namespace {
using op = numeric::local_operation;
namespace pg = execution::program;
template<class T> __global__ void action_kernel(op operation, local_device_binding<T> b,
                                                  nf1::capability capability) {
    const auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= b.count) return;
    const T a = b.left[i], c = operation == op::tanh ? T{} : b.right[i];
    T da{}, db{}, second{};
    switch (operation) {
    case op::add: da = T{1}; db = T{1}; break;
    case op::multiply: da = c; db = a; second = T{2} * b.left_direction[i] * b.right_direction[i]; break;
    case op::tanh: {
        const T y = tanh(a);
        da = T{1} - y * y;
        second = T{-2} * y * da * b.left_direction[i] * b.left_direction[i];
        break;
    }
    }
    if (capability == nf1::jvp)
        b.output[i] = da * b.left_direction[i] + (operation == op::tanh ? T{} : db * b.right_direction[i]);
    else if (capability == nf1::vjp) {
        b.left_adjoint[i] = da * b.cotangent[i];
        if (operation != op::tanh) b.right_adjoint[i] = db * b.cotangent[i];
    } else b.output[i] = second;
}
template<class T> bool complete(const local_device_binding<T>& b, nf1::capability action, bool binary) {
    if (!b.left || b.count == 0) return b.count == 0;
    if (action == nf1::vjp)
        return b.cotangent && b.left_adjoint && (!binary || b.right_adjoint);
    return b.output && b.left_direction && (!binary || b.right_direction);
}
template<class T, nf1::capability Capability> pg::program_status device_callback(
        const void* state, const pg::launch_binding_v2& launch, void* stream) noexcept {
    if (!state || !launch.input || launch.output || launch.values || !stream) return pg::program_status::invalid_argument;
    const auto& block = *static_cast<const local_block*>(state);
    const auto& response = *static_cast<const response_binding<T>*>(launch.input);
    if (response.request.action != Capability ||
        nf1::validate_derivative(block.block.contract, response.request, response.live_primal,
                                 response.direction, response.response) != nf1::status::success)
        return pg::program_status::invalid_argument;
    const auto arity = numeric::local_arity(block.operation);
    if (!arity || !complete(response.device, Capability, arity == 2)) return pg::program_status::invalid_argument;
    const auto grid = static_cast<unsigned>((response.device.count + 255) / 256);
    action_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(block.operation, response.device, Capability);
    return cudaGetLastError() == cudaSuccess ? pg::program_status::success : pg::program_status::launch_failed;
}
}
nf1::status make_local_device_block(op operation, const nf1::operation_contract& contract,
                                    local_block& output) noexcept {
    if (!(contract.capabilities & nf1::second_direction))
        return nf1::status::unsupported_capability;
    auto first_order = contract;
    first_order.capabilities &= ~nf1::second_direction;
    auto status = make_local_block(operation, first_order, output);
    if (status != nf1::status::success) return status;
    output.block.contract = contract;
    output.block.jvp_launch = device_callback<float, nf1::jvp>;
    output.block.vjp_launch = device_callback<float, nf1::vjp>;
    output.block.second_launch = device_callback<float, nf1::second_direction>;
    // CUDA response uses f32 only; f64 remains an explicit host capability.
    if (contract.numeric.state_storage != execution::numeric_type::f32)
        return nf1::status::unsupported_capability;
    return nf1::validate_compiled_block(output.block);
}
} // namespace cellerator::compute::differential
