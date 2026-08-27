#include <CelleraTorch/program_ops.hh>

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <exception>

namespace celleratorch {
namespace {

namespace execution = cellerator::execution;

program_op_status failure(program_op_status_code code,
                          const char *message) noexcept {
    program_op_status status{};
    status.code = code;
    status.message = message;
    return status;
}

execution::numeric_type native_numeric_type(c10::ScalarType type) noexcept {
    switch (type) {
    case c10::ScalarType::Byte: return execution::numeric_type::u8;
    case c10::ScalarType::UInt16: return execution::numeric_type::u16;
    case c10::ScalarType::UInt32: return execution::numeric_type::u32;
    case c10::ScalarType::Int: return execution::numeric_type::i32;
    case c10::ScalarType::Half: return execution::numeric_type::f16;
    case c10::ScalarType::BFloat16: return execution::numeric_type::bf16;
    case c10::ScalarType::Float: return execution::numeric_type::f32;
    case c10::ScalarType::Double: return execution::numeric_type::f64;
    default: return execution::numeric_type::invalid;
    }
}

program_op_status validate_tensor(
    const at::Tensor &tensor,
    const execution::dense_tensor_view &binding) noexcept {
    if (!tensor.defined())
        return failure(program_op_status_code::invalid_argument,
            "Torch tensor is undefined");
    if (!tensor.is_cuda())
        return failure(program_op_status_code::tensor_not_cuda,
            "Torch tensor must be CUDA-resident");

    const auto mapped_type = native_numeric_type(tensor.scalar_type());
    if (mapped_type == execution::numeric_type::invalid
        || mapped_type != binding.value_type)
        return failure(program_op_status_code::dtype_mismatch,
            "Torch tensor dtype does not match the native binding");

    if (tensor.dim() != static_cast<std::int64_t>(binding.rank))
        return failure(program_op_status_code::rank_mismatch,
            "Torch tensor rank does not match the native binding");
    if (binding.rank > execution::biological_operand_max_axes)
        return failure(program_op_status_code::rank_mismatch,
            "Native binding rank exceeds the biological operand limit");

    const auto tensor_device = tensor.device().index();
    if (binding.location.residency != execution::residency_kind::device
        || tensor_device < 0
        || tensor_device != binding.location.device_ordinal)
        return failure(program_op_status_code::device_mismatch,
            "Torch tensor device does not match the native binding");

    for (std::uint8_t axis = 0u; axis < binding.rank; ++axis) {
        if (tensor.size(axis)
            != static_cast<std::int64_t>(binding.shape[axis]))
            return failure(program_op_status_code::shape_mismatch,
                "Torch tensor shape does not match the native binding");
        if (tensor.stride(axis) != binding.stride[axis])
            return failure(program_op_status_code::stride_mismatch,
                "Torch tensor stride does not match the native binding");
    }
    return {};
}

} // namespace

program_op_status run_program_forward(
    execution::executable_program *program,
    const at::Tensor &input,
    const at::Tensor &output,
    execution::executable_program_launch launch,
    execution::executable_program_result *result) noexcept {
    if (program == nullptr || result == nullptr)
        return failure(program_op_status_code::invalid_argument,
            "Native program and result storage are required");
    if (launch.bindings.input_count != 1u
        || launch.bindings.output_count != 1u
        || launch.bindings.inputs == nullptr
        || launch.bindings.outputs == nullptr)
        return failure(program_op_status_code::invalid_argument,
            "Forward wrapper requires one native input and one native output");
    if (launch.bindings.inputs[0].kind
            != execution::operand_kind::dense_tensor
        || launch.bindings.outputs[0].kind
            != execution::operand_kind::dense_tensor)
        return failure(program_op_status_code::invalid_argument,
            "Forward wrapper accepts dense native operand bindings only");

    auto input_status = validate_tensor(
        input, launch.bindings.inputs[0].storage.dense);
    if (!input_status) return input_status;
    auto output_status = validate_tensor(
        output, launch.bindings.outputs[0].storage.dense);
    if (!output_status) return output_status;
    if (input.device() != output.device())
        return failure(program_op_status_code::device_mismatch,
            "Torch input and output must use the same CUDA device");

    execution::biological_operand_view native_input =
        launch.bindings.inputs[0];
    execution::biological_operand_view native_output =
        launch.bindings.outputs[0];
    native_input.storage.dense.data = input.data_ptr();
    native_output.storage.dense.data = output.data_ptr();
    launch.bindings.inputs = &native_input;
    launch.bindings.outputs = &native_output;

    try {
        const auto device = input.device().index();
        const auto stream = c10::cuda::getCurrentCUDAStream(device);
        launch.bindings.stream = {
            reinterpret_cast<void *>(stream.stream()),
            static_cast<execution::i32>(device),
            0u
        };
        const auto native_status =
            execution::run_executable_program(program, launch, result);
        if (!native_status) {
            auto status = failure(program_op_status_code::native_failure,
                native_status.message);
            status.native = native_status;
            return status;
        }
    } catch (const c10::Error &) {
        return failure(program_op_status_code::torch_failure,
            "Torch failed while resolving the current CUDA stream");
    } catch (const std::exception &) {
        return failure(program_op_status_code::torch_failure,
            "Unexpected framework failure while binding the native program");
    } catch (...) {
        return failure(program_op_status_code::torch_failure,
            "Unknown framework failure while binding the native program");
    }

    return {};
}

} // namespace celleratorch
