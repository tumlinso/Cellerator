#include <CelleraTorch/program_ops.hh>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

namespace execution = cellerator::execution;

struct native_capture {
    std::uint64_t calls = 0u;
    void *input = nullptr;
    void *output = nullptr;
    void *stream = nullptr;
    execution::i32 device = -1;
    execution::executable_program_status next_status{};
} capture;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

execution::biological_operand_view dense_binding(
    execution::numeric_type type,
    execution::i32 device,
    execution::u64 rows,
    execution::u64 columns) {
    execution::biological_operand_view binding{};
    binding.kind = execution::operand_kind::dense_tensor;
    binding.storage.dense.location = {
        execution::residency_kind::device, {}, device, 0u};
    binding.storage.dense.value_type = type;
    binding.storage.dense.rank = 2u;
    binding.storage.dense.shape[0] = rows;
    binding.storage.dense.shape[1] = columns;
    binding.storage.dense.stride[0] = static_cast<execution::i64>(columns);
    binding.storage.dense.stride[1] = 1;
    return binding;
}

} // namespace

namespace cellerator::execution {

executable_program_status run_executable_program(
    executable_program *program,
    const executable_program_launch &launch,
    executable_program_result *result) noexcept {
    ++capture.calls;
    capture.input = launch.bindings.inputs[0].storage.dense.data;
    capture.output = launch.bindings.outputs[0].storage.dense.data;
    capture.stream = launch.bindings.stream.stream;
    capture.device = launch.bindings.stream.device_ordinal;
    if (!capture.next_status) return capture.next_status;

    static output_axis_contract output_order{};
    result->candidate = program->selected_candidate;
    result->projection = program->selected_projection;
    result->selection = program->selection;
    result->output_orders = &output_order;
    result->output_order_count = 1u;
    result->structure_epoch_value = launch.expected_structure_epoch;
    result->consumed_generation = launch.expected_value_generation;
    result->completion_stream = launch.bindings.stream;
    result->enqueued = true;
    return {};
}

} // namespace cellerator::execution

int main() {
    if (c10::cuda::device_count() == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 0;
    }

    constexpr execution::i32 device = 0;
    const auto options = at::TensorOptions()
        .dtype(at::kFloat)
        .device(at::kCUDA, device);
    auto input = at::empty({2, 3}, options);
    auto output = at::empty({2, 3}, options);
    auto native_input = dense_binding(execution::numeric_type::f32, device, 2, 3);
    auto native_output = dense_binding(execution::numeric_type::f32, device, 2, 3);

    execution::executable_program program{};
    program.preparation_count = 1u;
    execution::executable_program_launch launch{};
    launch.bindings.inputs = &native_input;
    launch.bindings.outputs = &native_output;
    launch.bindings.input_count = 1u;
    launch.bindings.output_count = 1u;
    launch.expected_structure_epoch = {7u};
    launch.expected_value_generation = {11u};
    execution::executable_program_result result{};

    const auto selected_stream = c10::cuda::getStreamFromPool(false, device);
    {
        c10::cuda::CUDAStreamGuard guard(selected_stream);
        const auto status = celleratorch::run_program_forward(
            &program, input, output, launch, &result);
        require(static_cast<bool>(status),
            "valid forward binding must succeed");
        require(capture.stream
                == reinterpret_cast<void *>(selected_stream.stream()),
            "wrapper must bind the current Torch CUDA stream");
    }
    require(capture.input == input.data_ptr(),
        "wrapper must bind the Torch input without copying");
    require(capture.output == output.data_ptr(),
        "wrapper must bind the Torch output without copying");
    require(capture.device == device, "wrapper must preserve device identity");
    require(result.output_order_count == 1u && result.output_orders != nullptr,
        "native output order metadata must remain observable");
    require(result.structure_epoch_value.value == 7u
            && result.consumed_generation.value == 11u,
        "native structure epoch and generation must remain observable");

    auto relocated_input = at::empty({2, 3}, options);
    auto relocated_output = at::empty({2, 3}, options);
    const auto first_input = capture.input;
    require(static_cast<bool>(celleratorch::run_program_forward(
            &program, relocated_input, relocated_output, launch, &result)),
        "relocated dense tensors must not require native re-preparation");
    require(capture.input != first_input
            && capture.input == relocated_input.data_ptr(),
        "changing dense pointers must update launch binding only");
    require(program.preparation_count == 1u,
        "Torch calls must not reconstruct prepared native state");

    const auto calls_before_errors = capture.calls;
    auto cpu_input = at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat));
    auto status = celleratorch::run_program_forward(
        &program, cpu_input, output, launch, &result);
    require(status.code == celleratorch::program_op_status_code::tensor_not_cuda,
        "CPU inputs must be rejected");

    auto half_input = at::empty({2, 3}, options.dtype(at::kHalf));
    status = celleratorch::run_program_forward(
        &program, half_input, output, launch, &result);
    require(status.code == celleratorch::program_op_status_code::dtype_mismatch,
        "incompatible dtype must be rejected");

    auto wrong_rank = at::empty({6}, options);
    status = celleratorch::run_program_forward(
        &program, wrong_rank, output, launch, &result);
    require(status.code == celleratorch::program_op_status_code::rank_mismatch,
        "incompatible rank must be rejected");

    auto wrong_shape = at::empty({3, 2}, options);
    status = celleratorch::run_program_forward(
        &program, wrong_shape, output, launch, &result);
    require(status.code == celleratorch::program_op_status_code::shape_mismatch,
        "incompatible dimensions must be rejected");

    auto wrong_stride = input.as_strided({2, 3}, {1, 2});
    status = celleratorch::run_program_forward(
        &program, wrong_stride, output, launch, &result);
    require(status.code == celleratorch::program_op_status_code::stride_mismatch,
        "unsupported stride must be rejected without conversion");

    auto wrong_device_binding = native_input;
    wrong_device_binding.storage.dense.location.device_ordinal = 1;
    auto wrong_device_launch = launch;
    wrong_device_launch.bindings.inputs = &wrong_device_binding;
    status = celleratorch::run_program_forward(
        &program, input, output, wrong_device_launch, &result);
    require(status.code == celleratorch::program_op_status_code::device_mismatch,
        "native and Torch device mismatch must be rejected");

    status = celleratorch::run_program_forward(
        nullptr, input, output, launch, &result);
    require(status.code == celleratorch::program_op_status_code::invalid_argument,
        "missing native program must be rejected");
    require(capture.calls == calls_before_errors,
        "adapter validation failures must not enter native execution");

    capture.next_status.code =
        execution::executable_program_status_code::stale_or_unready_value;
    capture.next_status.message = "stale or unready value generation";
    status = celleratorch::run_program_forward(
        &program, input, output, launch, &result);
    require(status.code == celleratorch::program_op_status_code::native_failure
            && status.native.code
                == execution::executable_program_status_code::stale_or_unready_value,
        "native readiness failures must propagate without reinterpretation");

    capture.next_status.code =
        execution::executable_program_status_code::invalid_launch;
    capture.next_status.message = "unsupported operation width";
    status = celleratorch::run_program_forward(
        &program, input, output, launch, &result);
    require(status.code == celleratorch::program_op_status_code::native_failure
            && status.native.code
                == execution::executable_program_status_code::invalid_launch,
        "unsupported native launch contracts must propagate unchanged");

    std::cout << "CelleraTorch program op tests passed\n";
    return 0;
}
