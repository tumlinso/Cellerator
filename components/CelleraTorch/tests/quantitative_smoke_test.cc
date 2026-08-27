#define main celleratorch_embedded_quantitative_forward_main
#define run_executable_program celleratorch_quantitative_run_executable_program
#include "../../../tests/live/quantitative_forward_test.cu"
#undef run_executable_program
#undef main

// Keep the quantitative fixture implementation in this scoped validation
// translation unit; shared library/build ownership remains with CE-LIVE-43.
#include "../../../bench/ce_live/runtime_fixture/quantitative_fixture.cc"

#include <CelleraTorch/program_ops.hh>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct adapter_measurements {
    std::vector<float> native_milliseconds;
    std::vector<float> adapter_milliseconds;
    std::vector<float> native_enqueue_nanoseconds;
    std::vector<float> adapter_dispatch_nanoseconds;
    std::vector<float> view_adaptation_nanoseconds;
    std::uint64_t parity_checks = 0u;
} measurements;

float event_elapsed(cudaEvent_t begin, cudaEvent_t end) {
    float milliseconds = 0.0f;
    if (cudaEventElapsedTime(&milliseconds, begin, end) != cudaSuccess)
        return -1.0f;
    return milliseconds;
}

float sample_median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

cellerator::execution::executable_program_status failure(
    cellerator::execution::executable_program_status_code code,
    const char *message) {
    cellerator::execution::executable_program_status status{};
    status.code = code;
    status.message = message;
    return status;
}

torch::Tensor alias_dense(
    const cellerator::execution::dense_tensor_view &view) {
    std::vector<std::int64_t> sizes(view.rank);
    std::vector<std::int64_t> strides(view.rank);
    for (std::uint8_t axis = 0u; axis < view.rank; ++axis) {
        sizes[axis] = static_cast<std::int64_t>(view.shape[axis]);
        strides[axis] = view.stride[axis];
    }
    return torch::from_blob(view.data, sizes, strides, [](void *) {},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::Device(torch::kCUDA,
                view.location.device_ordinal))
            .requires_grad(false));
}

} // namespace

namespace cellerator::execution {

executable_program_status run_executable_program(
    executable_program *program,
    const executable_program_launch &launch,
    executable_program_result *result) noexcept;

// The embedded native quantitative test calls this instrumentation seam.  It
// runs the exact native program and then the CelleraTorch wrapper over the same
// prepared program, fixture, value generation, buffers, and caller stream.
executable_program_status celleratorch_quantitative_run_executable_program(
    executable_program *program,
    const executable_program_launch &launch,
    executable_program_result *result) noexcept {
    if (program == nullptr || result == nullptr
        || launch.bindings.input_count != 1u
        || launch.bindings.output_count != 1u)
        return failure(executable_program_status_code::invalid_launch,
            "quantitative adapter instrumentation requires one input/output");

    cudaEvent_t native_begin = nullptr, native_end = nullptr;
    cudaEvent_t adapter_begin = nullptr, adapter_end = nullptr;
    const cudaStream_t stream = static_cast<cudaStream_t>(
        launch.bindings.stream.stream);
    if (cudaEventCreate(&native_begin) != cudaSuccess
        || cudaEventCreate(&native_end) != cudaSuccess
        || cudaEventCreate(&adapter_begin) != cudaSuccess
        || cudaEventCreate(&adapter_end) != cudaSuccess)
        return failure(executable_program_status_code::execution_failed,
            "quantitative adapter timing event creation failed");

    executable_program_result native_result{};
    (void)cudaEventRecord(native_begin, stream);
    const auto native_host_begin = std::chrono::steady_clock::now();
    const auto native_status = run_executable_program(
        program, launch, &native_result);
    const auto native_host_end = std::chrono::steady_clock::now();
    (void)cudaEventRecord(native_end, stream);
    if (!native_status || cudaEventSynchronize(native_end) != cudaSuccess)
        return native_status;
    measurements.native_milliseconds.push_back(
        event_elapsed(native_begin, native_end));
    measurements.native_enqueue_nanoseconds.push_back(static_cast<float>(
        std::chrono::duration<double, std::nano>(
            native_host_end - native_host_begin).count()));

    // The native baseline is an extra validation execution.  Keep the public
    // preparation/run counter observed by the embedded test scoped to the
    // adapter executions under test.
    if (program->run_count != 0u) --program->run_count;

    const auto &native_output = launch.bindings.outputs[0].storage.dense;
    const std::size_t element_count = static_cast<std::size_t>(
        native_output.shape[0] * native_output.shape[1]);
    std::vector<float> expected(element_count);
    if (cudaMemcpy(expected.data(), native_output.data,
            element_count * sizeof(float), cudaMemcpyDeviceToHost)
        != cudaSuccess)
        return failure(executable_program_status_code::execution_failed,
            "native quantitative output copy failed");

    const auto adapt_begin = std::chrono::steady_clock::now();
    torch::Tensor input = alias_dense(
        launch.bindings.inputs[0].storage.dense);
    torch::Tensor output = alias_dense(native_output);
    const auto adapt_end = std::chrono::steady_clock::now();
    measurements.view_adaptation_nanoseconds.push_back(static_cast<float>(
        std::chrono::duration<double, std::nano>(
            adapt_end - adapt_begin).count()));

    const auto torch_stream = c10::cuda::getStreamFromExternal(
        stream, launch.bindings.stream.device_ordinal);
    c10::cuda::CUDAStreamGuard guard(torch_stream);
    (void)cudaEventRecord(adapter_begin, stream);
    const auto adapter_host_begin = std::chrono::steady_clock::now();
    const auto adapter_status = celleratorch::run_program_forward(
        program, input, output, launch, result);
    const auto adapter_host_end = std::chrono::steady_clock::now();
    (void)cudaEventRecord(adapter_end, stream);
    if (!adapter_status || cudaEventSynchronize(adapter_end) != cudaSuccess) {
        if (adapter_status.native.code
            != executable_program_status_code::ok)
            return adapter_status.native;
        return failure(executable_program_status_code::execution_failed,
            adapter_status.message);
    }
    measurements.adapter_milliseconds.push_back(
        event_elapsed(adapter_begin, adapter_end));
    measurements.adapter_dispatch_nanoseconds.push_back(static_cast<float>(
        std::chrono::duration<double, std::nano>(
            adapter_host_end - adapter_host_begin).count()));

    std::vector<float> actual(element_count);
    if (cudaMemcpy(actual.data(), native_output.data,
            element_count * sizeof(float), cudaMemcpyDeviceToHost)
        != cudaSuccess)
        return failure(executable_program_status_code::execution_failed,
            "CelleraTorch quantitative output copy failed");
    for (std::size_t index = 0u; index < element_count; ++index) {
        const float tolerance = 1.0e-6f
            * std::max(1.0f, std::fabs(expected[index]));
        if (std::fabs(actual[index] - expected[index]) > tolerance)
            return failure(executable_program_status_code::execution_failed,
                "CelleraTorch output differs from native Cellerator");
    }
    ++measurements.parity_checks;

    (void)cudaEventDestroy(adapter_end);
    (void)cudaEventDestroy(adapter_begin);
    (void)cudaEventDestroy(native_end);
    (void)cudaEventDestroy(native_begin);
    return {};
}

} // namespace cellerator::execution

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: celleraTorchQuantitativeSmokeTest FIXTURE_BIN\n";
        return 2;
    }
    char repeats[] = "3";
    char *embedded_argv[]{argv[0], argv[1], repeats};
    const int result = celleratorch_embedded_quantitative_forward_main(
        3, embedded_argv);
    if (result != 0) return result;
    if (measurements.native_milliseconds.empty()
        || measurements.native_milliseconds.size()
            != measurements.adapter_milliseconds.size()
        || measurements.parity_checks
            != measurements.adapter_milliseconds.size())
        return 3;

    const float native_ms = sample_median(
        measurements.native_milliseconds);
    const float adapter_ms = sample_median(
        measurements.adapter_milliseconds);
    const float view_ns = sample_median(
        measurements.view_adaptation_nanoseconds);
    const float native_enqueue_ns = sample_median(
        measurements.native_enqueue_nanoseconds);
    const float adapter_dispatch_ns = sample_median(
        measurements.adapter_dispatch_nanoseconds);
    std::cout << "{\"ce_live\":44,\"parity_checks\":"
              << measurements.parity_checks
              << ",\"native_median_us\":" << native_ms * 1000.0f
              << ",\"adapter_median_us\":" << adapter_ms * 1000.0f
              << ",\"dispatch_tax_us\":"
              << (adapter_ms - native_ms) * 1000.0f
              << ",\"native_enqueue_median_ns\":" << native_enqueue_ns
              << ",\"adapter_dispatch_median_ns\":" << adapter_dispatch_ns
              << ",\"host_dispatch_tax_ns\":"
              << adapter_dispatch_ns - native_enqueue_ns
              << ",\"view_adaptation_median_ns\":" << view_ns
              << ",\"fixture\":\"pbmc3k-r512-s7\"}\n";
    return 0;
}
