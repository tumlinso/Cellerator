#include <Cellerator/compute/operation/native_numeric/device_linear.hh>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using cellerator::compute::native_numeric::device_representation;
using cellerator::compute::native_numeric::resident_vector;
using cellerator::compute::native_numeric::enqueue_axpby;
using cellerator::compute::native_numeric::enqueue_elementwise_multiply;

namespace {
int fail(const char* message) {
    std::fprintf(stderr, "ceNativeArithmeticTest: %s\n", message);
    return 1;
}

bool same_generation(const resident_vector& actual,
                     const resident_vector& before) {
    return actual.generation.value == before.generation.value;
}

bool close(float actual, float expected) {
    return std::abs(actual - expected) <= 1e-6f;
}
}

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return fail("cudaGetDevice failed");

    cudaStream_t stream{};
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        return fail("could not create nondefault stream");

    constexpr std::uint64_t count = 6;
    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_x), count * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_y), count * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), count * sizeof(float)) != cudaSuccess)
        return fail("device allocation failed");

    resident_vector x{device_x, count, device_representation::f32, device, {41}};
    resident_vector y{device_y, count, device_representation::f32, device, {42}};
    resident_vector output{device_output, count, device_representation::f32, device, {43}};
    const auto x_before = x;
    const auto y_before = y;
    const auto output_before = output;

    const float host_x[count] = {-0.0f, 0.0f, 2.0f, -3.0f, 4.0f, 5.0f};
    const float host_y[count] = {2.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f};
    const float output_sentinel[count] = {-17.0f, -18.0f, -19.0f, -20.0f, -21.0f, -22.0f};
    float host_result[count]{};
    // The nondefault stream orders input copies, both kernels, and the readback.
    if (cudaMemcpyAsync(device_x, host_x, sizeof(host_x), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(device_y, host_y, sizeof(host_y), cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return fail("input upload failed");
    if (enqueue_elementwise_multiply(x, y, output, stream) != cudaSuccess)
        return fail("multiply enqueue failed");
    if (cudaMemcpyAsync(host_result, device_output, sizeof(host_result), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return fail("multiply readback failed");
    if (!std::signbit(host_result[0]) || std::signbit(host_result[1]) ||
        !close(host_result[2], 6.0f) || !close(host_result[3], -12.0f) ||
        !close(host_result[4], 2.0f) || !close(host_result[5], -5.0f))
        return fail("multiply values or signed-zero behavior differ");

    // Repeated inputs are legal; this also checks a constant input vector.
    const float host_constant[count] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    if (cudaMemcpyAsync(device_x, host_constant, sizeof(host_constant), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        enqueue_elementwise_multiply(x, x, output, stream) != cudaSuccess ||
        cudaMemcpyAsync(host_result, device_output, sizeof(host_result), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return fail("repeated-input multiply failed");
    for (float value : host_result) if (!close(value, 4.0f))
        return fail("constant repeated-input result differs");

    if (enqueue_axpby(2.0f, x, -0.5f, x, output, stream) != cudaSuccess ||
        cudaMemcpyAsync(host_result, device_output, sizeof(host_result), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return fail("axpby enqueue/readback failed");
    for (float value : host_result) if (!close(value, 3.0f))
        return fail("axpby constant result differs");

    if (cudaMemcpyAsync(device_output, output_sentinel, sizeof(output_sentinel),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return fail("output sentinel setup failed");

    resident_vector bad = x;
    bad.representation = device_representation::f16;
    if (enqueue_elementwise_multiply(bad, y, output, stream) != cudaErrorInvalidValue)
        return fail("non-FP32 input was not rejected");
    bad = x;
    bad.elements = count - 1;
    if (enqueue_elementwise_multiply(bad, y, output, stream) != cudaErrorInvalidValue)
        return fail("capacity mismatch was not rejected");
    bad = x;
    bad.device_ordinal = device + 1;
    if (enqueue_elementwise_multiply(bad, y, output, stream) != cudaErrorInvalidDevice)
        return fail("device metadata mismatch was not rejected");
    bad = x;
    bad.data = nullptr;
    if (enqueue_elementwise_multiply(bad, y, output, stream) != cudaErrorInvalidDevicePointer)
        return fail("null nonempty pointer was not rejected");
    bad = x;
    bad.data = const_cast<float*>(host_x);
    const auto host_pointer_error = enqueue_elementwise_multiply(bad, y, output, stream);
    if (host_pointer_error != cudaErrorInvalidValue &&
        host_pointer_error != cudaErrorInvalidDevicePointer)
        return fail("host pointer was not rejected");
    bad = x;
    bad.data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 3);
    if (enqueue_elementwise_multiply(bad, y, output, stream) != cudaErrorInvalidDevicePointer)
        return fail("wrapping pointer range was not rejected");
    if (enqueue_elementwise_multiply(x, y, x, stream) != cudaErrorInvalidValue)
        return fail("exact output/input alias was not rejected");
    resident_vector x_prefix = x;
    resident_vector y_prefix = y;
    resident_vector partial = output;
    x_prefix.elements = count - 1;
    y_prefix.elements = count - 1;
    partial.data = static_cast<char*>(x.data) + sizeof(float);
    partial.elements = count - 1;
    if (enqueue_elementwise_multiply(x_prefix, y_prefix, partial, stream) != cudaErrorInvalidValue)
        return fail("partial output/input alias was not rejected");

    resident_vector overflow{nullptr, std::numeric_limits<std::uint64_t>::max(),
                             device_representation::f32, device, {0}};
    if (enqueue_elementwise_multiply(overflow, overflow, overflow, stream) != cudaErrorInvalidValue)
        return fail("element-count arithmetic overflow was not rejected");
    resident_vector zero{nullptr, 0, device_representation::f32, device, {0}};
    if (enqueue_elementwise_multiply(zero, zero, zero, stream) != cudaSuccess)
        return fail("valid zero-length operation was not a no-op");
    float input_after[count]{};
    if (cudaMemcpyAsync(host_result, device_output, sizeof(host_result), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(input_after, device_x, sizeof(input_after), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return fail("rejection-preservation readback failed");
    for (std::uint64_t i = 0; i < count; ++i) {
        if (!close(host_result[i], output_sentinel[i]))
            return fail("rejected operation changed output storage");
        if (!close(input_after[i], 2.0f))
            return fail("rejected operation changed input storage");
    }

    if (!same_generation(x, x_before) || !same_generation(y, y_before) ||
        !same_generation(output, output_before))
        return fail("arithmetic changed value generations");

    cudaError_t cleanup = cudaSuccess;
    if (cudaFree(device_x) != cudaSuccess) cleanup = cudaErrorUnknown;
    if (cudaFree(device_y) != cudaSuccess) cleanup = cudaErrorUnknown;
    if (cudaFree(device_output) != cudaSuccess) cleanup = cudaErrorUnknown;
    if (cudaStreamDestroy(stream) != cudaSuccess) cleanup = cudaErrorUnknown;
    if (cleanup != cudaSuccess) return fail("CUDA cleanup failed");
    return 0;
}
