/*
 * Generic unfused epilogue correctness baseline (CP-MATH-02, 2026-08-18).
 * Reference: CPU bias/ReLU/exact-erf-GELU/tanh-GELU formulas in
 * math_runtime_test.cu. Target: Tesla V100, sm_70. Shapes: 2x3 row-major and
 * column-major f32 outputs. Command: build-cp-math-context/cpMathRuntimeTest.
 * Tolerances: 2e-6 exact paths, 2e-5 tanh approximation. This is not a
 * performance-selected replacement for a vendor-fused epilogue; future SpMM
 * backends should fuse when their vendor API supports the requested semantics.
 */

#include <Cellerator/compute/math/backend.hh>
#include <Cellerator/types.cuh>

#include <cmath>
#include <limits>

namespace cellerator::compute::math {

namespace {

__host__ __device__ constexpr bool has_bias(epilogue_kind kind) noexcept {
    return kind == epilogue_kind::bias || kind == epilogue_kind::bias_relu
        || kind == epilogue_kind::bias_gelu_exact_erf
        || kind == epilogue_kind::bias_gelu_tanh_approximate;
}

__host__ __device__ constexpr bool has_relu(epilogue_kind kind) noexcept {
    return kind == epilogue_kind::relu || kind == epilogue_kind::bias_relu;
}

__host__ __device__ constexpr bool has_exact_gelu(epilogue_kind kind) noexcept {
    return kind == epilogue_kind::gelu_exact_erf
        || kind == epilogue_kind::bias_gelu_exact_erf;
}

__host__ __device__ constexpr bool has_tanh_gelu(epilogue_kind kind) noexcept {
    return kind == epilogue_kind::gelu_tanh_approximate
        || kind == epilogue_kind::bias_gelu_tanh_approximate;
}

__device__ float apply_epilogue(float value, float bias, epilogue_kind kind) {
    if (has_bias(kind)) value += bias;
    if (has_relu(kind)) return fmaxf(value, 0.0f);
    if (has_exact_gelu(kind)) {
        return 0.5f * value * (1.0f + erff(value * 0.7071067811865475244f));
    }
    if (has_tanh_gelu(kind)) {
        const float cubic = value * value * value;
        return 0.5f * value
            * (1.0f + tanhf(0.7978845608028653559f
                * (value + 0.044715f * cubic)));
    }
    return value;
}

__global__ void generic_epilogue_f32(
    float *output,
    const float *bias,
    u64 rows,
    u64 columns,
    u64 leading_dimension,
    dense_layout_kind layout,
    epilogue_kind kind) {
    const u64 count = rows * columns;
    for (u64 logical = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         logical < count;
         logical += static_cast<u64>(blockDim.x) * gridDim.x) {
        const u64 row = logical / columns;
        const u64 column = logical - row * columns;
        const u64 physical = layout == dense_layout_kind::row_major
            ? row * leading_dimension + column
            : column * leading_dimension + row;
        const float bias_value = bias == nullptr ? 0.0f : bias[column];
        output[physical] = apply_epilogue(output[physical], bias_value, kind);
    }
}

backend_status cuda_failure(cudaError_t error, const char *message) noexcept {
    return {backend_status_code::runtime_failure,
        capability_code::supported,
        request_validation_code::ok,
        error,
        message};
}

} // namespace

backend_capability query_generic_unfused_epilogue_capability(
    const spmm_request &request) noexcept {
    const request_validation_result validation = validate_spmm_request(request);
    if (!validation) {
        return {capability_code::invalid_request,
            validation.code,
            validation.message};
    }
    if (request.output_storage_type_code != static_cast<u32>(real::value_f32)
        || (has_bias(request.epilogue.kind)
            && request.epilogue.bias_type_code
                != static_cast<u32>(real::value_f32))) {
        return {capability_code::unsupported_type,
            request_validation_code::ok,
            "generic epilogue supports f32 output and bias only"};
    }
    backend_capability result;
    result.epilogue_strategy = request.epilogue.kind == epilogue_kind::none
        ? epilogue_strategy_kind::none
        : epilogue_strategy_kind::generic_unfused;
    result.algorithm_identity = 0x4550494c4f475545ull;
    result.kernel_variant_identity = 0x4633320000000001ull;
    return result;
}

backend_status launch_generic_unfused_epilogue(
    DeviceMathContext *context,
    const spmm_request &request,
    const spmm_bindings &bindings) noexcept {
    if (context == nullptr || !context->initialized) {
        return {backend_status_code::invalid_argument,
            capability_code::supported,
            request_validation_code::ok,
            cudaSuccess,
            "generic epilogue requires an initialized device context"};
    }
    const backend_capability capability =
        query_generic_unfused_epilogue_capability(request);
    if (!capability) return detail::capability_failure(capability);
    if (request.epilogue.kind == epilogue_kind::none
        || request.m == 0u || request.n == 0u) {
        return {};
    }
    if (bindings.output == nullptr
        || (has_bias(request.epilogue.kind) && bindings.bias == nullptr)) {
        return {backend_status_code::invalid_argument,
            capability_code::supported,
            request_validation_code::missing_binding,
            cudaSuccess,
            "generic epilogue bindings are incomplete"};
    }
    if (request.m > std::numeric_limits<u64>::max() / request.n) {
        return {backend_status_code::invalid_argument,
            capability_code::supported,
            request_validation_code::invalid_shape,
            cudaSuccess,
            "generic epilogue element count overflows"};
    }

    constexpr unsigned threads = 256u;
    const u64 count = request.m * request.n;
    const u64 required_blocks = (count + threads - 1u) / threads;
    const unsigned blocks = static_cast<unsigned>(
        required_blocks < 65535u ? required_blocks : 65535u);
    generic_epilogue_f32<<<blocks, threads, 0u, context->execution.stream>>>(
        static_cast<float *>(bindings.output),
        static_cast<const float *>(bindings.bias),
        request.m,
        request.n,
        request.output_leading_dimension,
        request.output_layout,
        request.epilogue.kind);
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess
        ? backend_status{}
        : cuda_failure(error, "generic epilogue launch failed");
}

} // namespace cellerator::compute::math
