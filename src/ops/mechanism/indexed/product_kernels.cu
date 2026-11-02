#include "indexed_product_cuda.cuh"
#include "indexed_cuda_support.cuh"

#include <cuda_fp16.h>

namespace cellerator::ops::mechanism::indexed::detail {
namespace {

__device__ float read_x(const void* x, bool half, std::uint64_t i) {
    return half ? __half2float(static_cast<const __half*>(x)[i])
                : static_cast<const float*>(x)[i];
}

__global__ void save_input(const void* input, bool input_half, void* saved,
                           bool mixed, std::uint64_t count) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const float value = read_x(input, input_half, i);
        if (mixed) static_cast<__half*>(saved)[i] = __float2half_rn(value);
        else static_cast<float*>(saved)[i] = value;
    }
}

__global__ void products(device_plan p, const void* x, bool mixed,
                         std::uint64_t batch, float* result) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < batch * p.m; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const auto m = i % p.m;
        const auto b = i / p.m;
        float product = 1;
        for (auto a = p.arg_offsets[m]; a < p.arg_offsets[m + 1]; ++a)
            product *= read_x(x, mixed, b * p.n + p.args[a]);
        result[i] = product;
    }
}

__global__ void assemble(device_plan p, const float* products,
                         const void* coefficients, bool mixed,
                         std::uint64_t batch, float* output) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < batch * p.o; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const auto o = i % p.o;
        const auto b = i / p.o;
        float sum = 0;
        for (auto j = p.out_offsets[o]; j < p.out_offsets[o + 1]; ++j) {
            const auto m = p.out_mechanism[j];
            sum += p.out_scale[j] * read_x(coefficients, mixed, p.coefficient[m])
                 * products[b * p.m + m];
        }
        output[i] = sum;
    }
}

__global__ void local_vjp(device_plan p, const void* x, const void* coefficients,
                          bool mixed, const float* dy, std::uint64_t batch,
                          float* da, float* dm) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < batch * p.m; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const auto m = i % p.m;
        const auto b = i / p.m;
        float upstream = 0;
        for (auto j = p.mechanism_out_offsets[m]; j < p.mechanism_out_offsets[m + 1]; ++j)
            upstream += p.mechanism_scales[j] * dy[b * p.o + p.mechanism_outputs[j]];
        dm[i] = upstream;
        float prefix = 1;
        for (auto a = p.arg_offsets[m]; a < p.arg_offsets[m + 1]; ++a) {
            da[b * p.a + a] = prefix;
            prefix *= read_x(x, mixed, b * p.n + p.args[a]);
        }
        float suffix = 1;
        const float weighted = upstream * read_x(coefficients, mixed, p.coefficient[m]);
        for (auto a = p.arg_offsets[m + 1]; a > p.arg_offsets[m];) {
            --a;
            da[b * p.a + a] *= suffix * weighted;
            suffix *= read_x(x, mixed, b * p.n + p.args[a]);
        }
    }
}

__global__ void input_vjp(device_plan p, const float* da,
                          std::uint64_t batch, float* dx) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < batch * p.n; i += std::uint64_t(blockDim.x) * gridDim.x) {
        const auto n = i % p.n;
        const auto b = i / p.n;
        float sum = 0;
        for (auto j = p.input_offsets[n]; j < p.input_offsets[n + 1]; ++j)
            sum += da[b * p.a + p.input_slots[j]];
        dx[i] = sum;
    }
}

__global__ void coefficient_vjp(device_plan p, const float* products,
                                const float* dm, std::uint64_t batch, float* dk) {
    for (auto k = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < p.k; k += std::uint64_t(blockDim.x) * gridDim.x) {
        float sum = 0;
        for (std::uint64_t b = 0; b < batch; ++b)
            for (auto j = p.param_offsets[k]; j < p.param_offsets[k + 1]; ++j) {
                const auto m = p.param_mechanisms[j];
                sum += dm[b * p.m + m] * products[b * p.m + m];
            }
        dk[k] = sum;
    }
}

} // namespace

execution::program::program_status launch_training(
        const void*, const execution::program::launch_binding_v2& binding,
        void* stream_ptr) noexcept {
    const auto& b = *static_cast<const execution_binding*>(binding.values);
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    if (!b.backward) {
        save_input<<<blocks(b.batch * b.plan.n), 256, 0, stream>>>(
            b.input, b.input_half, b.saved_input, b.mixed, b.batch * b.plan.n);
        products<<<blocks(b.batch * b.plan.m), 256, 0, stream>>>(
            b.plan, b.saved_input, b.mixed, b.batch, b.products);
        assemble<<<blocks(b.batch * b.plan.o), 256, 0, stream>>>(
            b.plan, b.products, b.coefficients, b.mixed, b.batch, b.output);
    } else {
        local_vjp<<<blocks(b.batch * b.plan.m), 256, 0, stream>>>(
            b.plan, b.saved_input, b.coefficients, b.mixed,
            static_cast<const float*>(b.input), b.batch,
            b.argument_gradients, b.mechanism_gradients);
        input_vjp<<<blocks(b.batch * b.plan.n), 256, 0, stream>>>(
            b.plan, b.argument_gradients, b.batch, b.input_gradient);
        coefficient_vjp<<<blocks(b.plan.k), 256, 0, stream>>>(
            b.plan, b.products, b.mechanism_gradients, b.batch,
            b.coefficient_gradient);
    }
    return cudaGetLastError() == cudaSuccess
        ? execution::program::program_status::success
        : execution::program::program_status::launch_failed;
}

} // namespace cellerator::ops::mechanism::indexed::detail
