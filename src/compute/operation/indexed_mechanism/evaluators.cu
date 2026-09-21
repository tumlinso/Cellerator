#include <Cellerator/compute/operation/indexed_mechanism/evaluators.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace cellerator::compute::operation::indexed {
namespace {
__device__ float evaluate(evaluator_opcode opcode, const float* values, std::size_t count) {
    if (opcode == evaluator_opcode::sum) {
        float result = 0.0f;
        for (std::size_t i = 0; i < count; ++i) result += values[i];
        return result;
    }
    if (opcode == evaluator_opcode::product) {
        float result = 1.0f;
        for (std::size_t i = 0; i < count; ++i) result *= values[i];
        return result;
    }
    float result = 1.0f;
    for (std::size_t i = 1; i < count; ++i) result *= values[i];
    return values[0] - result;
}
__global__ void evaluate_f32_kernel(evaluator_opcode opcode, bool predicate,
                                    const float* values, std::size_t count, float* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && predicate) *output = evaluate(opcode, values, count);
}
__global__ void evaluate_f16_kernel(evaluator_opcode opcode, bool predicate,
                                    const std::uint16_t* values, std::size_t count, std::uint16_t* output) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || !predicate) return;
    if (opcode == evaluator_opcode::sum) {
        float result = 0.0f;
        for (std::size_t i = 0; i < count; ++i) result += __half2float(reinterpret_cast<const __half*>(values)[i]);
        reinterpret_cast<__half*>(output)[0] = __float2half_rn(result);
    } else if (opcode == evaluator_opcode::product) {
        float result = 1.0f;
        for (std::size_t i = 0; i < count; ++i) result *= __half2float(reinterpret_cast<const __half*>(values)[i]);
        reinterpret_cast<__half*>(output)[0] = __float2half_rn(result);
    } else {
        float result = 1.0f;
        for (std::size_t i = 1; i < count; ++i) result *= __half2float(reinterpret_cast<const __half*>(values)[i]);
        reinterpret_cast<__half*>(output)[0] = __float2half_rn(__half2float(reinterpret_cast<const __half*>(values)[0]) - result);
    }
}
bool valid(const registered_block& block, std::size_t count) noexcept {
    return v2::valid_stable_id(block.evaluator) && block.dependencies_known
        && block.writes_only_declared_outputs && block.deterministic
        && block.opcode >= evaluator_opcode::sum && block.opcode <= evaluator_opcode::first_minus_product_tail
        && count >= block.minimum_arguments
        && (block.opcode != evaluator_opcode::first_minus_product_tail || count >= 3);
}
evaluation_status result(cudaError_t status) noexcept {
    return status == cudaSuccess ? evaluation_status::success : evaluation_status::cuda_error;
}
}

evaluation_status evaluate_cuda_f32(const registered_block& block, bool predicate,
                                    const float* arguments, std::size_t argument_count,
                                    float* destination, void* stream) noexcept {
    if (!valid(block, argument_count)) return evaluation_status::invalid_arity;
    if (!destination || (predicate && !arguments)) return evaluation_status::invalid_binding;
    evaluate_f32_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(block.opcode, predicate, arguments, argument_count, destination);
    return result(cudaPeekAtLastError());
}
evaluation_status evaluate_cuda_f16(const registered_block& block, bool predicate,
                                    const std::uint16_t* arguments, std::size_t argument_count,
                                    std::uint16_t* destination, void* stream) noexcept {
    if (!valid(block, argument_count)) return evaluation_status::invalid_arity;
    if (!destination || (predicate && !arguments)) return evaluation_status::invalid_binding;
    evaluate_f16_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(block.opcode, predicate, arguments, argument_count, destination);
    return result(cudaPeekAtLastError());
}
} // namespace cellerator::compute::operation::indexed
