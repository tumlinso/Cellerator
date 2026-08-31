#include <Cellerator/compute/operation/fusion/fusion_kernels_v1.cuh>

#include <limits>

namespace cellerator::compute::operation::fusion {
namespace {

enum class residual_mode : std::uint8_t { mma_only, residual_add, fused };

template<residual_mode Mode>
__global__ void mma_residual_kernel(mma_residual_request_v1 request) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t index = first; index < request.local_output_count;
        index += stride) {
        if constexpr (Mode == residual_mode::mma_only)
            request.output[index] = request.mma_contribution[index];
        else if constexpr (Mode == residual_mode::residual_add)
            request.output[index] += request.same_owner_residual[index];
        else
            request.output[index] = request.mma_contribution[index]
                + request.same_owner_residual[index];
    }
}

template<residual_mode Mode>
status_v1 enqueue(mma_residual_request_v1 request) noexcept {
    constexpr std::uint32_t threads = 256u;
    constexpr std::uint32_t maximum_blocks = 65535u;
    const std::uint32_t required =
        (request.local_output_count + threads - 1u) / threads;
    mma_residual_kernel<Mode><<<
        required < maximum_blocks ? required : maximum_blocks, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

} // namespace

status_v1 validate_mma_residual_request_v1(
    const mma_residual_request_v1 &request) noexcept {
    if (request.mma_contribution == nullptr
        || request.same_owner_residual == nullptr || request.output == nullptr
        || request.local_output_count == 0u || request.owner_order_id == 0u
        || request.structure_epoch == 0u || request.value_generation == 0u
        || request.global_output_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.local_output_count)
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_mma_contribution_unfused_v1(
    const mma_residual_request_v1 &request) noexcept {
    if (validate_mma_residual_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    return enqueue<residual_mode::mma_only>(request);
}

status_v1 enqueue_same_owner_residual_unfused_v1(
    const mma_residual_request_v1 &request) noexcept {
    if (validate_mma_residual_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    return enqueue<residual_mode::residual_add>(request);
}

status_v1 enqueue_mma_same_owner_residual_fused_v1(
    const mma_residual_request_v1 &request) noexcept {
    if (validate_mma_residual_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    return enqueue<residual_mode::fused>(request);
}

} // namespace cellerator::compute::operation::fusion
