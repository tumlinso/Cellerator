#include <Cellerator/compute/operation/edge/dynamic_support_mask_v1.cuh>

#include <limits>

namespace cellerator::compute::operation::edge {
namespace {

__device__ bool active(const std::uint8_t *mask, std::uint32_t edge,
    mask_encoding_v1 encoding) {
    return encoding == mask_encoding_v1::byte_per_edge
        ? mask[edge] != 0u
        : (mask[edge / 8u] & static_cast<std::uint8_t>(1u << (edge % 8u)))
            != 0u;
}

__global__ void dynamic_support_mask_kernel(
    dynamic_support_mask_request_v1 request) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t edge = first;
        edge < request.stable_support_superset.local_edge_count;
        edge += stride)
        request.output[edge] = active(request.active_mask, edge,
            request.encoding) ? request.input[edge] : 0.0f;
}

} // namespace

status_v1 validate_dynamic_support_mask_request_v1(
    const dynamic_support_mask_request_v1 &request) noexcept {
    if (request.stable_support_superset.local_edge_count == 0u
        || request.input == nullptr || request.output == nullptr
        || request.active_mask == nullptr || request.structure_id == 0u
        || request.structure_epoch == 0u
        || request.input_value_generation == 0u
        || request.mask_value_generation == 0u
        || request.output_value_generation == 0u
        || request.stable_support_superset.global_edge_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.stable_support_superset.local_edge_count)
        return status_v1::invalid_argument;
    if (request.encoding > mask_encoding_v1::packed_lsb_bits)
        return status_v1::unsupported;
    return status_v1::success;
}

status_v1 enqueue_dynamic_support_mask_v1(
    const dynamic_support_mask_request_v1 &request) noexcept {
    const status_v1 validation = validate_dynamic_support_mask_request_v1(
        request);
    if (validation != status_v1::success) return validation;
    constexpr std::uint32_t threads = 256u;
    constexpr std::uint32_t maximum_blocks = 65535u;
    const std::uint32_t required =
        (request.stable_support_superset.local_edge_count + threads - 1u)
        / threads;
    dynamic_support_mask_kernel<<<
        required < maximum_blocks ? required : maximum_blocks, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess
        ? status_v1::success : status_v1::cuda_failure;
}

} // namespace cellerator::compute::operation::edge
