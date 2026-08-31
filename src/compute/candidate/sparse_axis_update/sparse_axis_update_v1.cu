#include <Cellerator/compute/operation/sparse_axis_update/sparse_axis_update_v1.cuh>

#include <cmath>
#include <limits>

namespace cellerator::compute::operation::sparse_axis_update {
namespace {

__device__ std::uint64_t axis_index(const request_v1 &request,
    std::uint32_t update) {
    if (request.index_kind == index_kind_v1::local_u32)
        return static_cast<const std::uint32_t *>(request.indices)[update];
    return static_cast<const std::uint64_t *>(request.indices)[update]
        - request.target_slice.global_axis_begin;
}

__device__ float apply_update(float current, float update,
    operation_v1 operation) {
    switch (operation) {
        case operation_v1::assign: return update;
        case operation_v1::add: return current + update;
        case operation_v1::subtract: return current - update;
        case operation_v1::multiply: return current * update;
        case operation_v1::maximum: return fmaxf(current, update);
    }
    return current;
}

__global__ void sparse_axis_update_kernel(request_v1 request) {
    const std::uint32_t first_update = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t update = first_update; update < request.update_count;
        update += stride) {
        const std::uint64_t axis = axis_index(request, update);
        if (axis >= request.target_slice.local_axis_count) continue;
        for (std::uint32_t component = 0u;
            component < request.target_slice.component_count; ++component) {
            const std::size_t target_position =
                static_cast<std::size_t>(axis)
                    * request.target_slice.component_count + component;
            const std::size_t update_position =
                static_cast<std::size_t>(update)
                    * request.target_slice.component_count + component;
            request.target[target_position] = apply_update(
                request.target[target_position], request.updates[update_position],
                request.operation);
        }
    }
}

} // namespace

status_v1 validate_request_v1(const request_v1 &request) noexcept {
    if (request.target_slice.local_axis_count == 0u
        || request.target_slice.component_count == 0u
        || request.target == nullptr || request.indices == nullptr
        || request.updates == nullptr || request.update_count == 0u
        || !request.indices_are_unique || request.target_axis_id == 0u
        || request.structure_epoch == 0u
        || request.input_value_generation == 0u
        || request.output_value_generation == 0u
        || request.target_slice.global_axis_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.target_slice.local_axis_count)
        return status_v1::invalid_argument;
    if (request.operation > operation_v1::maximum
        || request.index_kind > index_kind_v1::global_u64)
        return status_v1::unsupported;
    if (request.reserved[0] != 0u || request.reserved[1] != 0u)
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_v1(const request_v1 &request) noexcept {
    const status_v1 validation = validate_request_v1(request);
    if (validation != status_v1::success) return validation;
    constexpr std::uint32_t threads = 256u;
    constexpr std::uint32_t maximum_blocks = 65535u;
    const std::uint32_t required =
        (request.update_count + threads - 1u) / threads;
    sparse_axis_update_kernel<<<
        required < maximum_blocks ? required : maximum_blocks, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess
        ? status_v1::success : status_v1::cuda_failure;
}

} // namespace cellerator::compute::operation::sparse_axis_update
