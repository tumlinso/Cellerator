#include "../../../src/compute/architecture/providers/nvidia/sm70/value_pack.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {

void require_cuda(cudaError_t status) {
    assert(status == cudaSuccess);
}

} // namespace

int main() {
    projection::projection_value_map_v1 host_map[3]{};
    host_map[0].logical_edge_id.value = 2u;
    host_map[0].region_kind = projection::physical_region_kind_v1::mma;
    host_map[0].region_index = 0u;
    host_map[0].projection_slot = 0u;
    host_map[1].logical_edge_id.value = 0u;
    host_map[1].region_kind = projection::physical_region_kind_v1::mma;
    host_map[1].region_index = 0u;
    host_map[1].projection_slot = 17u;
    host_map[2].logical_edge_id.value = 1u;
    host_map[2].region_kind = projection::physical_region_kind_v1::residual;
    host_map[2].region_index = 0u;
    host_map[2].projection_slot = 0u;
    const __half host_values[3] = {
        __float2half(1.5f), __float2half(-2.0f), __float2half(3.25f)};
    const std::uint64_t host_mma_offsets[] = {0u};
    const std::uint64_t host_residual_offsets[] = {0u};

    projection::projection_value_map_v1 *device_map = nullptr;
    __half *device_values = nullptr;
    std::uint64_t *device_mma_offsets = nullptr;
    std::uint64_t *device_residual_offsets = nullptr;
    __half *device_mma = nullptr;
    __half *device_residual = nullptr;
    require_cuda(cudaMalloc(&device_map, sizeof(host_map)));
    require_cuda(cudaMalloc(&device_values, sizeof(host_values)));
    require_cuda(cudaMalloc(&device_mma_offsets, sizeof(host_mma_offsets)));
    require_cuda(cudaMalloc(&device_residual_offsets,
        sizeof(host_residual_offsets)));
    require_cuda(cudaMalloc(&device_mma, 256u * sizeof(__half)));
    require_cuda(cudaMalloc(&device_residual, sizeof(__half)));

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    require_cuda(cudaMemcpyAsync(device_map, host_map, sizeof(host_map),
        cudaMemcpyHostToDevice, stream));
    require_cuda(cudaMemcpyAsync(device_values, host_values, sizeof(host_values),
        cudaMemcpyHostToDevice, stream));
    require_cuda(cudaMemcpyAsync(device_mma_offsets, host_mma_offsets,
        sizeof(host_mma_offsets), cudaMemcpyHostToDevice, stream));
    require_cuda(cudaMemcpyAsync(device_residual_offsets, host_residual_offsets,
        sizeof(host_residual_offsets), cudaMemcpyHostToDevice, stream));

    sm70::value_pack_request_v1 request{};
    request.value_map = device_map;
    request.value_map_count = 3u;
    request.logical_edge_values = device_values;
    request.logical_edge_count = 3u;
    request.mma_region_offsets = device_mma_offsets;
    request.mma_region_count = 1u;
    request.residual_region_offsets = device_residual_offsets;
    request.residual_region_count = 1u;
    request.mma_values = device_mma;
    request.mma_value_count = 256u;
    request.residual_values = device_residual;
    request.residual_value_count = 1u;
    request.source_generation.value = 7u;
    request.stream = stream;
    sm70::value_pack_state_v1 state{};
    assert(sm70::enqueue_value_pack_v1(request, &state)
        == sm70::value_pack_status_v1::success);

    __half host_mma[256]{};
    __half host_residual[1]{};
    require_cuda(cudaMemcpyAsync(host_mma, device_mma, sizeof(host_mma),
        cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaMemcpyAsync(host_residual, device_residual,
        sizeof(host_residual), cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    assert(__half2float(host_mma[0]) == 3.25f);
    assert(__half2float(host_mma[17]) == 1.5f);
    assert(__half2float(host_mma[1]) == 0.0f);
    assert(__half2float(host_residual[0]) == -2.0f);
    assert(state.packed_generation.value == 7u);

    const __half next_values[3] = {
        __float2half(4.0f), __float2half(5.0f), __float2half(6.0f)};
    require_cuda(cudaMemcpyAsync(device_values, next_values, sizeof(next_values),
        cudaMemcpyHostToDevice, stream));
    request.source_generation.value = 8u;
    assert(sm70::enqueue_value_pack_v1(request, &state)
        == sm70::value_pack_status_v1::success);
    require_cuda(cudaMemcpyAsync(host_mma, device_mma, sizeof(host_mma),
        cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    assert(__half2float(host_mma[0]) == 6.0f);
    assert(__half2float(host_mma[17]) == 4.0f);
    assert(__half2float(host_mma[1]) == 0.0f);
    assert(state.packed_generation.value == 8u);

    require_cuda(cudaStreamDestroy(stream));
    require_cuda(cudaFree(device_residual));
    require_cuda(cudaFree(device_mma));
    require_cuda(cudaFree(device_residual_offsets));
    require_cuda(cudaFree(device_mma_offsets));
    require_cuda(cudaFree(device_values));
    require_cuda(cudaFree(device_map));
    return 0;
}
