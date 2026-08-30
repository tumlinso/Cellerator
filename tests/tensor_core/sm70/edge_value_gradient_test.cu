#include "../../../src/compute/architecture/providers/nvidia/sm70/edge_value_gradient.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace { void require_cuda(cudaError_t status) { assert(status == cudaSuccess); } }

int main() {
    constexpr std::uint32_t width = 17u;
    sm70::support_logical_edge_v1 edges[4]{};
    for (std::uint32_t index = 0u; index < 4u; ++index) {
        edges[index].logical_edge_id.value = 41u + index * 9u;
        edges[index].source_index = index % 3u;
        edges[index].destination_index = 3u - index;
    }
    const std::uint32_t order[] = {2u, 0u, 3u, 1u};
    projection::projection_value_map_v1 shuffled[4]{};
    for (std::uint32_t physical = 0u; physical < 4u; ++physical) {
        shuffled[physical].logical_edge_id = edges[order[physical]].logical_edge_id;
        shuffled[physical].region_kind = physical % 2u == 0u
            ? projection::physical_region_kind_v1::mma
            : projection::physical_region_kind_v1::residual;
        shuffled[physical].region_index = physical;
        shuffled[physical].projection_slot = 10u + physical;
    }
    __half source[3u * width]{};
    __half destination_gradient[4u * width]{};
    for (std::uint32_t i = 0u; i < 3u * width; ++i)
        source[i] = __float2half(static_cast<float>(i % 7u) - 2.0f);
    for (std::uint32_t i = 0u; i < 4u * width; ++i)
        destination_gradient[i] = __float2half(static_cast<float>(i % 5u) - 1.0f);

    sm70::support_logical_edge_v1 *device_edges = nullptr;
    projection::projection_value_map_v1 *device_map = nullptr;
    __half *device_source = nullptr;
    __half *device_destination_gradient = nullptr;
    float *device_output = nullptr;
    require_cuda(cudaMalloc(&device_edges, sizeof(edges)));
    require_cuda(cudaMalloc(&device_map, sizeof(shuffled)));
    require_cuda(cudaMalloc(&device_source, sizeof(source)));
    require_cuda(cudaMalloc(&device_destination_gradient,
        sizeof(destination_gradient)));
    require_cuda(cudaMalloc(&device_output, 4u * sizeof(float)));
    require_cuda(cudaMemcpy(device_edges, edges, sizeof(edges), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_map, shuffled, sizeof(shuffled), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_source, source, sizeof(source), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_destination_gradient, destination_gradient,
        sizeof(destination_gradient), cudaMemcpyHostToDevice));
    sm70::edge_value_gradient_request_v1 request{};
    request.logical_edges = device_edges;
    request.physical_value_map = device_map;
    request.logical_edge_count = 4u;
    request.source_activation = device_source;
    request.source_count = 3u;
    request.destination_gradient = device_destination_gradient;
    request.destination_count = 4u;
    request.dense_width = width;
    request.logical_edge_gradient = device_output;
    assert(sm70::enqueue_edge_value_gradient_v1(request)
        == sm70::edge_value_gradient_status_v1::success);
    float output[4]{};
    require_cuda(cudaMemcpy(output, device_output, sizeof(output),
        cudaMemcpyDeviceToHost));
    for (std::uint32_t edge = 0u; edge < 4u; ++edge) {
        float expected = 0.0f;
        for (std::uint32_t column = 0u; column < width; ++column)
            expected += __half2float(source[edges[edge].source_index * width + column])
                * __half2float(destination_gradient[
                    edges[edge].destination_index * width + column]);
        assert(std::fabs(output[edge] - expected) < 1.0e-4f);
    }
    require_cuda(cudaFree(device_output));
    require_cuda(cudaFree(device_destination_gradient));
    require_cuda(cudaFree(device_source));
    require_cuda(cudaFree(device_map));
    require_cuda(cudaFree(device_edges));
    return 0;
}
