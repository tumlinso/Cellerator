#include "../../../src/compute/architecture/providers/nvidia/sm70/contract_on_support.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {
void require_cuda(cudaError_t status) { assert(status == cudaSuccess); }
}

int main() {
    constexpr std::uint32_t width = 17u;
    sm70::support_logical_edge_v1 edges[4]{};
    projection::projection_value_map_v1 map[4]{};
    for (std::uint32_t index = 0u; index < 4u; ++index) {
        edges[index].logical_edge_id.value = 20u + index;
        edges[index].source_index = index % 3u;
        edges[index].destination_index = index;
        map[index].logical_edge_id = edges[index].logical_edge_id;
        map[index].region_kind = index < 2u
            ? projection::physical_region_kind_v1::mma
            : projection::physical_region_kind_v1::residual;
        map[index].region_index = index;
        map[index].projection_slot = index * 3u;
    }
    const std::uint8_t source_support[] = {1u, 0u, 1u};
    const std::uint8_t destination_support[] = {1u, 1u, 1u, 1u};
    sm70::support_projection_edge_v1 selected[4]{};
    sm70::contract_projection_request_v1 prepare{};
    prepare.logical_edges = edges;
    prepare.physical_value_map = map;
    prepare.logical_edge_count = 4u;
    prepare.source_support = source_support;
    prepare.source_count = 3u;
    prepare.destination_support = destination_support;
    prepare.destination_count = 4u;
    prepare.selected_edges = selected;
    prepare.selected_capacity = 4u;
    sm70::contract_projection_result_v1 prepared{};
    assert(sm70::prepare_contract_projection_v1(prepare, &prepared)
        == sm70::contract_projection_status_v1::success);
    assert(prepared.selected_edge_count == 3u);

    __half source_features[3u * width]{};
    __half destination_features[4u * width]{};
    for (std::uint32_t row = 0u; row < 3u; ++row)
        for (std::uint32_t column = 0u; column < width; ++column)
            source_features[row * width + column] = __float2half(
                static_cast<float>(row + column % 4u));
    for (std::uint32_t row = 0u; row < 4u; ++row)
        for (std::uint32_t column = 0u; column < width; ++column)
            destination_features[row * width + column] = __float2half(
                static_cast<float>(2u * row + column % 3u));

    sm70::support_logical_edge_v1 *device_edges = nullptr;
    sm70::support_projection_edge_v1 *device_selected = nullptr;
    __half *device_source = nullptr;
    __half *device_destination = nullptr;
    float *device_output = nullptr;
    require_cuda(cudaMalloc(&device_edges, sizeof(edges)));
    require_cuda(cudaMalloc(&device_selected,
        prepared.selected_edge_count * sizeof(selected[0])));
    require_cuda(cudaMalloc(&device_source, sizeof(source_features)));
    require_cuda(cudaMalloc(&device_destination,
        sizeof(destination_features)));
    require_cuda(cudaMalloc(&device_output, 4u * sizeof(float)));
    require_cuda(cudaMemcpy(device_edges, edges, sizeof(edges),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_selected, selected,
        prepared.selected_edge_count * sizeof(selected[0]),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_source, source_features,
        sizeof(source_features), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_destination, destination_features,
        sizeof(destination_features), cudaMemcpyHostToDevice));
    sm70::contract_on_support_request_v1 request{};
    request.logical_edges = device_edges;
    request.logical_edge_count = 4u;
    request.selected_edges = device_selected;
    request.selected_edge_count = prepared.selected_edge_count;
    request.source_features = device_source;
    request.source_count = 3u;
    request.destination_features = device_destination;
    request.destination_count = 4u;
    request.dense_width = width;
    request.logical_edge_output = device_output;
    assert(sm70::enqueue_contract_on_support_v1(request)
        == sm70::contract_on_support_status_v1::success);
    float output[4]{};
    require_cuda(cudaMemcpy(output, device_output, sizeof(output),
        cudaMemcpyDeviceToHost));
    for (std::uint32_t edge = 0u; edge < 4u; ++edge) {
        float expected = 0.0f;
        if (source_support[edges[edge].source_index] != 0u) {
            for (std::uint32_t column = 0u; column < width; ++column)
                expected += __half2float(source_features[
                    edges[edge].source_index * width + column])
                    * __half2float(destination_features[
                        edges[edge].destination_index * width + column]);
        }
        assert(std::fabs(output[edge] - expected) < 1.0e-4f);
    }
    assert(selected[0].region_kind == projection::physical_region_kind_v1::mma);
    assert(selected[2].region_kind
        == projection::physical_region_kind_v1::residual);

    require_cuda(cudaFree(device_output));
    require_cuda(cudaFree(device_destination));
    require_cuda(cudaFree(device_source));
    require_cuda(cudaFree(device_selected));
    require_cuda(cudaFree(device_edges));
    return 0;
}
