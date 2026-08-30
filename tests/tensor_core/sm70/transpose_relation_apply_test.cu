#include "../../../src/compute/architecture/providers/nvidia/sm70/transpose_relation_apply.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {

void require_cuda(cudaError_t status) {
    assert(status == cudaSuccess);
}

} // namespace

int main() {
    constexpr std::uint32_t sources = 20u;
    constexpr std::uint32_t destinations = 35u;
    constexpr std::uint32_t width = 17u;
    const sm70::logical_relation_edge_v1 edges[] = {
        {11u, 2u, 3u}, {12u, 2u, 20u}, {13u, 18u, 7u},
        {14u, 15u, 33u}, {15u, 19u, 34u}};
    constexpr std::uint32_t edge_count = sizeof(edges) / sizeof(edges[0]);
    sm70::target_edge_placement_v1 forward[edge_count]{};
    sm70::target_edge_placement_v1 transpose[edge_count]{};
    sm70::transpose_cover_request_v1 cover_request{};
    cover_request.logical_edges = edges;
    cover_request.logical_edge_count = edge_count;
    cover_request.source_count = sources;
    cover_request.destination_count = destinations;
    cover_request.forward = forward;
    cover_request.forward_capacity = edge_count;
    cover_request.transpose = transpose;
    cover_request.transpose_capacity = edge_count;
    assert(sm70::build_transpose_cover_v1(cover_request)
        == sm70::transpose_cover_status_v1::success);

    __half edge_values[edge_count]{};
    for (std::uint32_t edge = 0u; edge < edge_count; ++edge)
        edge_values[edge] = __float2half(
            static_cast<float>(static_cast<int>(edge) - 1) * 0.5f);
    std::vector<__half> destination_gradient(
        static_cast<std::size_t>(destinations) * width);
    for (std::uint32_t destination = 0u; destination < destinations;
        ++destination)
        for (std::uint32_t column = 0u; column < width; ++column)
            destination_gradient[
                static_cast<std::size_t>(destination) * width + column] =
                __float2half(static_cast<float>(
                    static_cast<int>(destination % 7u)
                    - static_cast<int>(column % 3u)));

    sm70::target_edge_placement_v1 *device_cover = nullptr;
    __half *device_values = nullptr;
    __half *device_destination_gradient = nullptr;
    float *device_source_gradient = nullptr;
    require_cuda(cudaMalloc(&device_cover, sizeof(transpose)));
    require_cuda(cudaMalloc(&device_values, sizeof(edge_values)));
    require_cuda(cudaMalloc(&device_destination_gradient,
        destination_gradient.size() * sizeof(__half)));
    require_cuda(cudaMalloc(&device_source_gradient,
        static_cast<std::size_t>(sources) * width * sizeof(float)));
    require_cuda(cudaMemcpy(device_cover, transpose, sizeof(transpose),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_values, edge_values, sizeof(edge_values),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_destination_gradient,
        destination_gradient.data(),
        destination_gradient.size() * sizeof(__half), cudaMemcpyHostToDevice));

    sm70::transpose_relation_apply_request_v1 request{};
    request.transpose_cover = device_cover;
    request.logical_edge_values = device_values;
    request.logical_edge_count = edge_count;
    request.destination_gradient = device_destination_gradient;
    request.destination_count = destinations;
    request.source_count = sources;
    request.dense_width = width;
    request.source_gradient = device_source_gradient;
    assert(sm70::enqueue_transpose_relation_apply_v1(request)
        == sm70::transpose_relation_apply_status_v1::success);
    std::vector<float> source_gradient(
        static_cast<std::size_t>(sources) * width);
    require_cuda(cudaMemcpy(source_gradient.data(), device_source_gradient,
        source_gradient.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::uint32_t source = 0u; source < sources; ++source) {
        for (std::uint32_t column = 0u; column < width; ++column) {
            float expected = 0.0f;
            for (std::uint32_t edge = 0u; edge < edge_count; ++edge) {
                if (edges[edge].source_index != source) continue;
                expected += __half2float(edge_values[edge]) * __half2float(
                    destination_gradient[static_cast<std::size_t>(
                        edges[edge].destination_index) * width + column]);
            }
            assert(std::fabs(source_gradient[
                static_cast<std::size_t>(source) * width + column]
                - expected) < 1.0e-5f);
        }
    }

    sm70::transpose_relation_apply_request_v1 invalid{};
    assert(sm70::enqueue_transpose_relation_apply_v1(invalid)
        == sm70::transpose_relation_apply_status_v1::invalid_argument);
    require_cuda(cudaFree(device_source_gradient));
    require_cuda(cudaFree(device_destination_gradient));
    require_cuda(cudaFree(device_values));
    require_cuda(cudaFree(device_cover));
    return 0;
}
