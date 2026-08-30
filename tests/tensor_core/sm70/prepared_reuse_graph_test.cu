#include "../../../src/compute/architecture/providers/nvidia/sm70/value_pack.cuh"

#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {

void require_cuda(cudaError_t status) {
    assert(status == cudaSuccess);
}

void copy_values(__half *destination,
    const std::array<float, 4u> &source, cudaStream_t stream) {
    std::array<__half, 4u> packed{};
    for (std::size_t index = 0u; index < packed.size(); ++index)
        packed[index] = __float2half(source[index]);
    require_cuda(cudaMemcpyAsync(destination, packed.data(),
        sizeof(packed), cudaMemcpyHostToDevice, stream));
}

void verify_packed(const __half *device_mma, const __half *device_residual,
    const std::array<float, 4u> &expected, cudaStream_t stream) {
    std::array<__half, 256u> mma{};
    std::array<__half, 2u> residual{};
    require_cuda(cudaMemcpyAsync(mma.data(), device_mma, sizeof(mma),
        cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaMemcpyAsync(residual.data(), device_residual,
        sizeof(residual), cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    for (std::size_t index = 0u; index < mma.size(); ++index) {
        const float value = __half2float(mma[index]);
        const float wanted = index == 0u ? expected[0]
            : (index == 255u ? expected[1] : 0.0f);
        assert(value == wanted);
    }
    assert(__half2float(residual[0]) == expected[2]);
    assert(__half2float(residual[1]) == expected[3]);
}

} // namespace

int main() {
    std::array<projection::projection_value_map_v1, 4u> map{};
    for (std::uint32_t index = 0u; index < map.size(); ++index)
        map[index].logical_edge_id.value = index;
    map[0].region_kind = projection::physical_region_kind_v1::mma;
    map[0].region_index = 0u;
    map[0].projection_slot = 0u;
    map[1].region_kind = projection::physical_region_kind_v1::mma;
    map[1].region_index = 0u;
    map[1].projection_slot = 255u;
    map[2].region_kind = projection::physical_region_kind_v1::residual;
    map[2].region_index = 0u;
    map[2].projection_slot = 0u;
    map[3].region_kind = projection::physical_region_kind_v1::residual;
    map[3].region_index = 0u;
    map[3].projection_slot = 1u;

    projection::projection_value_map_v1 *device_map = nullptr;
    __half *device_logical = nullptr;
    std::uint64_t *device_mma_offsets = nullptr;
    std::uint64_t *device_residual_offsets = nullptr;
    __half *device_mma = nullptr;
    __half *device_residual = nullptr;
    require_cuda(cudaMalloc(&device_map, sizeof(map)));
    require_cuda(cudaMalloc(&device_logical, 4u * sizeof(__half)));
    require_cuda(cudaMalloc(&device_mma_offsets, sizeof(std::uint64_t)));
    require_cuda(cudaMalloc(&device_residual_offsets, sizeof(std::uint64_t)));
    require_cuda(cudaMalloc(&device_mma, 256u * sizeof(__half)));
    require_cuda(cudaMalloc(&device_residual, 2u * sizeof(__half)));
    require_cuda(cudaMemcpy(device_map, map.data(), sizeof(map),
        cudaMemcpyHostToDevice));
    const std::uint64_t offset = 0u;
    require_cuda(cudaMemcpy(device_mma_offsets, &offset, sizeof(offset),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_residual_offsets, &offset, sizeof(offset),
        cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    sm70::value_pack_request_v1 request{};
    request.value_map = device_map;
    request.value_map_count = map.size();
    request.logical_edge_values = device_logical;
    request.logical_edge_count = map.size();
    request.mma_region_offsets = device_mma_offsets;
    request.mma_region_count = 1u;
    request.residual_region_offsets = device_residual_offsets;
    request.residual_region_count = 1u;
    request.mma_values = device_mma;
    request.mma_value_count = 256u;
    request.residual_values = device_residual;
    request.residual_value_count = 2u;
    request.stream = stream;
    sm70::value_pack_state_v1 state{};

    const auto *const stable_map = request.value_map;
    const auto *const stable_logical = request.logical_edge_values;
    auto *const stable_mma = request.mma_values;
    auto *const stable_residual = request.residual_values;

    const std::array<float, 4u> generation1 = {1.0f, 2.0f, 3.0f, 4.0f};
    copy_values(device_logical, generation1, stream);
    request.source_generation.value = 1u;
    assert(sm70::enqueue_value_pack_v1(request, &state)
        == sm70::value_pack_status_v1::success);
    verify_packed(device_mma, device_residual, generation1, stream);
    assert(state.packed_generation.value == 1u);

    const std::array<float, 4u> generation2 = {-2.0f, 5.0f, 7.0f, 9.0f};
    copy_values(device_logical, generation2, stream);
    request.source_generation.value = 2u;
    assert(sm70::enqueue_value_pack_v1(request, &state)
        == sm70::value_pack_status_v1::success);
    verify_packed(device_mma, device_residual, generation2, stream);
    assert(state.packed_generation.value == 2u);
    assert(request.value_map == stable_map);
    assert(request.logical_edge_values == stable_logical);
    assert(request.mma_values == stable_mma);
    assert(request.residual_values == stable_residual);

    request.source_generation.value = 3u;
    require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    assert(sm70::enqueue_value_pack_v1(request, &state)
        == sm70::value_pack_status_v1::success);
    cudaGraph_t graph = nullptr;
    require_cuda(cudaStreamEndCapture(stream, &graph));
    assert(state.packed_generation.value == 3u);

    std::size_t node_count = 0u;
    require_cuda(cudaGraphGetNodes(graph, nullptr, &node_count));
    assert(node_count == 3u);
    std::array<cudaGraphNode_t, 3u> nodes{};
    require_cuda(cudaGraphGetNodes(graph, nodes.data(), &node_count));
    std::uint32_t memset_nodes = 0u;
    std::uint32_t kernel_nodes = 0u;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        require_cuda(cudaGraphNodeGetType(node, &type));
        memset_nodes += type == cudaGraphNodeTypeMemset ? 1u : 0u;
        kernel_nodes += type == cudaGraphNodeTypeKernel ? 1u : 0u;
    }
    // The captured prepared update contains only two preallocated-buffer
    // clears and the projection-local pack kernel: no host, allocation,
    // structure-search, or structure-build node can enter replay.
    assert(memset_nodes == 2u);
    assert(kernel_nodes == 1u);

    cudaGraphExec_t graph_exec = nullptr;
    require_cuda(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0u));
    const std::array<float, 4u> generation3 = {8.0f, -4.0f, 6.0f, 2.0f};
    copy_values(device_logical, generation3, stream);
    require_cuda(cudaGraphLaunch(graph_exec, stream));
    verify_packed(device_mma, device_residual, generation3, stream);
    const std::array<float, 4u> generation4 = {0.5f, 1.5f, -3.0f, 11.0f};
    copy_values(device_logical, generation4, stream);
    require_cuda(cudaGraphLaunch(graph_exec, stream));
    verify_packed(device_mma, device_residual, generation4, stream);

    std::array<projection::projection_value_map_v1, 4u> recovered_map{};
    require_cuda(cudaMemcpy(recovered_map.data(), device_map, sizeof(map),
        cudaMemcpyDeviceToHost));
    assert(std::memcmp(recovered_map.data(), map.data(), sizeof(map)) == 0);
    assert(request.value_map == stable_map);
    assert(request.logical_edge_values == stable_logical);
    assert(request.mma_values == stable_mma);
    assert(request.residual_values == stable_residual);

    require_cuda(cudaGraphExecDestroy(graph_exec));
    require_cuda(cudaGraphDestroy(graph));
    require_cuda(cudaStreamDestroy(stream));
    require_cuda(cudaFree(device_residual));
    require_cuda(cudaFree(device_mma));
    require_cuda(cudaFree(device_residual_offsets));
    require_cuda(cudaFree(device_mma_offsets));
    require_cuda(cudaFree(device_logical));
    require_cuda(cudaFree(device_map));
    return 0;
}
