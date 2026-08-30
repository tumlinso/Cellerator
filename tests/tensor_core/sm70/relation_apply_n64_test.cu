#include "../../../src/compute/architecture/providers/nvidia/sm70/relation_apply_n64.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {

void require_cuda(cudaError_t status) {
    assert(status == cudaSuccess);
}

} // namespace

int main() {
    __half relation[2u * 256u]{};
    for (std::uint32_t row = 0u; row < 16u; ++row) {
        relation[row * 16u + row] = __float2half(1.0f);
        relation[256u + row * 16u + row] = __float2half(2.0f);
    }
    __half rhs[32u * 64u]{};
    for (std::uint32_t row = 0u; row < 32u; ++row)
        for (std::uint32_t column = 0u; column < 64u; ++column)
            rhs[row * 64u + column] = __float2half(
                static_cast<float>((row % 7u) + (column % 5u)));
    const std::uint32_t destination_offsets[] = {0u, 2u};
    const std::uint32_t source_bases[] = {0u, 16u};

    __half *device_relation = nullptr;
    __half *device_rhs = nullptr;
    std::uint32_t *device_destination_offsets = nullptr;
    std::uint32_t *device_source_bases = nullptr;
    float *device_output = nullptr;
    require_cuda(cudaMalloc(&device_relation, sizeof(relation)));
    require_cuda(cudaMalloc(&device_rhs, sizeof(rhs)));
    require_cuda(cudaMalloc(&device_destination_offsets,
        sizeof(destination_offsets)));
    require_cuda(cudaMalloc(&device_source_bases, sizeof(source_bases)));
    require_cuda(cudaMalloc(&device_output, 16u * 64u * sizeof(float)));
    require_cuda(cudaMemcpy(device_relation, relation, sizeof(relation),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_rhs, rhs, sizeof(rhs), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_destination_offsets, destination_offsets,
        sizeof(destination_offsets), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_source_bases, source_bases,
        sizeof(source_bases), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    sm70::relation_apply_n64_request_v1 request{};
    request.relation_tiles = device_relation;
    request.tile_count = 2u;
    request.destination_tile_offsets = device_destination_offsets;
    request.destination_group_count = 1u;
    request.tile_source_bases = device_source_bases;
    request.dense_rhs = device_rhs;
    request.source_count = 32u;
    request.output = device_output;
    request.stream = stream;
    assert(sm70::enqueue_relation_apply_n64_v1(request)
        == sm70::relation_apply_n64_status_v1::success);

    float output[16u * 64u]{};
    require_cuda(cudaMemcpyAsync(output, device_output, sizeof(output),
        cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    for (std::uint32_t row = 0u; row < 16u; ++row) {
        for (std::uint32_t column = 0u; column < 64u; ++column) {
            const float expected = __half2float(rhs[row * 64u + column])
                + 2.0f * __half2float(rhs[(16u + row) * 64u + column]);
            assert(std::fabs(output[row * 64u + column] - expected) < 1.0e-5f);
        }
    }

    require_cuda(cudaStreamDestroy(stream));
    require_cuda(cudaFree(device_output));
    require_cuda(cudaFree(device_source_bases));
    require_cuda(cudaFree(device_destination_offsets));
    require_cuda(cudaFree(device_rhs));
    require_cuda(cudaFree(device_relation));
    return 0;
}
