#include "../../../src/compute/architecture/providers/nvidia/sm70/relation_apply_n32.cuh"

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

void fill_diagonal(__half *tile, float scale) {
    for (std::uint32_t index = 0u; index < 256u; ++index)
        tile[index] = __float2half(0.0f);
    for (std::uint32_t row = 0u; row < 16u; ++row)
        tile[row * 16u + row] = __float2half(scale);
}

void check_group(
    const float *output,
    std::uint32_t output_group,
    const __half *rhs,
    std::uint32_t source_base,
    float scale) {
    for (std::uint32_t row = 0u; row < 16u; ++row) {
        for (std::uint32_t column = 0u; column < 32u; ++column) {
            const float expected = scale * __half2float(
                rhs[(source_base + row) * 32u + column]);
            const std::uint32_t index =
                (output_group * 16u + row) * 32u + column;
            assert(std::fabs(output[index] - expected) < 1.0e-5f);
        }
    }
}

} // namespace

int main() {
    static_assert(sm70::relation_apply_n32_empirical_required_v1,
        "N32 variants must remain empirical-required until evaluation");
    __half relation_tiles[3u * 256u]{};
    fill_diagonal(relation_tiles, 1.0f);
    fill_diagonal(relation_tiles + 256u, 2.0f);
    fill_diagonal(relation_tiles + 512u, 3.0f);
    __half rhs[48u * 32u]{};
    for (std::uint32_t row = 0u; row < 48u; ++row)
        for (std::uint32_t column = 0u; column < 32u; ++column)
            rhs[row * 32u + column] = __float2half(
                static_cast<float>((row % 7u) + (column % 5u) + 1u));
    const std::uint32_t one_group_offsets[] = {0u, 1u};
    const std::uint32_t paired_offsets[] = {0u, 1u, 2u};
    const std::uint32_t source_bases[] = {0u, 16u, 32u};

    __half *device_relation = nullptr;
    __half *device_rhs = nullptr;
    std::uint32_t *device_offsets = nullptr;
    std::uint32_t *device_source_bases = nullptr;
    float *device_output = nullptr;
    require_cuda(cudaMalloc(&device_relation, sizeof(relation_tiles)));
    require_cuda(cudaMalloc(&device_rhs, sizeof(rhs)));
    require_cuda(cudaMalloc(&device_offsets, sizeof(paired_offsets)));
    require_cuda(cudaMalloc(&device_source_bases, sizeof(source_bases)));
    require_cuda(cudaMalloc(&device_output, 2u * 16u * 32u * sizeof(float)));
    require_cuda(cudaMemcpy(device_relation, relation_tiles,
        sizeof(relation_tiles), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_rhs, rhs, sizeof(rhs),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_source_bases, source_bases,
        sizeof(source_bases), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    sm70::relation_apply_n32_request_v1 request{};
    request.variant = sm70::relation_apply_n32_variant_v1::two_warp_one_group;
    request.relation_tiles = device_relation;
    request.tile_count = 1u;
    require_cuda(cudaMemcpyAsync(device_offsets, one_group_offsets,
        sizeof(one_group_offsets), cudaMemcpyHostToDevice, stream));
    request.destination_tile_offsets = device_offsets;
    request.destination_group_count = 1u;
    request.tile_source_bases = device_source_bases;
    request.dense_rhs = device_rhs;
    request.source_count = 48u;
    request.output = device_output;
    request.stream = stream;
    assert(sm70::enqueue_relation_apply_n32_v1(request)
        == sm70::relation_apply_n32_status_v1::success);
    float output[2u * 16u * 32u]{};
    require_cuda(cudaMemcpyAsync(output, device_output,
        16u * 32u * sizeof(float), cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    check_group(output, 0u, rhs, 0u, 1.0f);

    request.variant =
        sm70::relation_apply_n32_variant_v1::four_warp_two_compatible_groups;
    request.relation_tiles = device_relation + 256u;
    request.tile_count = 2u;
    require_cuda(cudaMemcpyAsync(device_offsets, paired_offsets,
        sizeof(paired_offsets), cudaMemcpyHostToDevice, stream));
    request.destination_group_count = 2u;
    request.tile_source_bases = device_source_bases + 1u;
    assert(sm70::enqueue_relation_apply_n32_v1(request)
        == sm70::relation_apply_n32_status_v1::success);
    require_cuda(cudaMemcpyAsync(output, device_output, sizeof(output),
        cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    check_group(output, 0u, rhs, 16u, 2.0f);
    check_group(output, 1u, rhs, 32u, 3.0f);

    request.destination_group_count = 1u;
    assert(sm70::enqueue_relation_apply_n32_v1(request)
        == sm70::relation_apply_n32_status_v1::invalid_argument);

    require_cuda(cudaStreamDestroy(stream));
    require_cuda(cudaFree(device_output));
    require_cuda(cudaFree(device_source_bases));
    require_cuda(cudaFree(device_offsets));
    require_cuda(cudaFree(device_rhs));
    require_cuda(cudaFree(device_relation));
    return 0;
}
