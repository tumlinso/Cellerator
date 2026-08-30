#include "../../../src/compute/architecture/providers/nvidia/sm70/relation_apply_widths.cuh"

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

void verify_width(std::uint32_t width) {
    std::vector<__half> relation(2u * 256u);
    for (std::uint32_t row = 0u; row < 16u; ++row) {
        relation[row * 16u + row] = __float2half(1.0f);
        relation[256u + row * 16u + row] = __float2half(2.0f);
    }
    std::vector<__half> rhs(static_cast<std::size_t>(32u) * width);
    for (std::uint32_t row = 0u; row < 32u; ++row)
        for (std::uint32_t column = 0u; column < width; ++column)
            rhs[static_cast<std::size_t>(row) * width + column] =
                __float2half(static_cast<float>((row % 7u) + (column % 5u)));
    const std::uint32_t destination_offsets[] = {0u, 2u};
    const std::uint32_t source_bases[] = {0u, 16u};

    __half *device_relation = nullptr;
    __half *device_rhs = nullptr;
    std::uint32_t *device_destination_offsets = nullptr;
    std::uint32_t *device_source_bases = nullptr;
    float *device_output = nullptr;
    require_cuda(cudaMalloc(&device_relation,
        relation.size() * sizeof(__half)));
    require_cuda(cudaMalloc(&device_rhs, rhs.size() * sizeof(__half)));
    require_cuda(cudaMalloc(&device_destination_offsets,
        sizeof(destination_offsets)));
    require_cuda(cudaMalloc(&device_source_bases, sizeof(source_bases)));
    require_cuda(cudaMalloc(&device_output,
        static_cast<std::size_t>(16u) * width * sizeof(float)));
    require_cuda(cudaMemcpy(device_relation, relation.data(),
        relation.size() * sizeof(__half), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_rhs, rhs.data(), rhs.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_destination_offsets, destination_offsets,
        sizeof(destination_offsets), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_source_bases, source_bases,
        sizeof(source_bases), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    sm70::relation_apply_widths_request_v1 request{};
    request.relation_tiles = device_relation;
    request.tile_count = 2u;
    request.destination_tile_offsets = device_destination_offsets;
    request.destination_group_count = 1u;
    request.tile_source_bases = device_source_bases;
    request.dense_rhs = device_rhs;
    request.source_count = 32u;
    request.dense_width = width;
    request.output = device_output;
    request.stream = stream;
    assert(sm70::enqueue_relation_apply_widths_v1(request)
        == sm70::relation_apply_widths_status_v1::success);

    std::vector<float> output(static_cast<std::size_t>(16u) * width);
    require_cuda(cudaMemcpyAsync(output.data(), device_output,
        output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    for (std::uint32_t row = 0u; row < 16u; ++row) {
        for (std::uint32_t column = 0u; column < width; ++column) {
            const float expected =
                __half2float(rhs[static_cast<std::size_t>(row) * width
                    + column])
                + 2.0f * __half2float(
                    rhs[static_cast<std::size_t>(16u + row) * width
                        + column]);
            assert(std::fabs(output[static_cast<std::size_t>(row) * width
                + column] - expected) < 1.0e-5f);
        }
    }

    require_cuda(cudaStreamDestroy(stream));
    require_cuda(cudaFree(device_output));
    require_cuda(cudaFree(device_source_bases));
    require_cuda(cudaFree(device_destination_offsets));
    require_cuda(cudaFree(device_rhs));
    require_cuda(cudaFree(device_relation));
}

} // namespace

int main() {
    using route = sm70::relation_apply_width_route_v1;
    assert(sm70::relation_apply_widths_empirical_required_v1);
    assert(sm70::select_relation_apply_width_route_v1(0u)
        == route::sparse_fallback);
    assert(sm70::select_relation_apply_width_route_v1(1u)
        == route::specialized_n1);
    assert(sm70::select_relation_apply_width_route_v1(2u)
        == route::sparse_fallback);
    assert(sm70::select_relation_apply_width_route_v1(15u)
        == route::sparse_fallback);
    assert(sm70::select_relation_apply_width_route_v1(16u)
        == route::one_warp_n16);
    assert(sm70::select_relation_apply_width_route_v1(32u)
        == route::existing_n32);
    assert(sm70::select_relation_apply_width_route_v1(64u)
        == route::existing_n64);
    assert(sm70::select_relation_apply_width_route_v1(65u)
        == route::sparse_fallback);
    assert(sm70::select_relation_apply_width_route_v1(80u)
        == route::disjoint_column_panels);

    sm70::relation_apply_widths_request_v1 fallback{};
    fallback.dense_width = 15u;
    assert(sm70::enqueue_relation_apply_widths_v1(fallback)
        == sm70::relation_apply_widths_status_v1::fallback_required);
    sm70::relation_apply_widths_request_v1 invalid{};
    invalid.dense_width = 16u;
    assert(sm70::enqueue_relation_apply_widths_v1(invalid)
        == sm70::relation_apply_widths_status_v1::invalid_argument);

    verify_width(16u);
    verify_width(80u);
    return 0;
}
