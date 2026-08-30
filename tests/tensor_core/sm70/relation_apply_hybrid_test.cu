#include "../../../src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cuh"

#include <Cellerator/compute/projection/physical_mma_hybrid.hh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {

void require_cuda(cudaError_t status) {
    assert(status == cudaSuccess);
}

bool pure_sparse_fallback(void *context, cudaStream_t stream) noexcept {
    return context != nullptr && stream != nullptr;
}

cellerator::compute::math::feature_order_identity order(
    std::uint64_t identity) {
    cellerator::compute::math::feature_order_identity result{};
    result.kind = cellerator::compute::math::feature_order_kind::packed;
    result.feature_count = 16u;
    result.feature_axis_identity_version = 1u;
    result.feature_axis_identity = identity;
    result.packing_geometry_identity = 9u;
    return result;
}

} // namespace

int main() {
    constexpr std::uint32_t rows = 16u;
    constexpr std::uint32_t width = 64u;
    constexpr std::uint32_t sources = 32u;
    constexpr std::uint32_t mma_edges = 16u;
    constexpr std::uint32_t residual_edges = 1u;
    constexpr std::uint32_t logical_edges = mma_edges + residual_edges;
    constexpr std::uint32_t output_count = rows * width;

    projection::projection_value_map_v1 value_map[logical_edges]{};
    for (std::uint32_t row = 0u; row < rows; ++row) {
        value_map[row].logical_edge_id.value = row;
        value_map[row].region_kind = projection::physical_region_kind_v1::mma;
        value_map[row].region_index = 0u;
        value_map[row].projection_slot = row * rows + row;
    }
    value_map[mma_edges].logical_edge_id.value = mma_edges;
    value_map[mma_edges].region_kind =
        projection::physical_region_kind_v1::residual;
    value_map[mma_edges].region_index = 0u;
    value_map[mma_edges].projection_slot = 0u;

    __half logical_values[logical_edges]{};
    for (std::uint32_t edge = 0u; edge < mma_edges; ++edge)
        logical_values[edge] = __float2half(1.0f);
    logical_values[mma_edges] = __float2half(2.0f);
    __half rhs[sources * width]{};
    for (std::uint32_t source = 0u; source < sources; ++source)
        for (std::uint32_t column = 0u; column < width; ++column)
            rhs[source * width + column] = __float2half(
                static_cast<float>((source % 5u) + (column % 3u) + 1u));
    float prior[output_count]{};
    for (float &value : prior) value = 4.0f;

    projection::projection_value_map_v1 *device_map = nullptr;
    __half *device_logical_values = nullptr;
    std::uint64_t *device_mma_offsets = nullptr;
    std::uint64_t *device_residual_offsets = nullptr;
    __half *device_mma_values = nullptr;
    __half *device_residual_values = nullptr;
    std::uint32_t *device_destination_offsets = nullptr;
    std::uint32_t *device_tile_source_bases = nullptr;
    std::uint32_t *device_residual_row_offsets = nullptr;
    std::uint32_t *device_residual_columns = nullptr;
    __half *device_rhs = nullptr;
    float *device_accumulation = nullptr;
    float *device_prior = nullptr;
    float *device_output = nullptr;
    require_cuda(cudaMalloc(&device_map, sizeof(value_map)));
    require_cuda(cudaMalloc(&device_logical_values, sizeof(logical_values)));
    require_cuda(cudaMalloc(&device_mma_offsets, sizeof(std::uint64_t)));
    require_cuda(cudaMalloc(&device_residual_offsets, sizeof(std::uint64_t)));
    require_cuda(cudaMalloc(&device_mma_values, 256u * sizeof(__half)));
    require_cuda(cudaMalloc(&device_residual_values,
        residual_edges * sizeof(__half)));
    require_cuda(cudaMalloc(&device_destination_offsets,
        2u * sizeof(std::uint32_t)));
    require_cuda(cudaMalloc(&device_tile_source_bases,
        sizeof(std::uint32_t)));
    require_cuda(cudaMalloc(&device_residual_row_offsets,
        (rows + 1u) * sizeof(std::uint32_t)));
    require_cuda(cudaMalloc(&device_residual_columns,
        residual_edges * sizeof(std::uint32_t)));
    require_cuda(cudaMalloc(&device_rhs, sizeof(rhs)));
    require_cuda(cudaMalloc(&device_accumulation,
        output_count * sizeof(float)));
    require_cuda(cudaMalloc(&device_prior, sizeof(prior)));
    require_cuda(cudaMalloc(&device_output, output_count * sizeof(float)));

    const std::uint64_t region_offset = 0u;
    const std::uint32_t destination_offsets[] = {0u, 1u};
    const std::uint32_t tile_source_bases[] = {0u};
    std::uint32_t residual_row_offsets[rows + 1u]{};
    for (std::uint32_t row = 1u; row <= rows; ++row)
        residual_row_offsets[row] = residual_edges;
    const std::uint32_t residual_columns[] = {16u};
    require_cuda(cudaMemcpy(device_map, value_map, sizeof(value_map),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_logical_values, logical_values,
        sizeof(logical_values), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_mma_offsets, &region_offset,
        sizeof(region_offset), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_residual_offsets, &region_offset,
        sizeof(region_offset), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_destination_offsets, destination_offsets,
        sizeof(destination_offsets), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_tile_source_bases, tile_source_bases,
        sizeof(tile_source_bases), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_residual_row_offsets, residual_row_offsets,
        sizeof(residual_row_offsets), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_residual_columns, residual_columns,
        sizeof(residual_columns), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_rhs, rhs, sizeof(rhs),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_prior, prior, sizeof(prior),
        cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    sm70::relation_apply_hybrid_request_v1 request{};
    request.value_pack.value_map = device_map;
    request.value_pack.value_map_count = logical_edges;
    request.value_pack.logical_edge_values = device_logical_values;
    request.value_pack.logical_edge_count = logical_edges;
    request.value_pack.mma_region_offsets = device_mma_offsets;
    request.value_pack.mma_region_count = 1u;
    request.value_pack.residual_region_offsets = device_residual_offsets;
    request.value_pack.residual_region_count = 1u;
    request.value_pack.mma_values = device_mma_values;
    request.value_pack.mma_value_count = 256u;
    request.value_pack.residual_values = device_residual_values;
    request.value_pack.residual_value_count = residual_edges;
    request.value_pack.source_generation.value = 7u;
    request.value_pack.stream = stream;
    sm70::value_pack_state_v1 pack_state{};
    request.value_pack_state = &pack_state;
    request.mma.relation_tiles = device_mma_values;
    request.mma.tile_count = 1u;
    request.mma.destination_tile_offsets = device_destination_offsets;
    request.mma.destination_group_count = 1u;
    request.mma.tile_source_bases = device_tile_source_bases;
    request.mma.dense_rhs = device_rhs;
    request.mma.source_count = sources;
    request.mma.output = device_accumulation;
    request.mma.stream = stream;
    request.residual.row_offsets = device_residual_row_offsets;
    request.residual.row_count = rows;
    request.residual.column_indices = device_residual_columns;
    request.residual.edge_count = residual_edges;
    request.residual.edge_values = device_residual_values;
    request.residual.dense_rhs = device_rhs;
    request.residual.source_count = sources;
    request.residual.dense_width = width;
    request.residual.accumulation = device_accumulation;
    request.residual.stream = stream;
    request.beta_source = device_prior;
    request.output = device_output;
    request.output_count = output_count;
    request.alpha = 2.0f;
    request.beta = 0.5f;
    request.source_order = order(101u);
    request.destination_order = order(202u);
    request.hybrid_complete_cost.dynamic_input_pack_ns = 3.0;
    request.hybrid_complete_cost.kernel_ns = 8.0;
    request.hybrid_complete_cost.epilogue_ns = 1.0;
    request.pure_sparse_complete_cost.kernel_ns = 20.0;
    std::uint32_t fallback_context = 1u;
    request.pure_sparse_fallback = &pure_sparse_fallback;
    request.pure_sparse_context = &fallback_context;
    request.stream = stream;

    assert(sm70::enqueue_relation_apply_hybrid_v1(request)
        == sm70::relation_apply_hybrid_status_v1::success);
    float output[output_count]{};
    require_cuda(cudaMemcpyAsync(output, device_output, sizeof(output),
        cudaMemcpyDeviceToHost, stream));
    require_cuda(cudaStreamSynchronize(stream));
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t column = 0u; column < width; ++column) {
            float sum = __half2float(rhs[row * width + column]);
            if (row == 0u)
                sum += 2.0f * __half2float(rhs[16u * width + column]);
            const float expected = 2.0f * sum + 2.0f;
            assert(std::fabs(output[row * width + column] - expected)
                < 1.0e-5f);
        }
    }
    assert(pack_state.packed_generation.value == 7u);

    request.pure_sparse_fallback = nullptr;
    assert(sm70::enqueue_relation_apply_hybrid_v1(request)
        == sm70::relation_apply_hybrid_status_v1::invalid_argument);
    request.pure_sparse_fallback = &pure_sparse_fallback;
    request.value_pack.source_generation.value = 8u;
    for (std::uint32_t edge = 0u; edge < logical_edges; ++edge)
        logical_values[edge] = __float2half(0.5f);
    require_cuda(cudaMemcpyAsync(device_logical_values, logical_values,
        sizeof(logical_values), cudaMemcpyHostToDevice, stream));
    assert(sm70::enqueue_relation_apply_hybrid_v1(request)
        == sm70::relation_apply_hybrid_status_v1::success);
    require_cuda(cudaStreamSynchronize(stream));
    assert(pack_state.packed_generation.value == 8u);

    require_cuda(cudaStreamDestroy(stream));
    require_cuda(cudaFree(device_output));
    require_cuda(cudaFree(device_prior));
    require_cuda(cudaFree(device_accumulation));
    require_cuda(cudaFree(device_rhs));
    require_cuda(cudaFree(device_residual_columns));
    require_cuda(cudaFree(device_residual_row_offsets));
    require_cuda(cudaFree(device_tile_source_bases));
    require_cuda(cudaFree(device_destination_offsets));
    require_cuda(cudaFree(device_residual_values));
    require_cuda(cudaFree(device_mma_values));
    require_cuda(cudaFree(device_residual_offsets));
    require_cuda(cudaFree(device_mma_offsets));
    require_cuda(cudaFree(device_logical_values));
    require_cuda(cudaFree(device_map));
    return 0;
}
