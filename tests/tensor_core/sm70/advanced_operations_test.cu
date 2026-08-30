#include "../../../src/compute/architecture/providers/nvidia/sm70/transpose_relation_apply.cu"
#include "../../../src/compute/architecture/providers/nvidia/sm70/contract_on_support.cu"
#include "../../../src/compute/architecture/providers/nvidia/sm70/exchange_program.cc"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace operation = cellerator::compute::operation;
namespace projection = cellerator::compute::projection;
namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {

void require_cuda(cudaError_t status) { assert(status == cudaSuccess); }

struct exchange_trace_v1 {
    std::uint32_t calls = 0u;
    std::uint64_t generation = 0u;
};

bool record_exchange_step_v1(void *opaque, void *stream,
    std::uint64_t input_generation,
    std::uint64_t output_generation) noexcept {
    auto *trace = static_cast<exchange_trace_v1 *>(opaque);
    if (stream == nullptr || trace->generation != input_generation) return false;
    ++trace->calls;
    trace->generation = output_generation;
    return true;
}

} // namespace

int main() {
    constexpr std::uint32_t source_count = 2u;
    constexpr std::uint32_t destination_count = 3u;
    constexpr std::uint32_t width = 5u;
    constexpr std::uint32_t edge_count = 3u;
    const sm70::logical_relation_edge_v1 relation_edges[edge_count] = {
        {101u, 0u, 0u}, {102u, 0u, 2u}, {103u, 1u, 1u}};
    sm70::target_edge_placement_v1 forward_cover[edge_count]{};
    sm70::target_edge_placement_v1 transpose_cover[edge_count]{};
    sm70::transpose_cover_request_v1 cover_request{};
    cover_request.logical_edges = relation_edges;
    cover_request.logical_edge_count = edge_count;
    cover_request.source_count = source_count;
    cover_request.destination_count = destination_count;
    cover_request.forward = forward_cover;
    cover_request.forward_capacity = edge_count;
    cover_request.transpose = transpose_cover;
    cover_request.transpose_capacity = edge_count;
    assert(sm70::build_transpose_cover_v1(cover_request)
        == sm70::transpose_cover_status_v1::success);

    __half edge_values[edge_count] = {
        __float2half(0.5f), __float2half(-1.0f), __float2half(2.0f)};
    __half destination_gradient[destination_count * width]{};
    for (std::uint32_t index = 0u; index < destination_count * width; ++index)
        destination_gradient[index] = __float2half(
            static_cast<float>(static_cast<int>(index % 7u) - 2));
    sm70::target_edge_placement_v1 *device_transpose = nullptr;
    __half *device_values = nullptr;
    __half *device_destination_gradient = nullptr;
    float *device_source_gradient = nullptr;
    require_cuda(cudaMalloc(&device_transpose, sizeof(transpose_cover)));
    require_cuda(cudaMalloc(&device_values, sizeof(edge_values)));
    require_cuda(cudaMalloc(&device_destination_gradient,
        sizeof(destination_gradient)));
    require_cuda(cudaMalloc(&device_source_gradient,
        source_count * width * sizeof(float)));
    require_cuda(cudaMemcpy(device_transpose, transpose_cover,
        sizeof(transpose_cover), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_values, edge_values, sizeof(edge_values),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_destination_gradient, destination_gradient,
        sizeof(destination_gradient), cudaMemcpyHostToDevice));
    sm70::transpose_relation_apply_request_v1 transpose_request{};
    transpose_request.transpose_cover = device_transpose;
    transpose_request.logical_edge_values = device_values;
    transpose_request.logical_edge_count = edge_count;
    transpose_request.destination_gradient = device_destination_gradient;
    transpose_request.destination_count = destination_count;
    transpose_request.source_count = source_count;
    transpose_request.dense_width = width;
    transpose_request.source_gradient = device_source_gradient;
    assert(sm70::enqueue_transpose_relation_apply_v1(transpose_request)
        == sm70::transpose_relation_apply_status_v1::success);
    float source_gradient[source_count * width]{};
    require_cuda(cudaMemcpy(source_gradient, device_source_gradient,
        sizeof(source_gradient), cudaMemcpyDeviceToHost));
    for (std::uint32_t source = 0u; source < source_count; ++source) {
        for (std::uint32_t column = 0u; column < width; ++column) {
            float expected = 0.0f;
            for (std::uint32_t edge = 0u; edge < edge_count; ++edge) {
                if (relation_edges[edge].source_index != source) continue;
                expected += __half2float(edge_values[edge]) * __half2float(
                    destination_gradient[
                        relation_edges[edge].destination_index * width + column]);
            }
            assert(std::fabs(source_gradient[source * width + column]
                - expected) < 1.0e-5f);
        }
    }

    sm70::support_logical_edge_v1 support_edges[edge_count]{};
    projection::projection_value_map_v1 value_map[edge_count]{};
    for (std::uint32_t edge = 0u; edge < edge_count; ++edge) {
        support_edges[edge].logical_edge_id.value = relation_edges[edge].logical_edge_id;
        support_edges[edge].source_index = relation_edges[edge].source_index;
        support_edges[edge].destination_index = relation_edges[edge].destination_index;
        value_map[edge].logical_edge_id = support_edges[edge].logical_edge_id;
        value_map[edge].region_kind = edge == 2u
            ? projection::physical_region_kind_v1::residual
            : projection::physical_region_kind_v1::mma;
        value_map[edge].region_index = edge;
        value_map[edge].projection_slot = edge;
    }
    const std::uint8_t source_support[source_count] = {1u, 1u};
    const std::uint8_t destination_support[destination_count] = {1u, 1u, 1u};
    sm70::support_projection_edge_v1 selected_edges[edge_count]{};
    sm70::contract_projection_request_v1 projection_request{};
    projection_request.logical_edges = support_edges;
    projection_request.physical_value_map = value_map;
    projection_request.logical_edge_count = edge_count;
    projection_request.source_support = source_support;
    projection_request.source_count = source_count;
    projection_request.destination_support = destination_support;
    projection_request.destination_count = destination_count;
    projection_request.selected_edges = selected_edges;
    projection_request.selected_capacity = edge_count;
    sm70::contract_projection_result_v1 projection_result{};
    assert(sm70::prepare_contract_projection_v1(
        projection_request, &projection_result)
        == sm70::contract_projection_status_v1::success);
    assert(projection_result.selected_edge_count == edge_count);

    __half source_features[source_count * width]{};
    __half destination_features[destination_count * width]{};
    for (std::uint32_t index = 0u; index < source_count * width; ++index)
        source_features[index] = __float2half(static_cast<float>(index % 4u));
    for (std::uint32_t index = 0u; index < destination_count * width; ++index)
        destination_features[index] = __float2half(
            static_cast<float>(static_cast<int>(index % 5u) - 1));
    sm70::support_logical_edge_v1 *device_support_edges = nullptr;
    sm70::support_projection_edge_v1 *device_selected_edges = nullptr;
    __half *device_source_features = nullptr;
    __half *device_destination_features = nullptr;
    float *device_contraction = nullptr;
    require_cuda(cudaMalloc(&device_support_edges, sizeof(support_edges)));
    require_cuda(cudaMalloc(&device_selected_edges, sizeof(selected_edges)));
    require_cuda(cudaMalloc(&device_source_features, sizeof(source_features)));
    require_cuda(cudaMalloc(&device_destination_features,
        sizeof(destination_features)));
    require_cuda(cudaMalloc(&device_contraction, edge_count * sizeof(float)));
    require_cuda(cudaMemcpy(device_support_edges, support_edges,
        sizeof(support_edges), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_selected_edges, selected_edges,
        sizeof(selected_edges), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_source_features, source_features,
        sizeof(source_features), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_destination_features, destination_features,
        sizeof(destination_features), cudaMemcpyHostToDevice));
    sm70::contract_on_support_request_v1 contraction_request{};
    contraction_request.logical_edges = device_support_edges;
    contraction_request.logical_edge_count = edge_count;
    contraction_request.selected_edges = device_selected_edges;
    contraction_request.selected_edge_count = edge_count;
    contraction_request.source_features = device_source_features;
    contraction_request.source_count = source_count;
    contraction_request.destination_features = device_destination_features;
    contraction_request.destination_count = destination_count;
    contraction_request.dense_width = width;
    contraction_request.logical_edge_output = device_contraction;
    assert(sm70::enqueue_contract_on_support_v1(contraction_request)
        == sm70::contract_on_support_status_v1::success);
    float contraction[edge_count]{};
    require_cuda(cudaMemcpy(contraction, device_contraction,
        sizeof(contraction), cudaMemcpyDeviceToHost));
    for (std::uint32_t edge = 0u; edge < edge_count; ++edge) {
        float expected = 0.0f;
        for (std::uint32_t column = 0u; column < width; ++column)
            expected += __half2float(source_features[
                support_edges[edge].source_index * width + column])
                * __half2float(destination_features[
                    support_edges[edge].destination_index * width + column]);
        assert(std::fabs(contraction[edge] - expected) < 1.0e-5f);
    }

    using kind = operation::relation_algebra_kind_v1;
    exchange_trace_v1 trace{0u, 40u};
    const kind kinds[] = {kind::contract_on_support, kind::edge_map_or_gate,
        kind::segment_normalize, kind::relation_apply};
    sm70::prepared_exchange_step_v1 steps[4]{};
    for (std::uint32_t index = 0u; index < 4u; ++index) {
        steps[index].kind = kinds[index];
        steps[index].execute = &record_exchange_step_v1;
        steps[index].context = &trace;
        steps[index].input_generation = 40u + index;
        steps[index].output_generation = 41u + index;
    }
    sm70::prepared_exchange_program_v1 program{};
    program.step_count = 4u;
    program.steps = steps;
    program.stream = device_contraction;
    assert(sm70::run_prepared_exchange_program_v1(program)
        == sm70::exchange_program_status_v1::success);
    assert(trace.calls == 4u && trace.generation == 44u);
    steps[2].input_generation = 99u;
    assert(sm70::run_prepared_exchange_program_v1(program)
        == sm70::exchange_program_status_v1::invalid_argument);

    sm70::transpose_relation_apply_request_v1 invalid_transpose{};
    assert(sm70::enqueue_transpose_relation_apply_v1(invalid_transpose)
        == sm70::transpose_relation_apply_status_v1::invalid_argument);
    sm70::contract_on_support_request_v1 invalid_contraction{};
    assert(sm70::enqueue_contract_on_support_v1(invalid_contraction)
        == sm70::contract_on_support_status_v1::invalid_argument);

    require_cuda(cudaFree(device_contraction));
    require_cuda(cudaFree(device_destination_features));
    require_cuda(cudaFree(device_source_features));
    require_cuda(cudaFree(device_selected_edges));
    require_cuda(cudaFree(device_support_edges));
    require_cuda(cudaFree(device_source_gradient));
    require_cuda(cudaFree(device_destination_gradient));
    require_cuda(cudaFree(device_values));
    require_cuda(cudaFree(device_transpose));
    return 0;
}
