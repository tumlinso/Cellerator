#include <Cellerator/compute/operation/edge_map_or_gate.hh>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace operation = cellerator::compute::operation;
namespace execution = cellerator::execution;

namespace {

template<typename Condition>
void require(Condition condition, const char *message) {
    if (!static_cast<bool>(condition)) {
        std::fprintf(stderr, "edge_map_or_gate_test: %s\n", message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "edge_map_or_gate_test: %s: %s\n", message,
            cudaGetErrorString(status));
        std::exit(1);
    }
}

template<typename T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;

    explicit device_buffer(std::size_t size) : count(size) {
        if (count != 0u)
            require_cuda(cudaMalloc(&data, count * sizeof(T)),
                "allocate device buffer");
    }
    ~device_buffer() {
        if (data != nullptr) cudaFree(data);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
};

execution::axis_identity axis(std::uint32_t seed) {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

execution::device_location location(int device) {
    return {execution::residency_kind::device, {}, device, 1u};
}

execution::relation_structure structure() {
    execution::relation_structure result{};
    result.identity = {11u, 1u};
    result.epoch = {7u};
    result.source_axis = axis(20u);
    result.destination_axis = axis(30u);
    result.projections = {13u, 1u};
    result.logical_edge_count = 5u;
    return result;
}

execution::value_plane plane(float *values,
    execution::device_location where,
    execution::value_layout_kind layout,
    std::uint64_t generation) {
    execution::value_plane result{};
    result.structure = {11u, 1u};
    result.structure_epoch_value = {7u};
    result.values = values;
    result.location = where;
    result.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    result.quantization = {execution::quantization_kind::none,
        execution::numeric_type::invalid, execution::numeric_type::invalid,
        0u, nullptr, nullptr, 0u};
    result.layout = layout;
    result.generation = {generation};
    result.element_count = 5u;
    result.value_bytes = 5u * sizeof(float);
    return result;
}

operation::edge_map_or_gate_plan_v1 plan(
    operation::edge_operation_v1 kind,
    execution::value_layout_kind input_layout,
    execution::value_layout_kind output_layout) {
    operation::edge_map_or_gate_plan_v1 result{};
    result.operation = kind;
    result.input_layout = input_layout;
    result.output_layout = output_layout;
    result.structure = {{11u, 1u}, {7u}};
    result.projection_identity = {41u, 42u};
    result.projection = {43u, 1u};
    result.logical_edge_order = {44u, 45u};
    result.logical_edge_count = 5u;
    if (kind == operation::edge_operation_v1::multiplicative_gate)
        result.gate_type = execution::numeric_type::f32;
    else if (kind == operation::edge_operation_v1::predicate_gate)
        result.gate_type = execution::numeric_type::u8;
    return result;
}

template<typename T, std::size_t N>
void upload(T *destination, const std::array<T, N> &source,
    cudaStream_t stream, const char *message) {
    require_cuda(cudaMemcpyAsync(destination, source.data(), sizeof(source),
        cudaMemcpyHostToDevice, stream), message);
}

template<std::size_t N>
std::array<float, N> download(const float *source, cudaStream_t stream) {
    std::array<float, N> result{};
    require_cuda(cudaMemcpyAsync(result.data(), source, sizeof(result),
        cudaMemcpyDeviceToHost, stream), "download result");
    require_cuda(cudaStreamSynchronize(stream), "synchronize focused test");
    return result;
}

void require_values(const std::array<float, 5> &actual,
    const std::array<float, 5> &expected, const char *message) {
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        if (std::isnan(expected[index]))
            require(std::isnan(actual[index]), message);
        else
            require(actual[index] == expected[index], message);
    }
}

} // namespace

int main() {
    int device = 0;
    require_cuda(cudaGetDevice(&device), "query device");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device),
        "query device properties");
    require(properties.major == 7 && properties.minor == 0,
        "focused test requires sm_70");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create caller stream");
    const auto where = location(device);

    device_buffer<float> first(5u);
    device_buffer<float> second(5u);
    device_buffer<std::uint32_t> logical_to_projection(5u);
    device_buffer<std::uint32_t> projection_to_logical(5u);
    device_buffer<float> multiplier(5u);
    device_buffer<std::uint8_t> predicate(5u);
    upload(first.data, std::array<float, 5>{10, 20, 30, 40, 50}, stream,
        "upload logical values");
    upload(logical_to_projection.data,
        std::array<std::uint32_t, 5>{2, 4, 1, 3, 0}, stream,
        "upload logical-to-projection map");
    upload(projection_to_logical.data,
        std::array<std::uint32_t, 5>{4, 2, 0, 3, 1}, stream,
        "upload projection-to-logical map");
    upload(multiplier.data, std::array<float, 5>{1, 2, 0.5f, -1, 0}, stream,
        "upload multiplicative gate");
    upload(predicate.data, std::array<std::uint8_t, 5>{0, 1, 0, 1, 1}, stream,
        "upload predicate gate");

    const auto relation = structure();
    const execution::value_position_map_view positions{{11u, 1u}, {7u},
        execution::value_map_direction::forward, {}, logical_to_projection.data,
        projection_to_logical.data, where, 5u};
    const execution::stream_context caller_stream{stream, device, 0u};
    const execution::transient_workspace no_workspace{nullptr, 0u, where};
    const operation::logical_edge_gate_view_v1 no_gate{};

    auto map_plan = plan(operation::edge_operation_v1::map,
        execution::value_layout_kind::logical_edge_order,
        execution::value_layout_kind::projection_local_order);
    require(operation::validate_edge_map_or_gate_plan_v1(map_plan),
        "validate logical-to-projection map plan");
    require(operation::query_edge_map_or_gate_workspace_v1(map_plan).minimum_bytes
            == 0u,
        "edge transform unexpectedly requires workspace");
    auto logical_input = plane(first.data, where,
        execution::value_layout_kind::logical_edge_order, 1u);
    auto projected_output = plane(second.data, where,
        execution::value_layout_kind::projection_local_order, 2u);
    std::size_t free_before = 0u, total_before = 0u;
    require_cuda(cudaMemGetInfo(&free_before, &total_before),
        "measure before edge map launch");
    require(operation::run_edge_map_or_gate_v1(map_plan, relation,
        {&logical_input, {1u}}, projected_output, positions, no_gate,
        caller_stream, no_workspace), "map logical values into projection order");
    std::size_t free_after = 0u, total_after = 0u;
    require_cuda(cudaMemGetInfo(&free_after, &total_after),
        "measure after edge map launch");
    require(free_before == free_after && total_before == total_after,
        "edge map allocated device memory");
    require_values(download<5>(second.data, stream), {50, 30, 10, 40, 20},
        "logical-to-projection mapping is wrong");

    auto gate_plan = plan(operation::edge_operation_v1::multiplicative_gate,
        execution::value_layout_kind::projection_local_order,
        execution::value_layout_kind::logical_edge_order);
    const operation::logical_edge_gate_view_v1 multiplicative{
        multiplier.data, where, {44u, 45u}, execution::numeric_type::f32,
        {}, 5u};
    auto projected_input = plane(second.data, where,
        execution::value_layout_kind::projection_local_order, 2u);
    auto gated_logical = plane(first.data, where,
        execution::value_layout_kind::logical_edge_order, 3u);
    require(operation::run_edge_map_or_gate_v1(gate_plan, relation,
        {&projected_input, {2u}}, gated_logical, positions, multiplicative,
        caller_stream, no_workspace), "fuse projection map and multiplicative gate");
    require_values(download<5>(first.data, stream), {10, 40, 15, -40, 0},
        "multiplicative gate is wrong");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    upload(second.data, std::array<float, 5>{50, 30, nan, 40, 20}, stream,
        "upload projected values with NaN");
    auto predicate_plan = plan(operation::edge_operation_v1::predicate_gate,
        execution::value_layout_kind::projection_local_order,
        execution::value_layout_kind::projection_local_order);
    const operation::logical_edge_gate_view_v1 predicates{
        predicate.data, where, {44u, 45u}, execution::numeric_type::u8,
        {}, 5u};
    projected_input.generation = {4u};
    projected_output.values = first.data;
    projected_output.generation = {5u};
    require(operation::run_edge_map_or_gate_v1(predicate_plan, relation,
        {&projected_input, {4u}}, projected_output, positions, predicates,
        caller_stream, no_workspace), "apply predicate gate in projection order");
    require_values(download<5>(first.data, stream), {50, 0, 0, 40, 20},
        "predicate gate or false-NaN behavior is wrong");

    auto alias_output = projected_input;
    alias_output.values = logical_input.values;
    auto alias_status = operation::run_edge_map_or_gate_v1(map_plan, relation,
        {&logical_input, {1u}}, alias_output, positions, no_gate,
        caller_stream, no_workspace);
    require(alias_status.code == operation::edge_map_or_gate_status_v1::illegal_alias,
        "in-place permutation was accepted");
    auto stale = gate_plan;
    stale.structure.epoch.value += 1u;
    require(operation::run_edge_map_or_gate_v1(stale, relation,
        {&projected_input, {4u}}, gated_logical, positions, multiplicative,
        caller_stream, no_workspace).code
            == operation::edge_map_or_gate_status_v1::stale_structure,
        "stale structure epoch was accepted");
    require(operation::run_edge_map_or_gate_v1(gate_plan, relation,
        {&projected_input, {99u}}, gated_logical, positions, multiplicative,
        caller_stream, no_workspace).code
            == operation::edge_map_or_gate_status_v1::stale_value,
        "stale value generation was accepted");

    require_cuda(cudaStreamDestroy(stream), "destroy caller stream");
    std::puts("edge map-or-gate passed");
    return 0;
}
