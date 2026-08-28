#include <Cellerator/compute/candidate/tensor_core/v100_dense_fragment_candidate.hh>
#include <Cellerator/compute/candidate/tensor_core/v100_dense_fragment_plan.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace core = cellerator::compute::math::core;
namespace tc = cellerator::compute::math::tensor_core;
namespace execution = cellerator::execution;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "v100_dense_fragment_candidate_test: " << message << '\n';
    std::abort();
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::cerr << "v100_dense_fragment_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", binding=" << static_cast<unsigned>(status.binding)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "v100_dense_fragment_candidate_test: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    std::abort();
}

template<typename T>
struct device_array {
    T *data = nullptr;
    std::size_t size = 0u;

    explicit device_array(std::size_t count) : size(count) {
        if (count != 0u)
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
                count * sizeof(T)), "cudaMalloc");
    }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
    ~device_array() { if (data != nullptr) cudaFree(data); }
};

template<typename T>
void upload(device_array<T> &device, const std::vector<T> &host) {
    require(device.size >= host.size(), "device upload capacity");
    require_cuda(cudaMemcpy(device.data, host.data(),
        host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

execution::device_location device_location(int ordinal) {
    return {execution::residency_kind::device, {}, ordinal, 0u};
}

execution::dense_tensor_view dense_matrix(void *pointer,
    execution::numeric_type type, execution::axis_identity major_axis,
    execution::axis_identity minor_axis, std::uint64_t rows,
    std::uint64_t columns, int ordinal) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(ordinal);
    view.value_type = type;
    view.rank = 2u;
    view.axes[0] = major_axis;
    view.axes[1] = minor_axis;
    view.shape[0] = rows;
    view.shape[1] = columns;
    view.stride[0] = static_cast<std::int64_t>(columns);
    view.stride[1] = 1;
    return view;
}

core::numeric_policy numeric() {
    core::numeric_policy value{};
    value.sparse_storage = execution::numeric_type::f16;
    value.dense_storage = execution::numeric_type::f16;
    value.output_storage = execution::numeric_type::f32;
    value.multiply = execution::numeric_type::f16;
    value.accumulation = execution::numeric_type::f32;
    value.scalar = execution::numeric_type::f32;
    value.bias = execution::numeric_type::invalid;
    return value;
}

std::vector<float> reference(const std::vector<__half> &relation,
    const std::vector<__half> &rhs, std::uint32_t width) {
    std::vector<float> result(16u * width, 0.0f);
    for (std::uint32_t row = 0u; row < 16u; ++row)
        for (std::uint32_t source = 0u; source < 16u; ++source)
            for (std::uint32_t column = 0u; column < width; ++column)
                result[static_cast<std::size_t>(row) * width + column]
                    += __half2float(relation[row * 16u + source])
                        * __half2float(rhs[
                            static_cast<std::size_t>(source) * width + column]);
    return result;
}

void compare(const std::vector<float> &actual,
    const std::vector<float> &expected) {
    require(actual.size() == expected.size(), "result extent");
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        const float tolerance = 2.0e-3f
            + 2.0e-3f * std::abs(expected[index]);
        require(std::abs(actual[index] - expected[index]) <= tolerance,
            "independent FP16/FP32 referee mismatch");
    }
}

void test_projection_maps_and_explicit_residual() {
    std::vector<std::uint64_t> offsets(18u, 0u);
    std::vector<std::uint32_t> indices;
    for (std::uint32_t row = 0u; row < 16u; ++row) {
        for (std::uint32_t source = 0u; source < 8u; ++source)
            indices.push_back(source);
        offsets[row + 1u] = indices.size();
    }
    indices.push_back(16u);
    offsets[17u] = indices.size();
    tc::destination_row_csr_support_view support{offsets.data(),
        indices.data(), 17u, 17u, indices.size()};
    std::vector<std::uint16_t> tile_nnz(4u);
    std::vector<std::int64_t> tile_to_fragment(4u);
    std::vector<std::uint32_t> destination_bases(1u), source_bases(1u);
    std::vector<std::uint64_t> edge_to_slot(indices.size());
    std::vector<std::uint64_t> slot_to_edge(256u);
    tc::v100_dense_fragment_plan_buffers buffers{tile_nnz.data(),
        tile_to_fragment.data(), tile_nnz.size(), destination_bases.data(),
        source_bases.data(), destination_bases.size(), edge_to_slot.data(),
        edge_to_slot.size(), slot_to_edge.data(), slot_to_edge.size()};
    tc::v100_dense_fragment_plan_requirements requirements{};
    require(tc::build_v100_dense_fragment_plan_host(
            support, buffers, &requirements)
            == tc::dense_fragment_plan_status::ok,
        "build exact dense-fragment logical-edge maps");
    require(requirements.qualified_fragment_count == 1u
        && requirements.maximum_tile_nnz == 128u
        && requirements.residual_edge_count == 1u
        && destination_bases[0] == 0u && source_bases[0] == 0u,
        "qualification threshold and explicit tail residual");
    for (std::uint64_t edge = 0u; edge < 128u; ++edge) {
        const std::uint64_t slot = edge_to_slot[edge];
        require(slot != tc::invalid_dense_fragment_position
            && slot_to_edge[slot] == edge,
            "forward and inverse logical-edge maps");
    }
    require(edge_to_slot.back() == tc::invalid_dense_fragment_position,
        "tail edge remains owned by an explicit residual candidate");
}

void run_generation(std::uint32_t width, const std::vector<__half> &values,
    const std::vector<__half> &rhs, device_array<__half> &device_values,
    device_array<__half> &device_rhs, device_array<float> &device_output,
    core::prepared_operation &prepared, execution::launch_bindings &launch,
    execution::value_plane &plane, cudaStream_t stream) {
    upload(device_values, values);
    upload(device_rhs, rhs);
    require_cuda(cudaMemsetAsync(device_output.data, 0,
        device_output.size * sizeof(float), stream), "zero output");
    require(core::run_prepared_operation(prepared, launch),
        "run prepared dense-fragment operation");
    require_cuda(cudaStreamSynchronize(stream), "synchronize candidate");
    std::vector<float> actual(device_output.size);
    require_cuda(cudaMemcpy(actual.data(), device_output.data,
        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download candidate output");
    compare(actual, reference(values, rhs, width));
    require(plane.generation.value != 0u,
        "value generation remains explicit");
}

} // namespace

int main() {
    test_projection_maps_and_explicit_residual();
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
    require(properties.major == 7 && properties.minor == 0,
        "test requires the task's sm_70 resource");

    constexpr std::uint32_t width = 32u;
    const execution::structure_id structure_id{101u, 102u};
    const execution::structure_handle structure_handle{11u, 1u};
    const execution::structure_epoch epoch{7u};
    const execution::projection_id projection_id{201u, 202u};
    const execution::projection_handle projection_handle{21u, 1u};
    const execution::axis_identity source_axis = axis(10u);
    const execution::axis_identity destination_axis = axis(20u);
    const execution::axis_identity dense_axis = axis(30u);

    std::vector<std::uint32_t> destination_bases{0u};
    std::vector<std::uint32_t> source_bases{0u};
    device_array<std::uint32_t> device_destination_bases(1u);
    device_array<std::uint32_t> device_source_bases(1u);
    upload(device_destination_bases, destination_bases);
    upload(device_source_bases, source_bases);

    tc::v100_dense_fragment_projection_view view{};
    view.fragment_count = 1u;
    view.destination_count = 16u;
    view.source_count = 16u;
    view.logical_edge_count = 256u;
    view.packed_slot_count = 256u;
    view.persistent_structure = structure_id;
    view.runtime_structure = structure_handle;
    view.structure_epoch = epoch;
    view.persistent_projection = projection_id;
    view.runtime_projection = projection_handle;
    view.fragment_destination_bases = device_destination_bases.data;
    view.fragment_source_bases = device_source_bases.data;

    core::candidate_registry registry{};
    require(tc::register_v100_dense_fragment_candidate(&registry),
        "register bounded candidate explicitly");
    require(registry.size == 1u
        && core::same_stable_id(registry.candidates[0].identity,
            tc::v100_dense_fragment_candidate_id),
        "candidate behaves as an ordinary registry entry");

    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {structure_id, structure_handle, epoch};
    const core::projection_key projection_key{projection_id,
        projection_handle, core::projection_kind::dense_fragment,
        tc::v100_dense_fragment_schema_version,
        tc::v100_dense_fragment_variant};
    const core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::sparse_dense_multiply, 0u, {301u, 302u},
        1u, 1u, static_cast<std::uint64_t>(256u) * width};
    const core::prepare_policy policy{
        false, true, true, true, 8u, 0u, 0u};
    tc::v100_dense_fragment_prepared_state state{};
    core::prepared_operation prepared{};
    require(tc::prepare_v100_dense_fragment_operation(problem, structures,
        projection_key, numeric(), policy, view, device, width, source_axis,
        destination_axis, dense_axis, &state, &prepared),
        "prepare bounded candidate");
    require(prepared.binding_contract.output_effects[0].update
            == execution::output_update_kind::accumulate
        && prepared.binding_contract.output_effects[0]
            .requires_initialized_destination
        && execution::valid_output_effect_contract(
            prepared.binding_contract.output_effects[0])
        && prepared.binding_contract.output_orders[0].transition
            == execution::order_transition_kind::preserve
        && prepared.binding_contract.output_orders[1].transition
            == execution::order_transition_kind::preserve,
        "residual composition and output order remain explicit");

    std::vector<__half> values_a(256u), values_b(256u);
    std::vector<__half> rhs(16u * width);
    for (std::uint32_t row = 0u; row < 16u; ++row)
        for (std::uint32_t source = 0u; source < 16u; ++source) {
            const std::uint32_t slot = row * 16u + source;
            values_a[slot] = __float2half(
                static_cast<float>((slot % 11u) + 1u) / 16.0f);
            values_b[slot] = __float2half(
                static_cast<float>(static_cast<int>(slot % 9u) - 4) / 8.0f);
        }
    for (std::uint32_t source = 0u; source < 16u; ++source)
        for (std::uint32_t column = 0u; column < width; ++column)
            rhs[static_cast<std::size_t>(source) * width + column]
                = __float2half(static_cast<float>(
                    static_cast<int>((source * 7u + column * 3u) % 17u) - 8)
                    / 8.0f);

    device_array<__half> device_values(256u);
    device_array<__half> device_rhs(rhs.size());
    device_array<float> device_output(16u * width);
    execution::relation_structure relation{};
    relation.identity = structure_handle;
    relation.epoch = epoch;
    relation.source_axis = source_axis;
    relation.destination_axis = destination_axis;
    relation.projections = {1u, 1u};
    relation.logical_edge_count = 256u;
    execution::value_plane plane{};
    plane.structure = structure_handle;
    plane.structure_epoch_value = epoch;
    plane.values = device_values.data;
    plane.location = device_location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = 256u;
    plane.value_bytes = 256u * sizeof(__half);
    execution::value_binding binding{&plane, plane.generation};
    execution::biological_operand_view input{}, output{};
    input.kind = execution::operand_kind::dense_tensor;
    input.storage.dense = dense_matrix(device_rhs.data,
        execution::numeric_type::f16, source_axis, dense_axis,
        16u, width, device);
    output.kind = execution::operand_kind::dense_tensor;
    output.storage.dense = dense_matrix(device_output.data,
        execution::numeric_type::f32, destination_axis, dense_axis,
        16u, width, device);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");
    execution::launch_bindings launch{};
    launch.structures = &relation;
    launch.inputs = &input;
    launch.outputs = &output;
    launch.values = &binding;
    launch.input_count = 1u;
    launch.output_count = 1u;
    launch.value_count = 1u;
    launch.structure_count = 1u;
    launch.stream = {stream, device, 0u};
    launch.workspace = {nullptr, 0u, device_location(device)};

    run_generation(width, values_a, rhs, device_values, device_rhs,
        device_output, prepared, launch, plane, stream);
    plane.generation = {2u};
    binding.expected_generation = plane.generation;
    const void *persistent = prepared.persistent.data;
    run_generation(width, values_b, rhs, device_values, device_rhs,
        device_output, prepared, launch, plane, stream);
    require(prepared.persistent.data == persistent,
        "changing value generation did not rebuild structure");

    binding.expected_generation = {3u};
    require(core::run_prepared_operation(prepared, launch).binding
            == execution::binding_validation_code::stale_value,
        "stale value generation rejection");
    binding.expected_generation = plane.generation;
    relation.epoch.value += 1u;
    require(core::run_prepared_operation(prepared, launch).code
            == core::operation_status_code::stale_structure,
        "stale structure epoch rejection");
    relation.epoch = epoch;

    tc::v100_dense_fragment_prepared_state rejected_state{};
    core::prepared_operation rejected{};
    core::operation_problem unsupported = problem;
    unsupported.logical_work_items = 256u * 17u;
    require(tc::prepare_v100_dense_fragment_operation(unsupported, structures,
        projection_key, numeric(), policy, view, device, 17u, source_axis,
        destination_axis, dense_axis, &rejected_state, &rejected).code
            == core::operation_status_code::unsupported_problem,
        "unsupported N tail rejection");
    core::projection_key wrong_projection = projection_key;
    wrong_projection.persistent.low += 1u;
    require(tc::prepare_v100_dense_fragment_operation(problem, structures,
        wrong_projection, numeric(), policy, view, device, width, source_axis,
        destination_axis, dense_axis, &rejected_state, &rejected).code
            == core::operation_status_code::unsupported_problem,
        "incorrect projection identity rejection");
    core::numeric_policy wrong_numeric = numeric();
    wrong_numeric.dense_storage = execution::numeric_type::f32;
    require(tc::prepare_v100_dense_fragment_operation(problem, structures,
        projection_key, wrong_numeric, policy, view, device, width, source_axis,
        destination_axis, dense_axis, &rejected_state, &rejected).code
            == core::operation_status_code::unsupported_numeric_policy,
        "unsupported numeric tuple rejection");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
