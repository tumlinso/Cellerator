#include <Cellerator/compute/math/operation_core/cusparse_csr_candidate.hh>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace core = cellerator::compute::math::core;
namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace runtime = cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "cusparse_csr_candidate_test: " << message << '\n';
    std::abort();
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::cerr << "cusparse_csr_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "cusparse_csr_candidate_test: " << message << ": "
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
    ~device_array() { if (data != nullptr) (void) cudaFree(data); }
};

template<typename T>
void upload(device_array<T> &device, const std::vector<T> &host) {
    require(device.size >= host.size(), "upload capacity guard");
    if (!host.empty())
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

core::numeric_policy numeric() {
    core::numeric_policy result{};
    result.sparse_storage = execution::numeric_type::f32;
    result.dense_storage = execution::numeric_type::f32;
    result.output_storage = execution::numeric_type::f32;
    result.multiply = execution::numeric_type::f32;
    result.accumulation = execution::numeric_type::f32;
    result.scalar = execution::numeric_type::u32;
    result.bias = execution::numeric_type::invalid;
    return result;
}

execution::dense_tensor_view dense(void *pointer,
    execution::axis_identity major_axis,
    execution::axis_identity column_axis,
    std::uint32_t major,
    std::uint32_t columns,
    int device) {
    execution::dense_tensor_view result{};
    result.data = pointer;
    result.location = device_location(device);
    result.value_type = execution::numeric_type::f32;
    result.rank = columns == 1u ? 1u : 2u;
    result.axes[0] = major_axis;
    result.shape[0] = major;
    result.stride[0] = columns == 1u ? 1 : columns;
    if (columns != 1u) {
        result.axes[1] = column_axis;
        result.shape[1] = columns;
        result.stride[1] = 1;
    }
    return result;
}

std::vector<float> reference(
    const std::vector<std::uint32_t> &row_offsets,
    const std::vector<std::uint32_t> &features,
    const std::vector<float> &values,
    const std::vector<float> &input,
    std::uint32_t columns) {
    const std::uint32_t rows = row_offsets.size() - 1u;
    std::vector<float> result(static_cast<std::size_t>(rows) * columns, 0.0f);
    for (std::uint32_t row = 0u; row < rows; ++row)
        for (std::uint32_t edge = row_offsets[row];
             edge < row_offsets[row + 1u]; ++edge)
            for (std::uint32_t column = 0u; column < columns; ++column)
                result[static_cast<std::size_t>(row) * columns + column]
                    += values[edge]
                        * input[static_cast<std::size_t>(features[edge])
                            * columns + column];
    return result;
}

void require_close(
    const std::vector<float> &actual,
    const std::vector<float> &expected) {
    require(actual.size() == expected.size(), "result size mismatch");
    for (std::size_t index = 0u; index < actual.size(); ++index)
        require(std::fabs(actual[index] - expected[index]) < 1.0e-4f,
            "cuSPARSE numerical mismatch");
}

void run_width(std::uint32_t columns) {
    constexpr std::uint32_t rows = 3u;
    constexpr std::uint32_t features = 5u;
    const std::vector<std::uint32_t> row_offsets{0u, 3u, 5u, 8u};
    const std::vector<std::uint32_t> feature_ids{
        0u, 2u, 4u, 1u, 3u, 0u, 1u, 4u};
    std::vector<float> values{
        1.0f, -0.5f, 2.0f, 0.25f, 1.5f, -1.0f, 0.75f, 0.5f};
    std::vector<float> values_b = values;
    for (auto &value : values_b) value *= 2.0f;
    std::vector<float> input(static_cast<std::size_t>(features) * columns);
    std::vector<float> input_b(input.size());
    for (std::size_t index = 0u; index < input.size(); ++index) {
        input[index] = static_cast<float>((index % 11u) + 1u) * 0.125f;
        input_b[index] = static_cast<float>((index % 7u) + 1u) * -0.25f;
    }

    device_array<std::uint32_t> device_offsets{row_offsets.size()};
    device_array<std::uint32_t> device_features{feature_ids.size()};
    device_array<float> device_values{values.size()};
    device_array<float> device_values_b{values_b.size()};
    device_array<float> device_input{input.size()};
    device_array<float> device_input_b{input_b.size()};
    const std::size_t result_count = static_cast<std::size_t>(rows) * columns;
    device_array<float> guarded_output{result_count + 2u};
    device_array<float> guarded_output_b{result_count + 2u};
    upload(device_offsets, row_offsets);
    upload(device_features, feature_ids);
    upload(device_values, values);
    upload(device_values_b, values_b);
    upload(device_input, input);
    upload(device_input_b, input_b);
    std::vector<float> guards(result_count + 2u, 12345.0f);
    upload(guarded_output, guards);
    upload(guarded_output_b, guards);

    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    runtime::execution_session session{};
    runtime::execution_session_options options{};
    options.device = device;
    require(runtime::init_session(&session, options)
            == runtime::session_status::success,
        "session initialization");
    require(runtime::prepare_stream_libraries(&session, 0u)
            == runtime::session_status::success,
        "session library preparation");

    cm::execution_csr_view csr{};
    csr.row_count = rows;
    csr.full_row_count = rows;
    csr.feature_count = features;
    csr.nnz_count = values.size();
    csr.value_size_bytes = sizeof(float);
    csr.row_domain_identity = 0x3003u;
    csr.structure.identity_version = cm::execution_csr_structure_identity_version;
    csr.structure.value = 0x7072u;
    csr.feature_order.kind = cm::feature_order_kind::packed;
    csr.feature_order.feature_count = features;
    csr.feature_order.feature_axis_identity_version = 1u;
    csr.feature_order.feature_axis_identity = 0x5005u;
    csr.feature_order.packing_geometry_identity = 0x1001u;
    csr.row_offsets = device_offsets.data;
    csr.execution_feature_ids = device_features.data;
    csr.values = device_values.data;

    const auto feature_axis = axis(10u);
    const auto row_axis = axis(20u);
    const auto column_axis = axis(30u);
    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {{11u, 12u}, {21u, 1u}, {1u}};
    const core::projection_key projection{{31u, 32u}, {42u, 1u},
        core::projection_kind::csr, cm::execution_csr_schema_version, 1u};
    core::operation_problem problem{};
    problem.kind = columns == 1u
        ? core::operation_kind::weighted_relation_reduce
        : core::operation_kind::sparse_dense_multiply;
    problem.operation = {81u, columns};
    problem.input_count = 1u;
    problem.output_count = 1u;
    problem.logical_work_items = static_cast<std::uint64_t>(values.size())
        * columns;
    const core::prepare_policy policy{
        true, false, true, true, 16u, 0u, 0u};
    core::cusparse_csr_prepared_state state{};
    core::prepared_operation prepared{};
    require(core::prepare_cusparse_csr_operation(problem, structures,
        projection, numeric(), policy, csr, &session, 0u, columns,
        device_input.data, guarded_output.data + 1u, feature_axis, row_axis,
        column_axis, &state, &prepared), "cuSPARSE candidate preparation");
    const auto costs = core::cusparse_csr_costs(state);
    require(costs.descriptor_state_bytes == sizeof(state)
        && costs.descriptor_create_calls == 3u
        && costs.transient_workspace_bytes == 0u
        && costs.preprocess_calls == (columns == 1u ? 0u : 1u)
        && prepared.binding_contract.workspace.minimum_bytes == 0u
        && prepared.backend == core::backend_kind::vendor_library
        && (prepared.capability_flags & core::candidate_graph_capture) == 0u,
        "complete cost and capability contract");
    require(runtime::seal_session(&session) == runtime::session_status::success,
        "seal prepared session");
    const std::uint64_t allocations_after_prepare =
        session.accounting.plan.allocation_count;

    execution::relation_structure relation{};
    relation.identity = structures.structures[0].runtime;
    relation.epoch = structures.structures[0].epoch;
    relation.source_axis = feature_axis;
    relation.destination_axis = row_axis;
    relation.projections = {1u, 1u};
    relation.logical_edge_count = values.size();
    execution::value_plane plane{};
    plane.structure = relation.identity;
    plane.structure_epoch_value = relation.epoch;
    plane.values = device_values.data;
    plane.location = device_location(device);
    plane.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {2u};
    plane.element_count = values.size();
    plane.value_bytes = values.size() * sizeof(float);
    execution::value_binding value_binding{&plane, plane.generation};
    execution::biological_operand_view input_operand{}, output_operand{};
    input_operand.kind = execution::operand_kind::dense_tensor;
    input_operand.storage.dense = dense(device_input.data, feature_axis,
        column_axis, features, columns, device);
    output_operand.kind = execution::operand_kind::dense_tensor;
    output_operand.storage.dense = dense(guarded_output.data + 1u, row_axis,
        column_axis, rows, columns, device);
    const auto runtime_binding = runtime::bind_launch(&session, 0u, 0u);
    require(runtime_binding.status == runtime::session_status::success,
        "session launch binding");
    execution::launch_bindings launch{};
    launch.structures = &relation;
    launch.inputs = &input_operand;
    launch.outputs = &output_operand;
    launch.values = &value_binding;
    launch.input_count = 1u;
    launch.output_count = 1u;
    launch.value_count = 1u;
    launch.structure_count = 1u;
    launch.stream = {runtime_binding.execution.stream, device, 0u};
    launch.workspace = {nullptr, 0u, device_location(device)};
    require(core::run_prepared_operation(prepared, launch),
        "cuSPARSE prepared run");
    require_cuda(cudaStreamSynchronize(runtime_binding.execution.stream),
        "synchronize first run");
    std::vector<float> downloaded(result_count + 2u);
    require_cuda(cudaMemcpy(downloaded.data(), guarded_output.data,
        downloaded.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download guarded output");
    require(downloaded.front() == 12345.0f
        && downloaded.back() == 12345.0f, "output guard corruption");
    require_close(std::vector<float>(downloaded.begin() + 1,
        downloaded.end() - 1), reference(row_offsets, feature_ids,
            values, input, columns));

    plane.values = device_values_b.data;
    plane.generation = {3u};
    value_binding.expected_generation = plane.generation;
    input_operand.storage.dense.data = device_input_b.data;
    output_operand.storage.dense.data = guarded_output_b.data + 1u;
    require(core::run_prepared_operation(prepared, launch),
        "pointer-rebound cuSPARSE run");
    require_cuda(cudaStreamSynchronize(runtime_binding.execution.stream),
        "synchronize pointer-rebound run");
    require_cuda(cudaMemcpy(downloaded.data(), guarded_output_b.data,
        downloaded.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download rebound guarded output");
    require(downloaded.front() == 12345.0f
        && downloaded.back() == 12345.0f, "rebound output guard corruption");
    require_close(std::vector<float>(downloaded.begin() + 1,
        downloaded.end() - 1), reference(row_offsets, feature_ids,
            values_b, input_b, columns));
    require(session.accounting.plan.allocation_count == allocations_after_prepare,
        "run performed a session-visible allocation");

    core::clear_cusparse_csr_prepared_state(&state);
    runtime::clear_session(&session);
}

void test_registration_and_invalid_width() {
    core::candidate_registry registry{};
    require(core::register_cusparse_csr_candidates(&registry),
        "cuSPARSE candidate registration");
    require(registry.size == 2u
        && registry.candidates[0].backend == core::backend_kind::vendor_library
        && registry.candidates[1].backend == core::backend_kind::vendor_library,
        "cuSPARSE registry inventory");
    require(core::register_cusparse_csr_candidates(&registry).code
            == core::operation_status_code::duplicate_candidate,
        "duplicate candidate invalid registration rejection");

    core::cusparse_csr_prepared_state state{};
    core::prepared_operation prepared{};
    core::operation_problem problem{};
    core::structure_set_key structures{};
    core::projection_key projection{};
    cm::execution_csr_view csr{};
    require(core::prepare_cusparse_csr_operation(problem, structures,
        projection, numeric(), {}, csr, nullptr, 0u, 15u,
        nullptr, nullptr, {}, {}, {}, &state, &prepared).code
            == core::operation_status_code::invalid_argument,
        "invalid non-envelope width and null capacity rejection");
}

} // namespace

int main() {
    test_registration_and_invalid_width();
    const std::uint32_t widths[] = {1u, 16u, 17u, 31u, 32u, 48u, 64u};
    for (const std::uint32_t width : widths) run_width(width);
    return 0;
}
