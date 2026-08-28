#include <Cellerator/compute/candidate/csr_fallback_candidate.hh>
#include <Cellerator/compute/candidate/row_masked_n1_candidate.hh>
#include <Cellerator/execution/execution_contract.hh>
#include <Cellerator/planner/end_to_end_planner.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace core = cellerator::compute::math::core;
namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "csr_fallback_candidate_test: " << message << '\n';
    std::abort();
}

void require(core::operation_status status, const char *message) {
    if (status) return;
    std::cerr << "csr_fallback_candidate_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", binding=" << static_cast<unsigned>(status.binding)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "csr_fallback_candidate_test: " << message << ": "
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

execution::dense_tensor_view dense(void *pointer,
    execution::numeric_type type, execution::axis_identity value_axis,
    std::uint64_t count, int ordinal) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(ordinal);
    view.value_type = type;
    view.rank = 1u;
    view.axes[0] = value_axis;
    view.shape[0] = count;
    view.stride[0] = 1;
    return view;
}

core::numeric_policy numeric() {
    core::numeric_policy value{};
    value.sparse_storage = execution::numeric_type::f16;
    value.dense_storage = execution::numeric_type::f32;
    value.output_storage = execution::numeric_type::f32;
    value.multiply = execution::numeric_type::f32;
    value.accumulation = execution::numeric_type::f32;
    value.scalar = execution::numeric_type::u32;
    value.bias = execution::numeric_type::invalid;
    return value;
}

void test_registry_and_planner_coexistence(
    const core::operation_problem &problem,
    const core::structure_set_key &structures,
    const core::projection_key &csr_projection) {
    core::candidate_registry registry{};
    require(core::register_row_masked_n1_candidate(&registry),
        "row-masked registration");
    require(core::register_csr_fallback_candidate(&registry),
        "CSR registration");
    require(registry.size == 2u
        && registry.candidates[1].operation
            == core::operation_kind::weighted_relation_reduce
        && registry.candidates[1].projection == core::projection_kind::csr
        && registry.candidates[1].backend == core::backend_kind::native_direct
        && registry.candidates[1].transient_bytes == 0u
        && (registry.candidates[1].capability_flags
            & core::candidate_graph_capture) == 0u,
        "truthful CSR capability record");
    require(core::register_csr_fallback_candidate(&registry).code
            == core::operation_status_code::duplicate_candidate,
        "duplicate CSR rejection");

    planner::planner_candidate candidates[2]{};
    candidates[0].identity = registry.candidates[0].identity;
    candidates[0].name = registry.candidates[0].name;
    candidates[0].operation = &registry.candidates[0];
    candidates[0].projection = {{101u, 1u}, {201u, 1u},
        core::projection_kind::native_row_masked, 1u, 1u};
    candidates[0].analytical.kernel_ns = 20.0;
    candidates[0].analytical.persistent_bytes =
        registry.candidates[0].persistent_bytes;
    candidates[0].flags = planner::planner_candidate_correct
        | planner::planner_candidate_deterministic
        | planner::planner_candidate_graph_capture;
    candidates[1].identity = registry.candidates[1].identity;
    candidates[1].name = registry.candidates[1].name;
    candidates[1].operation = &registry.candidates[1];
    candidates[1].projection = csr_projection;
    candidates[1].analytical.projection_construction_ns = 5.0;
    candidates[1].analytical.kernel_ns = 10.0;
    candidates[1].analytical.persistent_bytes =
        registry.candidates[1].persistent_bytes;
    candidates[1].flags = planner::planner_candidate_correct
        | planner::planner_candidate_deterministic
        | planner::planner_candidate_conventional;

    planner::planner_request request{};
    request.problem = problem;
    request.keys.problem.identity = problem.operation;
    require(planner::make_persistent_structure_set_key(
        structures, &request.keys.structures), "persistent structure key");
    request.keys.geometry = {{1u, 1u}, {2u, 1u}, {3u, 1u},
        {4u, 1u}, {5u, 1u}, {6u, 1u}};
    request.keys.device = {1u, 7u, 0u, 700u};
    request.keys.build = {1u, 2u, 3u, 4u};
    request.keys.policy = {8u, 8u, 8u, 1u, 1u, 1u, 0u};
    request.candidates = candidates;
    request.candidate_count = 2u;
    request.policy.shortlist_size = 2u;
    request.policy.maximum_measurements = 2u;
    request.policy.minimum_tuning_work_items = 4096u;
    request.current_evidence_revision = 1u;
    planner::planner_result result{};
    require(planner::plan_end_to_end(request, &result)
        && result.legal_count == 2u
        && result.selected == &candidates[1]
        && result.conventional_winner,
        "row-masked and CSR planner coexistence");
}

} // namespace

int main() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    device_array<std::uint32_t> row_offsets{3u};
    device_array<std::uint32_t> feature_ids{3u};
    device_array<__half> projection_values{3u};
    device_array<__half> launch_values{3u};
    device_array<__half> launch_values_b{3u};
    device_array<__half> relocated_values{3u};
    device_array<float> weights{3u};
    device_array<float> output{2u};
    device_array<float> output_b{2u};
    upload(row_offsets, std::vector<std::uint32_t>{0u, 2u, 3u});
    upload(feature_ids, std::vector<std::uint32_t>{0u, 2u, 1u});
    upload(projection_values, std::vector<__half>{
        __float2half(9.0f), __float2half(9.0f), __float2half(9.0f)});
    upload(launch_values, std::vector<__half>{
        __float2half(1.0f), __float2half(2.0f), __float2half(3.0f)});
    upload(launch_values_b, std::vector<__half>{
        __float2half(4.0f), __float2half(5.0f), __float2half(6.0f)});
    upload(relocated_values, std::vector<__half>{
        __float2half(1.0f), __float2half(2.0f), __float2half(3.0f)});
    upload(weights, std::vector<float>{2.0f, 5.0f, 7.0f});

    cm::execution_csr_view csr{};
    csr.row_count = 2u;
    csr.full_row_count = 2u;
    csr.feature_count = 3u;
    csr.nnz_count = 3u;
    csr.value_size_bytes = sizeof(__half);
    csr.row_domain_identity = 0x3003u;
    csr.structure.identity_version = cm::execution_csr_structure_identity_version;
    csr.structure.value = 0x7072u;
    csr.feature_order.kind = cm::feature_order_kind::packed;
    csr.feature_order.feature_count = 3u;
    csr.feature_order.feature_axis_identity_version = 1u;
    csr.feature_order.feature_axis_identity = 0x5005u;
    csr.feature_order.packing_geometry_identity = 0x1001u;
    csr.row_offsets = row_offsets.data;
    csr.execution_feature_ids = feature_ids.data;
    csr.values = projection_values.data;

    const execution::axis_identity feature_axis = axis(10u);
    const execution::axis_identity row_axis = axis(20u);
    const core::operation_problem problem{core::operation_core_schema_version,
        core::operation_kind::weighted_relation_reduce, 0u, {72u, 1u},
        1u, 1u, 3u};
    core::structure_set_key structures{};
    structures.count = 1u;
    structures.structures[0] = {{11u, 12u}, {21u, 1u}, {1u}};
    const core::projection_key projection{{31u, 32u}, {42u, 1u},
        core::projection_kind::csr, cm::execution_csr_schema_version, 1u};
    test_registry_and_planner_coexistence(problem, structures, projection);

    core::csr_fallback_prepared_state state{};
    core::prepared_operation prepared{};
    const core::prepare_policy policy{true, false, true, true, 8u, 0u, 0u};
    require(core::prepare_csr_fallback_operation(problem, structures,
        projection, numeric(), policy, csr, device, feature_axis, row_axis,
        &state, &prepared), "CSR candidate preparation");
    const void *const prepared_state_address = prepared.persistent.data;
    const void *const prepared_row_offsets = state.projection.row_offsets;
    const void *const prepared_feature_ids =
        state.projection.execution_feature_ids;
    require(state.projection.row_offsets == row_offsets.data
        && state.projection.execution_feature_ids == feature_ids.data
        && state.projection.values == projection_values.data
        && prepared.binding_contract.workspace.minimum_bytes == 0u
        && prepared.binding_contract.output_effects[0].update
            == execution::output_update_kind::overwrite
        && prepared.binding_contract.output_orders[0].transition
            == execution::order_transition_kind::preserve,
        "prepared projection ownership, workspace, effect, and order contract");

    execution::relation_structure relation{};
    relation.identity = structures.structures[0].runtime;
    relation.epoch = structures.structures[0].epoch;
    relation.source_axis = feature_axis;
    relation.destination_axis = row_axis;
    relation.projections = {1u, 1u};
    relation.logical_edge_count = 3u;
    execution::value_plane plane{};
    plane.structure = relation.identity;
    plane.structure_epoch_value = relation.epoch;
    plane.values = launch_values.data;
    plane.location = device_location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {2u};
    plane.element_count = 3u;
    plane.value_bytes = 3u * sizeof(__half);
    execution::value_binding value_binding{&plane, {2u}};
    execution::biological_operand_view input{}, output_operand{};
    input.kind = execution::operand_kind::dense_tensor;
    input.storage.dense = dense(weights.data, execution::numeric_type::f32,
        feature_axis, 3u, device);
    output_operand.kind = execution::operand_kind::dense_tensor;
    output_operand.storage.dense = dense(output.data,
        execution::numeric_type::f32, row_axis, 2u, device);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(
        &stream, cudaStreamNonBlocking), "create stream");
    execution::launch_bindings launch{};
    launch.structures = &relation;
    launch.inputs = &input;
    launch.outputs = &output_operand;
    launch.values = &value_binding;
    launch.input_count = 1u;
    launch.output_count = 1u;
    launch.value_count = 1u;
    launch.structure_count = 1u;
    launch.stream = {stream, device, 0u};
    launch.workspace = {nullptr, 0u, device_location(device)};
    require(core::run_prepared_operation(prepared, launch),
        "existing CSR prepared execution");
    require_cuda(cudaStreamSynchronize(stream), "synchronize result");
    std::vector<float> result(2u);
    require_cuda(cudaMemcpy(result.data(), output.data,
        result.size() * sizeof(float), cudaMemcpyDeviceToHost), "download result");
    require(std::fabs(result[0] - 16.0f) < 1.0e-5f
        && std::fabs(result[1] - 15.0f) < 1.0e-5f,
        "launch-bound values and numerical parity");

    execution::value_plane plane_b = plane;
    plane_b.values = launch_values_b.data;
    plane_b.generation = {3u};
    execution::value_binding binding_b{&plane_b, {3u}};
    output_operand.storage.dense.data = output_b.data;
    launch.values = &binding_b;
    require(core::run_prepared_operation(prepared, launch),
        "second value generation over prepared CSR structure");
    require_cuda(cudaStreamSynchronize(stream),
        "synchronize second CSR generation");
    require_cuda(cudaMemcpy(result.data(), output_b.data,
        result.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download second CSR generation");
    require(std::fabs(result[0] - 43.0f) < 1.0e-5f
        && std::fabs(result[1] - 30.0f) < 1.0e-5f,
        "second CSR generation numerical parity");
    require(prepared.persistent.data == prepared_state_address
        && state.projection.row_offsets == prepared_row_offsets
        && state.projection.execution_feature_ids == prepared_feature_ids,
        "value generation rebuilt CSR prepared geometry");

    execution::value_plane relocated_plane = plane;
    relocated_plane.values = relocated_values.data;
    execution::value_binding relocated_binding{&relocated_plane, {2u}};
    output_operand.storage.dense.data = output.data;
    launch.values = &relocated_binding;
    require(core::run_prepared_operation(prepared, launch),
        "pointer-relocated CSR value plane");
    require_cuda(cudaStreamSynchronize(stream),
        "synchronize relocated CSR generation");
    require_cuda(cudaMemcpy(result.data(), output.data,
        result.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download relocated CSR generation");
    require(std::fabs(result[0] - 16.0f) < 1.0e-5f
        && std::fabs(result[1] - 15.0f) < 1.0e-5f,
        "CSR pointer relocation changed semantic value identity");

    core::csr_fallback_prepared_state rejected_state{};
    core::prepared_operation rejected{};
    core::projection_key wrong_projection = projection;
    wrong_projection.kind = core::projection_kind::native_row_masked;
    require(core::prepare_csr_fallback_operation(problem, structures,
        wrong_projection, numeric(), policy, csr, device, feature_axis, row_axis,
        &rejected_state, &rejected).code
            == core::operation_status_code::unsupported_projection,
        "projection rejection");
    core::numeric_policy rejected_numeric = numeric();
    rejected_numeric.sparse_storage = execution::numeric_type::f32;
    require(core::prepare_csr_fallback_operation(problem, structures,
        projection, rejected_numeric, policy, csr, device, feature_axis, row_axis,
        &rejected_state, &rejected).code
            == core::operation_status_code::unsupported_numeric_policy,
        "numeric rejection");
    core::prepare_policy capture_required = policy;
    capture_required.graph_capture_required = true;
    require(core::prepare_csr_fallback_operation(problem, structures,
        projection, numeric(), capture_required, csr, device,
        feature_axis, row_axis, &rejected_state, &rejected).code
            == core::operation_status_code::capability_rejected,
        "graph capture rejection");
    core::prepare_policy no_preprocessing = policy;
    no_preprocessing.allow_persistent_preprocessing = false;
    require(core::prepare_csr_fallback_operation(problem, structures,
        projection, numeric(), no_preprocessing, csr, device,
        feature_axis, row_axis, &rejected_state, &rejected).code
            == core::operation_status_code::capability_rejected,
        "preconstructed projection policy rejection");
    launch.values = &binding_b;
    binding_b.expected_generation.value = 4u;
    require(core::run_prepared_operation(prepared, launch).binding
            == execution::binding_validation_code::stale_value,
        "stale value generation rejection");
    binding_b.expected_generation = plane_b.generation;
    relation.epoch.value += 1u;
    require(core::run_prepared_operation(prepared, launch).code
            == core::operation_status_code::stale_structure,
        "structure epoch invalidates prepared CSR operation");
    relation.epoch = structures.structures[0].epoch;
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}
