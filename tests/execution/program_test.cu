#include <Cellerator/execution/program.hh>

#include <Cellerator/compute/candidate/csr_fallback_candidate.hh>
#include <Cellerator/compute/candidate/row_masked_n1_candidate.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace core = cellerator::compute::math::core;
namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;
namespace runtime = cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "program_test: " << message << '\n';
        std::abort();
    }
}

void require(execution::executable_program_status status,
    const char *message) {
    if (!status) {
        std::cerr << "program_test: " << message
                  << " (code=" << static_cast<unsigned>(status.code)
                  << ", detail=" << status.message
                  << ", operation=" << status.operation.message << ")\n";
        std::abort();
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::cerr << "program_test: " << message << ": "
                  << cudaGetErrorString(status) << '\n';
        std::abort();
    }
}

template<typename T>
struct device_buffer {
    T *data = nullptr;
    explicit device_buffer(std::size_t count) {
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
            count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (data != nullptr) (void) cudaFree(data); }
};

template<typename T, std::size_t N>
void upload(device_buffer<T> &device, const T (&host)[N]) {
    require_cuda(cudaMemcpy(device.data, host, sizeof(host),
        cudaMemcpyHostToDevice), "upload");
}

execution::program_axis axis(
    std::uint32_t live, std::uint64_t domain, std::uint64_t order) {
    return {{{live, 1u}, {live + 1u, 1u},
                {live + 2u, 1u}, {live + 3u, 1u}},
        {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {domain, 1u}, {order, 1u}, {0x700u, 1u}, {0x800u, 1u}}};
}

execution::device_location location(int device) {
    return {execution::residency_kind::device, {}, device, 0u};
}

execution::dense_tensor_view dense(
    void *data, execution::axis_identity axis_value,
    std::uint64_t size, int device) {
    execution::dense_tensor_view view{};
    view.data = data;
    view.location = location(device);
    view.value_type = execution::numeric_type::f32;
    view.rank = 1u;
    view.axes[0] = axis_value;
    view.shape[0] = size;
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
    return value;
}

struct csr_fixture {
    device_buffer<std::uint32_t> rows{3u};
    device_buffer<std::uint32_t> features{3u};
    device_buffer<__half> values{3u};
    cm::execution_csr_view view{};

    csr_fixture() {
        const std::uint32_t host_rows[]{0u, 2u, 3u};
        const std::uint32_t host_features[]{0u, 2u, 1u};
        const __half host_values[]{
            __float2half(1.0f), __float2half(2.0f), __float2half(3.0f)};
        upload(rows, host_rows);
        upload(features, host_features);
        upload(values, host_values);
        view.row_count = 2u;
        view.full_row_count = 2u;
        view.feature_count = 3u;
        view.nnz_count = 3u;
        view.value_size_bytes = sizeof(__half);
        view.row_domain_identity = 0x3003u;
        view.structure.identity_version =
            cm::execution_csr_structure_identity_version;
        view.structure.value = 0x7073u;
        view.feature_order.kind = cm::feature_order_kind::packed;
        view.feature_order.feature_count = 3u;
        view.feature_order.feature_axis_identity_version = 1u;
        view.feature_order.feature_axis_identity = 0x5005u;
        view.feature_order.packing_geometry_identity = 0x1001u;
        view.row_offsets = rows.data;
        view.execution_feature_ids = features.data;
        view.values = values.data;
    }
};

bool measure(void *, const planner::planner_candidate &candidate,
    planner::measured_candidate *result) noexcept {
    result->correct = true;
    result->sample_count = 5u;
    result->spread_percent = 0.5;
    result->phases = candidate.analytical;
    result->phases.kernel_ns = core::same_stable_id(
            candidate.identity, core::csr_fallback_candidate_id)
        ? 10.0 : 100.0;
    return true;
}

execution::executable_program_request make_request(
    runtime::execution_session *session,
    const execution::activated_projection_reference *projections,
    const execution::program_candidate_cost *costs,
    void *state, std::size_t state_bytes) {
    execution::executable_program_request request{};
    request.problem = {core::operation_core_schema_version,
        core::operation_kind::weighted_relation_reduce, 0u,
        {73u, 1u}, 1u, 1u, 3u};
    request.structures.count = 1u;
    request.structures.structures[0] = {{11u, 12u}, {21u, 1u}, {7u}};
    request.numeric = numeric();
    request.preparation = {true, false, true, true, 8u, 0u, 0u};
    request.source_axis = axis(10u, 0x100u, 0x200u);
    request.destination_axis = axis(20u, 0x300u, 0x400u);
    request.dense_column_axis = axis(30u, 0x500u, 0x600u);
    request.planning.problem.identity = request.problem.operation;
    require(planner::make_persistent_structure_set_key(
        request.structures, &request.planning.structures),
        "persistent structure key");
    request.planning.geometry = {
        request.source_axis.persistent.domain,
        request.destination_axis.persistent.domain,
        request.source_axis.persistent.geometry,
        request.source_axis.persistent.order,
        request.destination_axis.persistent.order,
        request.source_axis.persistent.partition};
    request.planning.device = {1u, 7u, 0u, 700u};
    request.planning.build = {10u, 20u, 30u, 40u};
    request.planning.policy = {1u, 1u, 1u, 1u, 1u, 1u, 0u};
    request.planner_policy.shortlist_size = 2u;
    request.planner_policy.maximum_measurements = 2u;
    request.planner_policy.minimum_tuning_work_items = 1u;
    request.planner_policy.maximum_spread_percent = 10.0;
    request.planner_policy.tune_one_shot = true;
    request.measurement = {nullptr, measure};
    request.current_evidence_revision = 1u;
    request.catalog = core::built_in_candidate_catalog();
    request.projections = projections;
    request.projection_count = 2u;
    request.costs = costs;
    request.cost_count = 2u;
    request.session = session;
    request.dense_width = 1u;
    request.preparation_state = {state, state_bytes};
    return request;
}

void check_output(const device_buffer<float> &output,
    float first, float second) {
    float host[2]{};
    require_cuda(cudaMemcpy(host, output.data, sizeof(host),
        cudaMemcpyDeviceToHost), "download output");
    require(std::fabs(host[0] - first) < 1.0e-5f
        && std::fabs(host[1] - second) < 1.0e-5f,
        "independent numerical referee");
}

} // namespace

int main() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    runtime::execution_session session{};
    runtime::execution_session_options options{};
    options.device = device;
    require(runtime::init_session(&session, options)
            == runtime::session_status::success,
        "initialize sole execution session");

    csr_fixture fixture;
    cellpack::persistent_packing_payload_view activated_row{};
    const core::projection_key row_key{{31u, 32u}, {41u, 1u},
        core::projection_kind::native_row_masked,
        cellpack::persistent_packing_payload_schema_version, 1u};
    const core::projection_key csr_key{{33u, 34u}, {42u, 1u},
        core::projection_kind::csr, cm::execution_csr_schema_version, 1u};
    const execution::activated_projection_reference projections[]{
        execution::program_projection(row_key, activated_row),
        execution::program_projection(csr_key, fixture.view)};
    execution::program_candidate_cost costs[2]{};
    costs[0] = {core::row_masked_n1_candidate_id, row_key.persistent,
        {}, planner::planner_candidate_correct
            | planner::planner_candidate_deterministic
            | planner::planner_candidate_graph_capture, 0u};
    costs[0].phases.semantic_packing_ns = 800.0;
    costs[0].phases.kernel_ns = 10.0;
    costs[1] = {core::csr_fallback_candidate_id, csr_key.persistent,
        {}, planner::planner_candidate_correct
            | planner::planner_candidate_deterministic
            | planner::planner_candidate_conventional, 0u};
    costs[1].phases.projection_construction_ns = 100.0;
    costs[1].phases.kernel_ns = 50.0;
    alignas(64) unsigned char state[2048]{};
    auto request = make_request(
        &session, projections, costs, state, sizeof(state));
    execution::executable_program program{};
    require(execution::compile_executable_program(request, &program),
        "compile planner-backed program");
    require(program.candidate_count == 2u && program.legal_count == 2u
        && program.selection == planner::selection_source::empirical
        && program.conventional_winner
        && core::same_stable_id(program.selected_candidate,
            core::csr_fallback_candidate_id)
        && program.preparation_count == 1u && session.plans.size == 1u,
        "enumeration, planner selection, or preparation metadata");

    planner::total_cost row_once{}, row_reused{}, csr_once{}, csr_reused{};
    require(static_cast<bool>(planner::compute_total_cost(
        costs[0].phases, 1u, 1u, 1u, &row_once)), "row cost");
    require(static_cast<bool>(planner::compute_total_cost(
        costs[0].phases, 100u, 100u, 1u, &row_reused)), "reused row cost");
    require(static_cast<bool>(planner::compute_total_cost(
        costs[1].phases, 1u, 1u, 1u, &csr_once)), "CSR cost");
    require(static_cast<bool>(planner::compute_total_cost(
        costs[1].phases, 100u, 100u, 1u, &csr_reused)), "reused CSR cost");
    require(csr_once.amortized_total_ns < row_once.amortized_total_ns
        && row_reused.amortized_total_ns < csr_reused.amortized_total_ns,
        "reuse horizon did not change complete-cost ordering");

    auto invalid = request;
    invalid.source_axis = request.destination_axis;
    require(execution::compile_executable_program(invalid, &program).code
            == execution::executable_program_status_code::identity_mismatch,
        "swapped logical axes were accepted");
    invalid = request;
    invalid.numeric.dense_storage = execution::numeric_type::f16;
    require(execution::compile_executable_program(invalid, &program).code
            == execution::executable_program_status_code::no_compatible_candidate,
        "unsupported numeric tuple was accepted");
    invalid = request;
    invalid.dense_width = 17u;
    require(execution::compile_executable_program(invalid, &program).code
            == execution::executable_program_status_code::no_compatible_candidate,
        "unsupported width was accepted");
    execution::activated_projection_reference incompatible[]{
        projections[0], projections[1]};
    incompatible[0].key.kind = core::projection_kind::csr;
    incompatible[1].key.kind = core::projection_kind::native_feature_major;
    invalid = request;
    invalid.projections = incompatible;
    require(execution::compile_executable_program(invalid, &program).code
            == execution::executable_program_status_code::no_compatible_candidate,
        "incompatible projection was accepted");

    // Negative compile attempts reset the output object.
    require(execution::compile_executable_program(request, &program),
        "recompile program after negative cases");
    device_buffer<float> input_a{3u}, input_b{3u}, output_a{2u}, output_b{2u};
    device_buffer<__half> values_b{3u};
    const float host_a[]{2.0f, 5.0f, 7.0f};
    const float host_b[]{1.0f, 2.0f, 3.0f};
    const __half host_values_b[]{
        __float2half(2.0f), __float2half(4.0f), __float2half(6.0f)};
    upload(input_a, host_a);
    upload(input_b, host_b);
    upload(values_b, host_values_b);
    cudaStream_t producer = nullptr, consumer = nullptr;
    require_cuda(cudaStreamCreateWithFlags(
        &producer, cudaStreamNonBlocking), "create producer stream");
    require_cuda(cudaStreamCreateWithFlags(
        &consumer, cudaStreamNonBlocking), "create consumer stream");
    runtime::value_readiness_record readiness;
    require(runtime::initialize_value_readiness(&readiness, device)
            == runtime::value_readiness_status::success,
        "initialize readiness");

    execution::relation_structure relation{{21u, 1u}, {7u},
        request.source_axis.live, request.destination_axis.live,
        {1u, 1u}, 3u};
    execution::value_plane plane{};
    plane.structure = relation.identity;
    plane.structure_epoch_value = relation.epoch;
    plane.values = fixture.values.data;
    plane.location = location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = 3u;
    plane.value_bytes = 3u * sizeof(__half);
    execution::value_binding value{&plane, {1u}};
    execution::biological_operand_view input{}, output{};
    input.kind = output.kind = execution::operand_kind::dense_tensor;
    input.storage.dense = dense(
        input_a.data, request.source_axis.live, 3u, device);
    output.storage.dense = dense(
        output_a.data, request.destination_axis.live, 2u, device);
    execution::launch_bindings bindings{};
    bindings.structures = &relation;
    bindings.inputs = &input;
    bindings.outputs = &output;
    bindings.values = &value;
    bindings.input_count = bindings.output_count =
        bindings.value_count = bindings.structure_count = 1u;
    bindings.stream = {producer, device, 0u};
    bindings.workspace = {nullptr, 0u, location(device)};
    require(runtime::publish_value_generation(
            &readiness, 7u, 1u, producer, cudaSuccess)
            == runtime::value_readiness_status::success,
        "publish first generation");
    execution::executable_program_launch launch{
        bindings, &readiness, {7u}, {1u}};
    execution::executable_program_result result{};
    require(execution::run_executable_program(&program, launch, &result),
        "same-stream first generation");
    require_cuda(cudaStreamSynchronize(producer), "wait first result");
    check_output(output_a, 16.0f, 15.0f);
    require(result.enqueued && result.output_order_count == 1u
        && result.completion_stream.stream == producer,
        "observable result metadata");

    plane.values = values_b.data;
    plane.generation = value.expected_generation = {2u};
    input.storage.dense = dense(
        input_b.data, request.source_axis.live, 3u, device);
    output.storage.dense = dense(
        output_b.data, request.destination_axis.live, 2u, device);
    bindings.stream.stream = consumer;
    launch = {bindings, &readiness, {7u}, {2u}};
    require(runtime::publish_value_generation(
            &readiness, 7u, 2u, producer, cudaSuccess)
            == runtime::value_readiness_status::success,
        "publish second generation");
    require(execution::run_executable_program(&program, launch, &result),
        "cross-stream second generation");
    require_cuda(cudaStreamSynchronize(consumer), "wait second result");
    check_output(output_b, 14.0f, 12.0f);
    require(program.preparation_count == 1u && program.run_count == 2u,
        "pointer or generation change rebuilt structure");

    auto bad = launch;
    bad.expected_structure_epoch = {8u};
    require(execution::run_executable_program(&program, bad, &result).code
            == execution::executable_program_status_code::stale_structure,
        "stale structure epoch was accepted");
    bad = launch;
    bad.expected_value_generation = {3u};
    require(execution::run_executable_program(&program, bad, &result).code
            == execution::executable_program_status_code::stale_or_unready_value,
        "wrong value generation was accepted");
    plane.generation = value.expected_generation = {3u};
    bad = {bindings, &readiness, {7u}, {3u}};
    require(execution::run_executable_program(&program, bad, &result).code
            == execution::executable_program_status_code::stale_or_unready_value,
        "unready generation was accepted");
    plane.generation = value.expected_generation = {2u};
    execution::relation_structure wrong_relation = relation;
    wrong_relation.identity = {99u, 1u};
    bad = launch;
    bad.bindings.structures = &wrong_relation;
    require(!execution::run_executable_program(&program, bad, &result),
        "wrong structure identity was accepted");
    const auto workspace = program.prepared.binding_contract.workspace;
    program.prepared.binding_contract.workspace = {64u, 64u, 0u};
    require(!execution::run_executable_program(&program, launch, &result),
        "insufficient workspace was accepted");
    program.prepared.binding_contract.workspace = workspace;
    auto *order = const_cast<execution::output_axis_contract *>(
        program.prepared.binding_contract.output_orders);
    const auto output_axis = order[0].output_axis;
    order[0].output_axis = request.source_axis.live;
    require(!execution::run_executable_program(&program, launch, &result),
        "invalid output order was accepted");
    order[0].output_axis = output_axis;
    auto *effect = const_cast<execution::output_effect_contract *>(
        program.prepared.binding_contract.output_effects);
    const auto saved_effect = effect[0];
    effect[0].requires_initialized_destination = true;
    require(!execution::run_executable_program(&program, launch, &result),
        "invalid output effect was accepted");
    effect[0] = saved_effect;

    require(runtime::clear_value_readiness(&readiness)
            == runtime::value_readiness_status::success,
        "clear readiness");
    require_cuda(cudaStreamDestroy(consumer), "destroy consumer stream");
    require_cuda(cudaStreamDestroy(producer), "destroy producer stream");
    runtime::clear_session(&session);
    std::cout << "program_test passed candidates=" << program.candidate_count
              << " runs=" << program.run_count << '\n';
    return 0;
}
