#include <Cellerator/execution/program.hh>
#include <Cellerator/runtime/device_descriptor.hh>
#include <Cellerator/runtime/session.cuh>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;
namespace runtime = cellerator::runtime;

namespace {

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "hot path validation failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

constexpr core::stable_id candidate_id{0x10601u, 0x10602u};
constexpr core::stable_id provider_id{0x10603u, 0x10604u};
constexpr core::stable_id projection_view_id{0x10605u, 0x10606u};
constexpr std::uint32_t element_count = 32u;

std::uint64_t numeric_query_count = 0u;
std::uint64_t preparation_count = 0u;
std::uint64_t projection_parse_count = 0u;
std::uint64_t dispatch_count = 0u;

struct fake_projection_view {
    float scale = 1.0f;
    std::uint32_t schema_version = 1u;
};

struct prepared_state {
    execution::operand_axis_contract input_contract{};
    execution::operand_axis_contract output_contract{};
    execution::output_axis_contract output_order{};
    execution::output_effect_contract output_effect{};
    float scale = 1.0f;
};

__global__ void hot_path_kernel(const float *input, float *output,
    float scale, std::uint64_t generation, std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = input[index] * scale
            + static_cast<float>(generation);
}

bool supports_numeric(const core::numeric_policy &numeric) noexcept {
    ++numeric_query_count;
    return numeric.sparse_storage == execution::numeric_type::f32
        && numeric.dense_storage == execution::numeric_type::f32
        && numeric.output_storage == execution::numeric_type::f32
        && numeric.multiply == execution::numeric_type::f32
        && numeric.accumulation == execution::numeric_type::f32;
}

core::operation_status unused_legacy_prepare(const core::operation_candidate &,
    const core::operation_problem &, const core::structure_set_key &,
    const core::projection_key &, const core::numeric_policy &,
    const core::prepare_policy &, core::prepared_operation *) noexcept {
    return {core::operation_status_code::preparation_failed,
        execution::binding_validation_code::ok,
        "hot path test called legacy preparation"};
}

core::operation_status run_hot_path(const core::prepared_operation &prepared,
    const execution::launch_bindings &bindings) noexcept {
    ++dispatch_count;
    const auto *state =
        static_cast<const prepared_state *>(prepared.persistent.data);
    const auto *input = static_cast<const float *>(
        bindings.inputs[0].storage.dense.data);
    auto *output =
        static_cast<float *>(bindings.outputs[0].storage.dense.data);
    const std::uint64_t generation =
        bindings.values[0].expected_generation.value;
    constexpr std::uint32_t threads = 128u;
    hot_path_kernel<<<(element_count + threads - 1u) / threads, threads, 0u,
        static_cast<cudaStream_t>(bindings.stream.stream)>>>(
        input, output, state->scale, generation, element_count);
    return cudaGetLastError() == cudaSuccess
        ? core::operation_status{}
        : core::operation_status{core::operation_status_code::execution_failed,
            execution::binding_validation_code::ok,
            "hot path kernel launch failed"};
}

core::operation_status prepare_hot_path(
    const core::candidate_preparation_adapter_v2 &adapter,
    const core::candidate_preparation_request_v2 &request,
    const execution::activated_projection_reference_v2 &projection,
    core::prepared_operation *prepared) noexcept {
    ++preparation_count;
    ++projection_parse_count;
    if (adapter.candidate == nullptr
        || request.state.data == nullptr
        || request.state.bytes < sizeof(prepared_state)
        || projection.view_bytes != sizeof(fake_projection_view))
        return {core::operation_status_code::preparation_failed,
            execution::binding_validation_code::ok,
            "hot path preparation inputs are incomplete"};
    const auto &view =
        *static_cast<const fake_projection_view *>(projection.view);
    if (view.schema_version != 1u || !std::isfinite(view.scale))
        return {core::operation_status_code::unsupported_projection,
            execution::binding_validation_code::ok,
            "hot path projection is invalid"};

    auto *state = static_cast<prepared_state *>(request.state.data);
    *state = prepared_state{};
    state->scale = view.scale;
    state->input_contract.kind = execution::operand_kind::dense_tensor;
    state->input_contract.rank = 1u;
    state->input_contract.axes[0] = request.feature_axis;
    state->output_contract.kind = execution::operand_kind::dense_tensor;
    state->output_contract.rank = 1u;
    state->output_contract.axes[0] = request.row_axis;
    state->output_order.input_axis = request.row_axis;
    state->output_order.output_axis = request.row_axis;
    state->output_order.transition = execution::order_transition_kind::preserve;
    state->output_order.axis_index = 0u;
    state->output_order.operand_index = 0u;
    state->output_order.may_fuse = 1u;
    state->output_order.may_remain_packed = 1u;
    state->output_effect.update = execution::output_update_kind::overwrite;
    state->output_effect.input_scale_binding_id =
        execution::invalid_scalar_binding_id;
    state->output_effect.destination_scale_binding_id =
        execution::invalid_scalar_binding_id;

    prepared->problem = request.problem;
    prepared->structures = request.structures;
    prepared->projection = projection.key;
    prepared->numeric = request.numeric;
    prepared->kernel = adapter.candidate->candidate.identity;
    prepared->backend = core::backend_kind::native_direct;
    prepared->capability_flags = core::candidate_deterministic
        | core::candidate_graph_capture;
    prepared->persistent = {state, sizeof(*state)};
    prepared->binding_contract.structures[0] = {
        request.structures.structures[0].runtime,
        request.structures.structures[0].epoch};
    prepared->binding_contract.inputs = &state->input_contract;
    prepared->binding_contract.outputs = &state->output_contract;
    prepared->binding_contract.output_orders = &state->output_order;
    prepared->binding_contract.output_effects = &state->output_effect;
    prepared->binding_contract.input_count = 1u;
    prepared->binding_contract.output_count = 1u;
    prepared->binding_contract.output_order_count = 1u;
    prepared->binding_contract.structure_count = 1u;
    prepared->binding_contract.output_effect_count = 1u;
    prepared->binding_contract.workspace = {0u, 1u, 0u};
    prepared->run = run_hot_path;
    return {};
}

core::candidate_descriptor_v2 make_descriptor() {
    core::candidate_descriptor_v2 result{};
    result.candidate.identity = candidate_id;
    result.candidate.name = "ce-geo-hot-path-test-provider";
    result.candidate.operation = core::operation_kind::weighted_relation_reduce;
    result.candidate.projection = core::projection_kind::architecture_specific;
    result.candidate.backend = core::backend_kind::native_direct;
    result.candidate.capability_flags = core::candidate_deterministic
        | core::candidate_graph_capture;
    result.candidate.supports_numeric = supports_numeric;
    result.candidate.prepare = unused_legacy_prepare;
    result.provider_identity = provider_id;
    result.projection_contract = {projection_view_id, 1u, 0u, 1u, 1u};
    result.minimum_dense_width = 1u;
    result.maximum_dense_width = 1u;
    result.state_bytes = sizeof(prepared_state);
    result.state_alignment = alignof(prepared_state);
    return result;
}

execution::program_axis make_axis(std::uint32_t live,
    std::uint64_t domain, std::uint64_t order) {
    return {{{live, 1u}, {live + 1u, 1u}, {live + 2u, 1u},
                {live + 3u, 1u}},
        {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {domain, 1u}, {order, 1u}, {0x700u, 1u}, {0x800u, 1u}}};
}

core::numeric_policy make_numeric() {
    core::numeric_policy result{};
    result.sparse_storage = execution::numeric_type::f32;
    result.dense_storage = execution::numeric_type::f32;
    result.output_storage = execution::numeric_type::f32;
    result.multiply = execution::numeric_type::f32;
    result.accumulation = execution::numeric_type::f32;
    result.scalar = execution::numeric_type::u32;
    return result;
}

execution::device_location device_location(int device) {
    return {execution::residency_kind::device, {}, device, 0u};
}

execution::dense_tensor_view dense_view(float *data,
    const execution::axis_identity &axis, int device) {
    execution::dense_tensor_view result{};
    result.data = data;
    result.location = device_location(device);
    result.value_type = execution::numeric_type::f32;
    result.rank = 1u;
    result.axes[0] = axis;
    result.shape[0] = element_count;
    result.stride[0] = 1;
    return result;
}

execution::executable_program_request_v2 make_request(
    runtime::execution_session *session,
    const core::candidate_preparation_adapter_v2 *adapter,
    const execution::activated_projection_reference_v2 *projection,
    const execution::program_candidate_cost *cost, prepared_state *state,
    std::uint32_t projection_slot) {
    execution::executable_program_request_v2 result{};
    result.problem.kind = core::operation_kind::weighted_relation_reduce;
    result.problem.operation = {0x10610u, 0x10611u};
    result.problem.input_count = 1u;
    result.problem.output_count = 1u;
    result.problem.logical_work_items = element_count;
    result.structures.count = 1u;
    result.structures.structures[0] = {
        {0x10620u, 0x10621u}, {51u, 1u}, {7u}};
    result.numeric = make_numeric();
    result.preparation = {true, true, true, true, 16u, 0u, 0u};
    result.source_axis = make_axis(10u, 0x100u, 0x200u);
    result.destination_axis = make_axis(20u, 0x300u, 0x400u);
    result.dense_column_axis = make_axis(30u, 0x500u, 0x600u);
    result.planning.problem.identity = result.problem.operation;
    require(planner::make_persistent_structure_set_key(
                result.structures, &result.planning.structures),
        "persistent structure key construction");
    result.planning.geometry = {result.source_axis.persistent.domain,
        result.destination_axis.persistent.domain,
        result.source_axis.persistent.geometry,
        result.source_axis.persistent.order,
        result.destination_axis.persistent.order,
        result.source_axis.persistent.partition};
    result.planning.device = {1u, 7u, 0u, 700u};
    result.planning.build = {1u, 2u, 3u, 4u};
    result.planning.policy = {16u, 16u, 16u, 1u, 1u, 1u, 0u};
    result.planner_policy.deterministic = true;
    result.planner_policy.graph_capture_required = true;
    result.current_evidence_revision = 1u;
    result.catalog = {adapter, 1u, 0u};
    result.projections = projection;
    result.projection_count = 1u;
    result.costs = cost;
    result.cost_count = 1u;
    result.session = session;
    result.dense_width = 1u;
    result.preparation_state = {state, sizeof(*state)};
    (void) projection_slot;
    return result;
}

struct launch_fixture {
    execution::relation_structure relation{};
    execution::value_plane plane{};
    execution::value_binding value{};
    execution::biological_operand_view input{};
    execution::biological_operand_view output{};
    execution::launch_bindings bindings{};
    execution::executable_program_launch launch{};
};

launch_fixture make_launch(const execution::executable_program_request_v2 &request,
    float *input, float *output, float *value_storage, cudaStream_t stream,
    int device, runtime::value_readiness_record *readiness,
    std::uint64_t generation) {
    launch_fixture result{};
    result.relation = {{51u, 1u}, {7u}, request.source_axis.live,
        request.destination_axis.live, {61u, 1u}, 1u};
    result.plane.structure = result.relation.identity;
    result.plane.structure_epoch_value = result.relation.epoch;
    result.plane.values = value_storage;
    result.plane.location = device_location(device);
    result.plane.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    result.plane.quantization.kind = execution::quantization_kind::none;
    result.plane.layout = execution::value_layout_kind::logical_edge_order;
    result.plane.generation = {generation};
    result.plane.element_count = 1u;
    result.plane.value_bytes = sizeof(float);
    result.value = {&result.plane, {generation}};
    result.input.kind = execution::operand_kind::dense_tensor;
    result.output.kind = execution::operand_kind::dense_tensor;
    result.input.storage.dense = dense_view(
        input, request.source_axis.live, device);
    result.output.storage.dense = dense_view(
        output, request.destination_axis.live, device);
    result.bindings.structures = &result.relation;
    result.bindings.inputs = &result.input;
    result.bindings.outputs = &result.output;
    result.bindings.values = &result.value;
    result.bindings.input_count = 1u;
    result.bindings.output_count = 1u;
    result.bindings.value_count = 1u;
    result.bindings.structure_count = 1u;
    result.bindings.stream = {stream, device, 0u};
    result.bindings.workspace = {nullptr, 0u, device_location(device)};
    result.launch = {result.bindings, readiness, {7u}, {generation}};
    return result;
}

void set_generation(launch_fixture *fixture, std::uint64_t generation) {
    fixture->plane.generation.value = generation;
    fixture->value.expected_generation.value = generation;
    fixture->launch.bindings = fixture->bindings;
    fixture->launch.expected_value_generation.value = generation;
}

void upload_input(float *device, float offset, cudaStream_t stream) {
    std::array<float, element_count> host{};
    for (std::uint32_t index = 0u; index < element_count; ++index)
        host[index] = offset + static_cast<float>(index) * 0.25f;
    require_cuda(cudaMemcpyAsync(device, host.data(), sizeof(host),
                     cudaMemcpyHostToDevice, stream),
        "upload graph-stable input");
}

void check_output(const float *device, float input_offset, float scale,
    std::uint64_t generation, cudaStream_t stream) {
    std::array<float, element_count> host{};
    require_cuda(cudaMemcpyAsync(host.data(), device, sizeof(host),
                     cudaMemcpyDeviceToHost, stream),
        "download hot-path output");
    require_cuda(cudaStreamSynchronize(stream), "synchronize output check");
    for (std::uint32_t index = 0u; index < element_count; ++index) {
        const float expected =
            (input_offset + static_cast<float>(index) * 0.25f) * scale
            + static_cast<float>(generation);
        require(std::abs(host[index] - expected) < 1.0e-6f,
            "hot-path numerical result mismatch");
    }
}

bool same_accounting(const runtime::session_accounting &left,
    const runtime::session_accounting &right) {
    return left.structure.current_bytes == right.structure.current_bytes
        && left.structure.high_water_bytes == right.structure.high_water_bytes
        && left.structure.allocation_count == right.structure.allocation_count
        && left.plan.current_bytes == right.plan.current_bytes
        && left.plan.high_water_bytes == right.plan.high_water_bytes
        && left.plan.allocation_count == right.plan.allocation_count
        && left.graph_stable.current_bytes
            == right.graph_stable.current_bytes
        && left.graph_stable.high_water_bytes
            == right.graph_stable.high_water_bytes
        && left.graph_stable.allocation_count
            == right.graph_stable.allocation_count
        && left.transient.current_bytes == right.transient.current_bytes
        && left.transient.high_water_bytes == right.transient.high_water_bytes
        && left.transient.allocation_count == right.transient.allocation_count
        && left.device_query_count == right.device_query_count
        && left.handle_prepare_count == right.handle_prepare_count
        && left.launch_bind_count == right.launch_bind_count
        && left.synchronization_count == right.synchronization_count;
}

} // namespace

int main() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    std::array<cudaStream_t, 2u> streams{};
    for (auto &stream : streams)
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "create external stream");

    runtime::execution_session session{};
    runtime::execution_session_options options{};
    options.device = device;
    options.external_streams = streams.data();
    options.external_stream_count = streams.size();
    options.owned_stream_count = 0u;
    require(runtime::init_session(&session, options)
            == runtime::session_status::success,
        "initialize execution session over external streams");
    for (std::uint32_t index = 0u; index < streams.size(); ++index)
        require(runtime::prepare_stream_libraries(&session, index)
                == runtime::session_status::success,
            "prepare stream libraries before seal");

    std::array<float *, 5u> buffers{};
    for (auto &buffer : buffers) {
        void *allocation = nullptr;
        require(runtime::reserve_persistent(&session,
                    runtime::persistent_lifetime::graph_stable,
                    element_count * sizeof(float), &allocation)
                == runtime::session_status::success,
            "reserve graph-stable hot-path buffer");
        buffer = static_cast<float *>(allocation);
    }

    const core::candidate_descriptor_v2 descriptor = make_descriptor();
    const core::candidate_preparation_adapter_v2 adapter{
        core::candidate_preparation_adapter_schema_version_v2,
        sizeof(core::candidate_preparation_adapter_v2), &descriptor,
        prepare_hot_path, {}};
    const std::array<fake_projection_view, 2u> views{{
        {2.0f, 1u}, {-0.5f, 1u}}};
    std::array<execution::activated_projection_reference_v2, 2u> projections{};
    std::array<execution::program_candidate_cost, 2u> costs{};
    for (std::uint32_t index = 0u; index < projections.size(); ++index) {
        projections[index].key = {{0x10630u + index, 0x10640u + index},
            {71u + index, 1u}, core::projection_kind::architecture_specific,
            1u, 1u};
        projections[index].provider_identity = provider_id;
        projections[index].contract = descriptor.projection_contract;
        projections[index].location = device_location(device);
        projections[index].view = &views[index];
        projections[index].view_bytes = sizeof(views[index]);
        costs[index].candidate = candidate_id;
        costs[index].projection = projections[index].key.persistent;
        costs[index].phases.kernel_ns = 10.0 + index;
        costs[index].planner_flags = planner::planner_candidate_correct
            | planner::planner_candidate_deterministic
            | planner::planner_candidate_graph_capture;
    }
    std::array<prepared_state, 2u> states{};
    std::array<execution::executable_program, 2u> programs{};
    std::array<execution::executable_program_request_v2, 2u> requests{};
    for (std::uint32_t index = 0u; index < programs.size(); ++index) {
        requests[index] = make_request(&session, &adapter, &projections[index],
            &costs[index], &states[index], index);
        require(static_cast<bool>(execution::compile_executable_program_v2(
                    requests[index], &programs[index])),
            "compile independent hot-path plan");
    }
    require(preparation_count == 2u && projection_parse_count == 2u
            && session.plans.size == 2u
            && programs[0].preparation_count == 1u
            && programs[1].preparation_count == 1u,
        "cold preparation did not produce exactly two independent plans");

    runtime::value_readiness_record readiness{};
    require(runtime::initialize_value_readiness(&readiness, device)
            == runtime::value_readiness_status::success,
        "initialize value readiness before seal");
    require(runtime::seal_session(&session) == runtime::session_status::success,
        "seal prepared execution session");
    for (float *buffer : buffers)
        require(runtime::graph_stable_address(
                    session, buffer, element_count * sizeof(float)),
            "session rejected prepared graph-stable address");

    void *late_allocation = nullptr;
    require(runtime::reserve_persistent(&session,
                runtime::persistent_lifetime::plan, sizeof(float),
                &late_allocation)
            == runtime::session_status::invalid_state
            && late_allocation == nullptr,
        "sealed session admitted a persistent allocation");
    require(runtime::reserve_transient(
                &session, 0u, sizeof(float), &late_allocation)
            == runtime::session_status::invalid_state
            && late_allocation == nullptr,
        "sealed session admitted a transient allocation");
    require(runtime::prepare_stream_libraries(&session, 0u)
            == runtime::session_status::invalid_state,
        "sealed session rebuilt stream descriptors");
    std::uint64_t sealed_query_count = 0u;
    runtime::device_descriptor_v1 queried{};
    require(runtime::query_device_descriptor_v1(
                device, true, &queried, &sealed_query_count)
            == runtime::device_descriptor_status_v1::invalid_state
            && sealed_query_count == 0u,
        "sealed session reached device discovery");
    execution::executable_program rejected{};
    require(execution::compile_executable_program_v2(requests[0], &rejected).code
            == execution::executable_program_status_code::invalid_argument,
        "sealed session admitted catalog search or preparation");

    const std::uint64_t cold_numeric_queries = numeric_query_count;
    const std::uint64_t cold_preparations = preparation_count;
    const std::uint64_t cold_projection_parses = projection_parse_count;
    const runtime::session_accounting sealed_accounting = session.accounting;

    upload_input(buffers[0], 1.0f, streams[0]);
    upload_input(buffers[2], -2.0f, streams[0]);
    const float value = 3.0f;
    require_cuda(cudaMemcpyAsync(buffers[4], &value, sizeof(value),
                     cudaMemcpyHostToDevice, streams[0]),
        "upload value plane");
    require(runtime::publish_value_generation(
                &readiness, 7u, 1u, streams[0], cudaSuccess)
            == runtime::value_readiness_status::success,
        "publish generation one");
    auto launch_a = make_launch(requests[0], buffers[0], buffers[1], buffers[4],
        streams[0], device, &readiness, 1u);
    auto launch_b = make_launch(requests[1], buffers[2], buffers[3], buffers[4],
        streams[1], device, &readiness, 1u);
    execution::executable_program_result result_a{}, result_b{};
    require(static_cast<bool>(execution::run_executable_program(
                &programs[0], launch_a.launch, &result_a)),
        "run first plan on producer stream");
    require(static_cast<bool>(execution::run_executable_program(
                &programs[1], launch_b.launch, &result_b)),
        "run concurrent plan on consumer stream");
    check_output(buffers[1], 1.0f, views[0].scale, 1u, streams[0]);
    check_output(buffers[3], -2.0f, views[1].scale, 1u, streams[1]);
    require(result_a.completion_stream.stream == streams[0]
            && result_b.completion_stream.stream == streams[1]
            && result_a.consumed_generation.value == 1u
            && result_b.consumed_generation.value == 1u,
        "hot path lost external stream or generation identity");

    upload_input(buffers[0], 4.0f, streams[0]);
    require(runtime::publish_value_generation(
                &readiness, 7u, 2u, streams[0], cudaSuccess)
            == runtime::value_readiness_status::success,
        "publish generation two");
    set_generation(&launch_a, 2u);
    require(static_cast<bool>(execution::run_executable_program(
                &programs[0], launch_a.launch, &result_a)),
        "reuse first plan for new value generation");
    check_output(buffers[1], 4.0f, views[0].scale, 2u, streams[0]);
    require(programs[0].preparation_count == 1u
            && programs[1].preparation_count == 1u,
        "generation change rebuilt a prepared plan");

    upload_input(buffers[0], -1.5f, streams[0]);
    require(runtime::publish_value_generation(
                &readiness, 7u, 3u, streams[0], cudaSuccess)
            == runtime::value_readiness_status::success,
        "publish graph capture generation");
    set_generation(&launch_a, 3u);
    require_cuda(cudaStreamBeginCapture(
                     streams[0], cudaStreamCaptureModeThreadLocal),
        "begin hot-path graph capture");
    require(static_cast<bool>(execution::run_executable_program(
                &programs[0], launch_a.launch, &result_a)),
        "capture sealed prepared launch");
    cudaGraph_t graph = nullptr;
    require_cuda(cudaStreamEndCapture(streams[0], &graph),
        "end hot-path graph capture");
    std::size_t node_count = 0u;
    require_cuda(cudaGraphGetNodes(graph, nullptr, &node_count),
        "count hot-path graph nodes");
    require(node_count == 1u,
        "captured hot path contains work other than its direct kernel");
    std::array<cudaGraphNode_t, 1u> nodes{};
    require_cuda(cudaGraphGetNodes(graph, nodes.data(), &node_count),
        "read hot-path graph node");
    cudaGraphNodeType node_type{};
    require_cuda(cudaGraphNodeGetType(nodes[0], &node_type),
        "inspect hot-path graph node");
    require(node_type == cudaGraphNodeTypeKernel,
        "captured hot path contains allocation, copy, or host work");
    cudaGraphExec_t graph_exec = nullptr;
    require_cuda(cudaGraphInstantiate(
                     &graph_exec, graph, nullptr, nullptr, 0u),
        "instantiate hot-path graph");
    require_cuda(cudaGraphLaunch(graph_exec, streams[0]),
        "launch hot-path graph first replay");
    check_output(buffers[1], -1.5f, views[0].scale, 3u, streams[0]);
    upload_input(buffers[0], 0.75f, streams[0]);
    require_cuda(cudaGraphLaunch(graph_exec, streams[0]),
        "launch hot-path graph second replay");
    check_output(buffers[1], 0.75f, views[0].scale, 3u, streams[0]);

    require(numeric_query_count == cold_numeric_queries
            && preparation_count == cold_preparations
            && projection_parse_count == cold_projection_parses,
        "sealed hot path repeated catalog, descriptor, or projection work");
    require(same_accounting(session.accounting, sealed_accounting),
        "sealed hot path changed allocation, query, handle, bind, or sync accounting");
    require(programs[0].run_count == 3u && programs[1].run_count == 1u
            && dispatch_count == 4u,
        "prepared dispatch count is inconsistent with host launches");

    require_cuda(cudaGraphExecDestroy(graph_exec), "destroy graph executable");
    require_cuda(cudaGraphDestroy(graph), "destroy graph");
    require(runtime::clear_value_readiness(&readiness)
            == runtime::value_readiness_status::success,
        "clear value readiness");
    runtime::clear_session(&session);
    for (auto &stream : streams)
        require_cuda(cudaStreamDestroy(stream), "destroy external stream");
    std::cout << "hot path validation passed plans=2 generations=3 graph_nodes=1"
              << " cold_preparations=" << cold_preparations << '\n';
    return EXIT_SUCCESS;
}
