#define main ce_live_training_program_contract_test
#include "training_program_test.cu"
#undef main

namespace {

void run_concurrency_acceptance() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice concurrency");
    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking),
        "create producer stream");
    require_cuda(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking),
        "create consumer stream");
    const cudaStream_t external_streams[]{producer, consumer};

    runtime::execution_session session{};
    runtime::execution_session_options options{};
    options.device = device;
    options.external_streams = external_streams;
    options.external_stream_count = 2u;
    options.owned_stream_count = 0u;
    require(runtime::init_session(&session, options)
            == runtime::session_status::success,
        "initialize two-stream execution session");

    fixture f;
    execution::training_program_request request{};
    request.forward = f.forward;
    request.transpose = f.transpose;
    request.feature_axis = f.feature_axis;
    request.module_axis = f.module_axis;
    request.dense_axis = f.dense_axis;
    request.session = &session;
    execution::training_program program{};
    require(execution::compile_training_program(request, &program),
        "compile concurrent training program");

    std::vector<float> host_input(f.features * 16u);
    std::vector<float> host_gradient(f.rows * 16u, 0.125f);
    for (std::size_t index = 0u; index < host_input.size(); ++index)
        host_input[index] = 0.25f
            + 0.001f * static_cast<float>(index);
    device_buffer<float> input_a(host_input.size());
    device_buffer<float> input_b(host_input.size());
    device_buffer<float> output_a(f.rows * 16u);
    device_buffer<float> output_b(f.rows * 16u);
    device_buffer<float> output_gradient(host_gradient.size());
    device_buffer<float> input_gradient(f.features * 16u);
    device_buffer<float> bias(16u);
    upload(input_a, host_input);
    upload(input_b, host_input);
    upload(output_gradient, host_gradient);
    require_cuda(cudaMemset(bias.data, 0, 16u * sizeof(float)),
        "zero concurrency bias");

    const std::size_t workspace_bytes =
        cm::native_training_workspace_bytes(f.rows, f.nnz);
    device_buffer<float> workspace(workspace_bytes / sizeof(float));
    cm::native_training_workspace workspace_view{};
    workspace_view.activated = workspace.data;
    workspace_view.preactivation_gradient = workspace.data + f.rows * 16u;
    workspace_view.inverse_rms = workspace.data + f.rows * 32u;
    workspace_view.sparse_gradient = workspace_view.inverse_rms + f.rows;
    workspace_view.bias_gradient = workspace_view.sparse_gradient + f.nnz;
    workspace_view.bytes = workspace_bytes;
    workspace_view.location = location(device);

    execution::value_plane plane{};
    plane.structure = f.structure;
    plane.structure_epoch_value = f.epoch;
    plane.values = f.values.data;
    plane.location = location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = f.nnz;
    plane.value_bytes = f.nnz * sizeof(__half);

    runtime::value_readiness_record generation_two;
    runtime::value_readiness_record generation_three;
    runtime::value_readiness_record generation_four;
    require(runtime::initialize_value_readiness(&generation_two, device)
                == runtime::value_readiness_status::success
            && runtime::initialize_value_readiness(&generation_three, device)
                == runtime::value_readiness_status::success
            && runtime::initialize_value_readiness(&generation_four, device)
                == runtime::value_readiness_status::success,
        "initialize generation readiness records");

    execution::training_program_launch first{};
    first.native.structure = {f.structure, f.epoch, f.feature_axis,
        f.module_axis, {1u, 1u}, f.nnz};
    first.native.learned_values = &plane;
    first.native.expected_generation = {1u};
    first.native.next_generation = {2u};
    first.native.next_value_readiness = &generation_two;
    first.native.input = dense(input_a.data, f.feature_axis,
        f.dense_axis, f.features, device);
    first.native.output = dense(output_a.data, f.module_axis,
        f.dense_axis, f.rows, device);
    first.native.output_gradient = dense(output_gradient.data,
        f.module_axis, f.dense_axis, f.rows, device);
    first.native.input_gradient = dense(input_gradient.data,
        f.feature_axis, f.dense_axis, f.features, device);
    first.native.bias = bias.data;
    first.native.bias_location = location(device);
    first.native.learning_rate = 1.0e-4f;
    first.native.normalization_epsilon = 1.0e-4f;
    first.native.stream = {producer, device, 0u};
    first.native.workspace = workspace_view;

    const auto accounting_before = session.accounting;
    execution::training_program_result first_result{};
    require(execution::run_training_program(&program, first, &first_result),
        "enqueue producer-stream generation");
    require(first_result.completion_stream.stream == producer
            && generation_two.generation() == 2u,
        "producer stream or readiness publication was lost");

    auto second = first;
    second.current_value_readiness = &generation_two;
    second.native.expected_generation = {2u};
    second.native.next_generation = {3u};
    second.native.next_value_readiness = &generation_three;
    second.native.input = dense(input_b.data, f.feature_axis,
        f.dense_axis, f.features, device);
    second.native.output = dense(output_b.data, f.module_axis,
        f.dense_axis, f.rows, device);
    second.native.stream = {consumer, device, 0u};
    execution::training_program_result second_result{};
    require(execution::run_training_program(&program, second, &second_result),
        "enqueue cross-stream generation");
    require_cuda(cudaStreamSynchronize(consumer),
        "synchronize cross-stream consumer");
    require(second_result.completion_stream.stream == consumer
            && plane.generation.value == 3u
            && generation_three.generation() == 3u
            && program.preparation_count == 1u && program.run_count == 2u,
        "two-stream reuse rebuilt state or lost generation identity");
    require(session.accounting.structure.allocation_count
                == accounting_before.structure.allocation_count
            && session.accounting.plan.allocation_count
                == accounting_before.plan.allocation_count
            && session.accounting.graph_stable.allocation_count
                == accounting_before.graph_stable.allocation_count
            && session.accounting.device_query_count
                == accounting_before.device_query_count
            && session.accounting.handle_prepare_count
                == accounting_before.handle_prepare_count
            && session.accounting.synchronization_count
                == accounting_before.synchronization_count,
        "hot run path changed allocation, device, handle, or sync accounting");

    auto stale_generation = second;
    stale_generation.current_value_readiness = &generation_two;
    require(execution::run_training_program(
        &program, stale_generation, &second_result).code
            == execution::training_program_status_code::stale_generation,
        "stale generation readiness was accepted");
    auto stale_structure = second;
    stale_structure.current_value_readiness = &generation_three;
    stale_structure.native.expected_generation = {3u};
    stale_structure.native.next_generation = {4u};
    stale_structure.native.structure.epoch.value += 1u;
    require(execution::run_training_program(
        &program, stale_structure, &second_result).code
            == execution::training_program_status_code::value_not_ready,
        "stale structure epoch was accepted");
    auto insufficient_workspace = second;
    insufficient_workspace.current_value_readiness = &generation_three;
    insufficient_workspace.native.expected_generation = {3u};
    insufficient_workspace.native.next_generation = {4u};
    insufficient_workspace.native.workspace.bytes -= sizeof(float);
    require(execution::run_training_program(
        &program, insufficient_workspace, &second_result).code
            == execution::training_program_status_code::insufficient_workspace,
        "insufficient graph-stable workspace was accepted");

    auto captured = second;
    captured.current_value_readiness = &generation_three;
    captured.native.expected_generation = {3u};
    captured.native.next_generation = {4u};
    captured.native.next_value_readiness = &generation_four;
    execution::training_program_result captured_result{};
    require_cuda(cudaStreamBeginCapture(
        consumer, cudaStreamCaptureModeThreadLocal), "begin CUDA graph capture");
    require(execution::run_training_program(
        &program, captured, &captured_result), "capture prepared training run");
    cudaGraph_t graph = nullptr;
    require_cuda(cudaStreamEndCapture(consumer, &graph),
        "end CUDA graph capture");
    std::size_t graph_nodes = 0u;
    require_cuda(cudaGraphGetNodes(graph, nullptr, &graph_nodes),
        "query captured graph nodes");
    require(graph_nodes >= 5u && captured_result.enqueued
            && captured_result.completion_stream.stream == consumer,
        "captured graph omitted prepared execution or readiness publication");
    cudaGraphExec_t graph_exec = nullptr;
    require_cuda(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0u),
        "instantiate captured graph");
    require_cuda(cudaGraphLaunch(graph_exec, consumer), "launch captured graph");
    require_cuda(cudaStreamSynchronize(consumer),
        "synchronize captured graph");
    require(plane.generation.value == 4u
            && generation_four.generation() == 4u
            && program.preparation_count == 1u && program.run_count == 3u,
        "captured execution changed preparation or generation contract");

    require_cuda(cudaGraphExecDestroy(graph_exec), "destroy graph executable");
    require_cuda(cudaGraphDestroy(graph), "destroy graph");
    require(runtime::clear_value_readiness(&generation_four)
                == runtime::value_readiness_status::success
            && runtime::clear_value_readiness(&generation_three)
                == runtime::value_readiness_status::success
            && runtime::clear_value_readiness(&generation_two)
                == runtime::value_readiness_status::success,
        "clear concurrency readiness records");
    runtime::clear_session(&session);
    require_cuda(cudaStreamDestroy(consumer), "destroy consumer stream");
    require_cuda(cudaStreamDestroy(producer), "destroy producer stream");
}

} // namespace

int main() {
    require(ce_live_training_program_contract_test() == 0,
        "base training program contract failed");
    run_concurrency_acceptance();
    return 0;
}
