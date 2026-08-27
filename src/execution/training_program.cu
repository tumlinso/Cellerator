#include <Cellerator/execution/training_program.hh>

namespace cellerator::execution {
namespace {

training_program_status fail(training_program_status_code code,
    const char *message) noexcept {
    training_program_status status{};
    status.code = code;
    status.message = message;
    return status;
}

training_program_status from_native(
    compute::math::native_training_status native) noexcept {
    training_program_status status{};
    status.native = native;
    status.message = native.message;
    using native_code = compute::math::native_training_status_code;
    switch (native.code) {
    case native_code::ok:
        return status;
    case native_code::invalid_argument:
        status.code = training_program_status_code::invalid_argument;
        break;
    case native_code::incompatible_identity:
        status.code = training_program_status_code::incompatible_identity;
        break;
    case native_code::stale_generation:
        status.code = training_program_status_code::stale_generation;
        break;
    case native_code::invalid_binding:
        status.code = training_program_status_code::invalid_binding;
        break;
    case native_code::insufficient_workspace:
        status.code = training_program_status_code::insufficient_workspace;
        break;
    case native_code::cuda_failure:
    case native_code::readiness_failure:
        status.code = training_program_status_code::execution_failed;
        break;
    }
    return status;
}

output_axis_contract preserved(axis_identity axis, std::uint8_t axis_index,
    std::uint16_t operand_index) noexcept {
    output_axis_contract result{};
    result.input_axis = axis;
    result.output_axis = axis;
    result.transition = order_transition_kind::preserve;
    result.axis_index = axis_index;
    result.operand_index = operand_index;
    result.may_remain_packed = 1u;
    return result;
}

} // namespace

training_program_status compile_training_program(
    const training_program_request &request,
    training_program *program) noexcept {
    if (program == nullptr
        || request.schema_version != training_program_schema_version
        || request.session == nullptr || !request.session->initialized
        || request.session->device < 0
        || request.dense_width != compute::math::native_training_dense_width)
        return fail(training_program_status_code::invalid_argument,
            "training program compile arguments are invalid");

    training_program compiled{};
    const auto native = compute::math::prepare_native_training_slice(
        request.forward, request.transpose, request.session->device,
        request.feature_axis, request.module_axis, request.dense_axis,
        &compiled.prepared);
    if (!native) return from_native(native);

    compiled.session = request.session;
    compiled.forward_projection = request.forward.header.projection_identity;
    compiled.transpose_projection = request.transpose.header.projection_identity;
    compiled.forward_output_order = preserved(request.module_axis, 0u, 0u);
    compiled.input_gradient_order = preserved(request.feature_axis, 0u, 1u);
    compiled.preparation_count = 1u;
    *program = compiled;
    return {};
}

training_program_status run_training_program(
    training_program *program,
    const training_program_launch &launch,
    training_program_result *result) noexcept {
    if (program == nullptr || result == nullptr
        || program->schema_version != training_program_schema_version
        || program->session == nullptr || !program->session->initialized
        || program->session->device != program->prepared.device_ordinal
        || launch.native.stream.device_ordinal != program->session->device)
        return fail(training_program_status_code::invalid_argument,
            "training program launch arguments are invalid");

    runtime::value_readiness_status readiness =
        runtime::value_readiness_status::success;
    if (launch.current_value_readiness != nullptr) {
        readiness = runtime::wait_for_value_generation(
            *launch.current_value_readiness,
            launch.native.structure.epoch.value,
            launch.native.expected_generation.value,
            static_cast<cudaStream_t>(launch.native.stream.stream),
            launch.native.stream.device_ordinal);
    } else if (launch.native.expected_generation.value != 1u) {
        readiness = runtime::value_readiness_status::invalid_state;
    }
    if (readiness != runtime::value_readiness_status::success) {
        training_program_status status = fail(
            training_program_status_code::value_not_ready,
            "training input generation is not ready on the caller stream");
        status.readiness = readiness;
        return status;
    }

    const value_generation consumed = launch.native.expected_generation;
    const auto native = compute::math::run_native_training_step(
        program->prepared, launch.native);
    if (!native) return from_native(native);

    compute::math::native_training_parameter_descriptors descriptors{};
    const auto described = compute::math::describe_native_training_parameters(
        program->prepared, launch.native, &descriptors);
    if (!described) return from_native(described);

    ++program->run_count;
    *result = training_program_result{};
    result->backend = program->backend;
    result->forward_projection = program->forward_projection;
    result->transpose_projection = program->transpose_projection;
    result->forward_output_order = program->forward_output_order;
    result->input_gradient_order = program->input_gradient_order;
    result->structure_epoch_value = launch.native.structure.epoch;
    result->consumed_generation = consumed;
    result->published_generation = native.published_generation;
    result->completion_stream = launch.native.stream;
    result->readiness = native.readiness;
    result->parameter_count = descriptors.count;
    for (std::size_t index = 0u; index < descriptors.count; ++index)
        result->parameters[index] = descriptors.parameters[index];
    result->enqueued = true;
    return {};
}

} // namespace cellerator::execution
