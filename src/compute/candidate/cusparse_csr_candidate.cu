#include <Cellerator/compute/candidate/cusparse_csr_candidate.hh>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::math::core {
namespace {

operation_status fail(operation_status_code code, const char *message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
}

operation_status vendor_failure(const char *message) noexcept {
    return fail(operation_status_code::preparation_failed, message);
}

bool supports_numeric(const numeric_policy &numeric) noexcept {
    return numeric.sparse_storage == execution::numeric_type::f32
        && numeric.dense_storage == execution::numeric_type::f32
        && numeric.output_storage == execution::numeric_type::f32
        && numeric.multiply == execution::numeric_type::f32
        && numeric.accumulation == execution::numeric_type::f32
        && numeric.scalar == execution::numeric_type::u32
        && numeric.bias == execution::numeric_type::invalid
        && numeric.rounding == rounding_policy::nearest_even
        && numeric.saturation == saturation_policy::none
        && numeric.quantization == quantization_granularity::none;
}

bool supported_width(std::uint32_t columns) noexcept {
    return columns == 1u || columns == 16u || columns == 17u
        || columns == 31u || columns == 32u || columns == 48u
        || columns == 64u;
}

bool valid_device_csr(const execution_csr_view &view) noexcept {
    return view.schema_version == execution_csr_schema_version
        && view.structure.schema_version
            == sparse_structure_identity_schema_version
        && view.structure.identity_version
            == execution_csr_structure_identity_version
        && view.structure.value != 0u
        && view.feature_order.schema_version
            == feature_order_identity_schema_version
        && view.feature_order.kind == feature_order_kind::packed
        && view.feature_order.feature_count == view.feature_count
        && view.feature_order.feature_axis_identity != 0u
        && view.feature_order.feature_axis_identity_version != 0u
        && view.feature_order.packing_geometry_identity != 0u
        && view.row_domain_identity != 0u
        && view.row_count != 0u && view.feature_count != 0u
        && view.nnz_count != 0u && view.value_size_bytes == sizeof(float)
        && view.row_offsets != nullptr
        && view.execution_feature_ids != nullptr && view.values != nullptr;
}

bool same_location(
    execution::device_location lhs,
    execution::device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

operation_status validate_launch(
    const cusparse_csr_prepared_state &state,
    const execution::launch_bindings &launch,
    const execution::dense_tensor_view **input,
    execution::dense_tensor_view **output,
    const execution::value_plane **values) noexcept {
    if (launch.input_count != 1u || launch.output_count != 1u
        || launch.value_count != 1u || launch.inputs == nullptr
        || launch.outputs == nullptr || launch.values == nullptr
        || launch.values[0].plane == nullptr
        || launch.inputs[0].kind != execution::operand_kind::dense_tensor
        || launch.outputs[0].kind != execution::operand_kind::dense_tensor)
        return fail(operation_status_code::invalid_launch_bindings,
            "cuSPARSE CSR launch arity is invalid");
    const auto &dense_input = launch.inputs[0].storage.dense;
    auto &dense_output = launch.outputs[0].storage.dense;
    const auto &value_plane = *launch.values[0].plane;
    const bool vector = state.operation == cusparse_csr_operation::spmv;
    if ((vector && (dense_input.rank != 1u || dense_output.rank != 1u
            || dense_input.shape[0] != state.projection.feature_count
            || dense_output.shape[0] != state.projection.row_count
            || dense_input.stride[0] != 1 || dense_output.stride[0] != 1))
        || (!vector && (dense_input.rank != 2u || dense_output.rank != 2u
            || dense_input.shape[0] != state.projection.feature_count
            || dense_output.shape[0] != state.projection.row_count
            || dense_input.shape[1] != state.dense_columns
            || dense_output.shape[1] != state.dense_columns
            || dense_input.stride[0] != state.dense_columns
            || dense_output.stride[0] != state.dense_columns
            || dense_input.stride[1] != 1 || dense_output.stride[1] != 1))
        || dense_input.value_type != execution::numeric_type::f32
        || dense_output.value_type != execution::numeric_type::f32
        || value_plane.numeric.storage != execution::numeric_type::f32
        || value_plane.numeric.dequantized != execution::numeric_type::f32
        || value_plane.numeric.accumulation != execution::numeric_type::f32
        || value_plane.layout
            != execution::value_layout_kind::projection_local_order
        || value_plane.element_count != state.projection.nnz_count
        || value_plane.value_bytes
            != value_plane.element_count * sizeof(float)
        || dense_input.location.residency == execution::residency_kind::host
        || dense_output.location.residency == execution::residency_kind::host
        || value_plane.location.residency == execution::residency_kind::host
        || !same_location(dense_input.location, dense_output.location)
        || !same_location(dense_input.location, value_plane.location)
        || dense_input.location.device_ordinal != state.device_ordinal
        || launch.stream.device_ordinal != state.device_ordinal
        || launch.stream.stream != state.prepared_stream)
        return fail(operation_status_code::invalid_launch_bindings,
            "cuSPARSE CSR shape, stride, value, residency, or stream is invalid");
    *input = &dense_input;
    *output = &dense_output;
    *values = &value_plane;
    return {};
}

operation_status run_cusparse_csr(
    const prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    if (prepared.persistent.data == nullptr
        || prepared.persistent.bytes != sizeof(cusparse_csr_prepared_state))
        return fail(operation_status_code::execution_failed,
            "cuSPARSE CSR prepared state is absent");
    auto &state = *static_cast<cusparse_csr_prepared_state *>(
        const_cast<void *>(prepared.persistent.data));
    const execution::dense_tensor_view *input = nullptr;
    execution::dense_tensor_view *output = nullptr;
    const execution::value_plane *values = nullptr;
    const operation_status valid = validate_launch(
        state, launch, &input, &output, &values);
    if (!valid) return valid;
    if (cusparseSpMatSetValues(state.sparse,
            const_cast<void *>(values->values)) != CUSPARSE_STATUS_SUCCESS)
        return fail(operation_status_code::execution_failed,
            "cuSPARSE CSR value pointer rebind failed");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (state.operation == cusparse_csr_operation::spmv) {
        if (cusparseDnVecSetValues(state.input_vector, input->data)
                != CUSPARSE_STATUS_SUCCESS
            || cusparseDnVecSetValues(state.output_vector, output->data)
                != CUSPARSE_STATUS_SUCCESS)
            return fail(operation_status_code::execution_failed,
                "cuSPARSE SpMV dense pointer rebind failed");
        if (cusparseSpMV(state.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, state.sparse, state.input_vector, &beta,
                state.output_vector, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                state.preprocessing_workspace) != CUSPARSE_STATUS_SUCCESS)
            return fail(operation_status_code::execution_failed,
                "cuSPARSE SpMV execution failed");
    } else {
        if (cusparseDnMatSetValues(state.input_matrix, input->data)
                != CUSPARSE_STATUS_SUCCESS
            || cusparseDnMatSetValues(state.output_matrix, output->data)
                != CUSPARSE_STATUS_SUCCESS)
            return fail(operation_status_code::execution_failed,
                "cuSPARSE SpMM dense pointer rebind failed");
        if (cusparseSpMM(state.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, state.sparse,
                state.input_matrix, &beta, state.output_matrix, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, state.preprocessing_workspace)
            != CUSPARSE_STATUS_SUCCESS)
            return fail(operation_status_code::execution_failed,
                "cuSPARSE SpMM execution failed");
    }
    return {};
}

operation_status prepare_impl(
    const operation_candidate &candidate,
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &,
    prepared_operation *prepared) noexcept {
    if (prepared == nullptr || prepared->persistent.data == nullptr
        || prepared->persistent.bytes != sizeof(cusparse_csr_prepared_state))
        return fail(operation_status_code::preparation_failed,
            "cuSPARSE CSR requires caller-owned prepared state");
    auto *state = static_cast<cusparse_csr_prepared_state *>(
        const_cast<void *>(prepared->persistent.data));
    const bool vector = state->operation == cusparse_csr_operation::spmv;
    if (state->schema_version != cusparse_csr_candidate_schema_version
        || !valid_device_csr(state->projection)
        || !supported_width(state->dense_columns)
        || vector != (state->dense_columns == 1u)
        || state->handle == nullptr || state->sparse == nullptr
        || (vector && (state->input_vector == nullptr
            || state->output_vector == nullptr))
        || (!vector && (state->input_matrix == nullptr
            || state->output_matrix == nullptr))
        || !execution::valid_axis_identity(state->feature_axis)
        || !execution::valid_axis_identity(state->row_axis)
        || (!vector && !execution::valid_axis_identity(state->column_axis))
        || structures.count != 1u || problem.input_count != 1u
        || problem.output_count != 1u
        || problem.logical_work_items
            != static_cast<std::uint64_t>(state->projection.nnz_count)
                * state->dense_columns
        || projection.schema_version != execution_csr_schema_version
        || projection.variant != 1u)
        return fail(operation_status_code::unsupported_problem,
            "cuSPARSE CSR preparation metadata is incompatible");

    state->input_contract = {};
    state->input_contract.kind = execution::operand_kind::dense_tensor;
    state->input_contract.rank = vector ? 1u : 2u;
    state->input_contract.axes[0] = state->feature_axis;
    if (!vector) state->input_contract.axes[1] = state->column_axis;
    state->output_contract = {};
    state->output_contract.kind = execution::operand_kind::dense_tensor;
    state->output_contract.rank = vector ? 1u : 2u;
    state->output_contract.axes[0] = state->row_axis;
    if (!vector) state->output_contract.axes[1] = state->column_axis;
    state->output_orders[0] = {state->row_axis, state->row_axis,
        execution::order_transition_kind::preserve, 0u, 0u, 0u, 1u, {}, {}};
    if (!vector)
        state->output_orders[1] = {state->column_axis, state->column_axis,
            execution::order_transition_kind::preserve,
            1u, 0u, 0u, 1u, {}, {}};
    state->output_effect = {execution::output_update_kind::overwrite,
        false, false, 0u, execution::invalid_scalar_binding_id,
        execution::invalid_scalar_binding_id};

    const persistent_kernel_state persistent = prepared->persistent;
    *prepared = {};
    prepared->problem = problem;
    prepared->structures = structures;
    prepared->projection = projection;
    prepared->numeric = numeric;
    prepared->kernel = candidate.identity;
    prepared->backend = candidate.backend;
    prepared->capability_flags = candidate.capability_flags;
    prepared->persistent = persistent;
    prepared->binding_contract.structures[0] = {
        structures.structures[0].runtime, structures.structures[0].epoch};
    prepared->binding_contract.inputs = &state->input_contract;
    prepared->binding_contract.outputs = &state->output_contract;
    prepared->binding_contract.output_orders = state->output_orders;
    prepared->binding_contract.output_effects = &state->output_effect;
    prepared->binding_contract.input_count = 1u;
    prepared->binding_contract.output_count = 1u;
    prepared->binding_contract.output_order_count = vector ? 1u : 2u;
    prepared->binding_contract.structure_count = 1u;
    prepared->binding_contract.output_effect_count = 1u;
    prepared->binding_contract.workspace = {0u, 1u, 0u};
    prepared->run = run_cusparse_csr;
    return validate_prepared_operation(*prepared);
}

operation_candidate candidate(
    cusparse_csr_operation operation) noexcept {
    operation_candidate result{};
    const bool vector = operation == cusparse_csr_operation::spmv;
    result.identity = vector
        ? cusparse_csr_spmv_candidate_id : cusparse_csr_spmm_candidate_id;
    result.name = vector
        ? "cellerator-cusparse-csr-spmv-f32"
        : "cellerator-cusparse-csr-spmm-f32";
    result.operation = vector
        ? operation_kind::weighted_relation_reduce
        : operation_kind::sparse_dense_multiply;
    result.projection = projection_kind::csr;
    result.backend = backend_kind::vendor_library;
    result.capability_flags = candidate_deterministic
        | candidate_persistent_preprocessing;
    result.persistent_bytes = sizeof(cusparse_csr_prepared_state);
    result.transient_bytes = 0u;
    result.supports_numeric = supports_numeric;
    result.prepare = prepare_impl;
    return result;
}

} // namespace

operation_candidate cusparse_csr_spmv_candidate() noexcept {
    return candidate(cusparse_csr_operation::spmv);
}

operation_candidate cusparse_csr_spmm_candidate() noexcept {
    return candidate(cusparse_csr_operation::spmm);
}

operation_status register_cusparse_csr_candidates(
    candidate_registry *registry) noexcept {
    if (registry == nullptr)
        return fail(operation_status_code::invalid_argument,
            "cuSPARSE CSR registration requires a registry");
    candidate_registry staged = *registry;
    operation_status status = register_candidate(
        &staged, cusparse_csr_spmv_candidate());
    if (!status) return status;
    status = register_candidate(&staged, cusparse_csr_spmm_candidate());
    if (!status) return status;
    *registry = staged;
    return {};
}

operation_status prepare_cusparse_csr_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const execution_csr_view &device_csr,
    runtime::execution_session *session,
    std::uint32_t stream_index,
    std::uint32_t dense_columns,
    void *initial_dense,
    void *initial_output,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    execution::axis_identity column_axis,
    cusparse_csr_prepared_state *state,
    prepared_operation *prepared) noexcept {
    if (state == nullptr || prepared == nullptr || session == nullptr
        || initial_dense == nullptr || initial_output == nullptr)
        return fail(operation_status_code::invalid_argument,
            "cuSPARSE CSR preparation argument is null");
    if (!session->initialized || session->sealed
        || stream_index >= session->stream_count
        || !session->streams[stream_index].libraries_prepared
        || session->streams[stream_index].cusparse.handle == nullptr
        || session->streams[stream_index].execution.device != session->device
        || !valid_device_csr(device_csr) || !supported_width(dense_columns))
        return fail(operation_status_code::unsupported_problem,
            "cuSPARSE CSR session, projection, or width is invalid");
    const bool vector = dense_columns == 1u;
    const operation_candidate selected = vector
        ? cusparse_csr_spmv_candidate() : cusparse_csr_spmm_candidate();
    const operation_status problem_status =
        validate_operation_problem(problem, structures);
    if (!problem_status) return problem_status;
    const operation_status numeric_status = validate_numeric_policy(numeric);
    if (!numeric_status) return numeric_status;
    if (problem.kind != selected.operation)
        return fail(operation_status_code::unsupported_problem,
            "cuSPARSE CSR operation kind does not match dense width");
    if (projection.kind != projection_kind::csr
        || projection.schema_version != execution_csr_schema_version
        || projection.variant != 1u)
        return fail(operation_status_code::unsupported_projection,
            "cuSPARSE CSR projection contract is incompatible");
    if (!supports_numeric(numeric))
        return fail(operation_status_code::unsupported_numeric_policy,
            "cuSPARSE CSR requires explicit f32 CSR and f32 dense arithmetic");
    if (policy.graph_capture_required)
        return fail(operation_status_code::capability_rejected,
            "cuSPARSE CSR pointer rebinding is not graph-capture compatible");
    if (!policy.allow_persistent_preprocessing)
        return fail(operation_status_code::capability_rejected,
            "cuSPARSE CSR persistent descriptors are disabled by policy");
    if (device_csr.row_count > static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max())
        || device_csr.feature_count > static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max())
        || device_csr.nnz_count > static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max()))
        return fail(operation_status_code::unsupported_problem,
            "cuSPARSE CSR dimensions exceed the 32-bit projection ABI");

    *state = {};
    state->operation = vector
        ? cusparse_csr_operation::spmv : cusparse_csr_operation::spmm;
    state->device_ordinal = session->device;
    state->dense_columns = dense_columns;
    state->projection = device_csr;
    state->feature_axis = feature_axis;
    state->row_axis = row_axis;
    state->column_axis = column_axis;
    auto &slot = session->streams[stream_index];
    state->prepared_stream = slot.execution.stream;
    state->handle = slot.cusparse.handle;
    state->costs.descriptor_state_bytes = sizeof(*state);

    const auto rows = static_cast<std::int64_t>(device_csr.row_count);
    const auto columns = static_cast<std::int64_t>(device_csr.feature_count);
    const auto nnz = static_cast<std::int64_t>(device_csr.nnz_count);
    if (cusparseCreateCsr(&state->sparse, rows, columns, nnz,
            const_cast<std::uint32_t *>(device_csr.row_offsets),
            const_cast<std::uint32_t *>(device_csr.execution_feature_ids),
            const_cast<void *>(device_csr.values), CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
        != CUSPARSE_STATUS_SUCCESS) {
        clear_cusparse_csr_prepared_state(state);
        return vendor_failure("cusparseCreateCsr failed");
    }
    ++state->costs.descriptor_create_calls;

    cusparseStatus_t descriptor_status = CUSPARSE_STATUS_SUCCESS;
    if (vector) {
        descriptor_status = cusparseCreateDnVec(
            &state->input_vector, columns, initial_dense, CUDA_R_32F);
        if (descriptor_status == CUSPARSE_STATUS_SUCCESS)
            descriptor_status = cusparseCreateDnVec(
                &state->output_vector, rows, initial_output, CUDA_R_32F);
    } else {
        descriptor_status = cusparseCreateDnMat(&state->input_matrix,
            columns, dense_columns, dense_columns, initial_dense,
            CUDA_R_32F, CUSPARSE_ORDER_ROW);
        if (descriptor_status == CUSPARSE_STATUS_SUCCESS)
            descriptor_status = cusparseCreateDnMat(&state->output_matrix,
                rows, dense_columns, dense_columns, initial_output,
                CUDA_R_32F, CUSPARSE_ORDER_ROW);
    }
    if (descriptor_status != CUSPARSE_STATUS_SUCCESS) {
        clear_cusparse_csr_prepared_state(state);
        return vendor_failure("cuSPARSE dense descriptor creation failed");
    }
    state->costs.descriptor_create_calls += 2u;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::size_t workspace_bytes = 0u;
    cusparseStatus_t query_status = vector
        ? cusparseSpMV_bufferSize(state->handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, state->sparse,
            state->input_vector, &beta, state->output_vector, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &workspace_bytes)
        : cusparseSpMM_bufferSize(state->handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, state->sparse,
            state->input_matrix, &beta, state->output_matrix, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &workspace_bytes);
    if (query_status != CUSPARSE_STATUS_SUCCESS) {
        clear_cusparse_csr_prepared_state(state);
        return vendor_failure("cuSPARSE workspace query failed");
    }
    if (policy.persistent_memory_limit != 0u
        && (workspace_bytes > policy.persistent_memory_limit
            || sizeof(*state) > policy.persistent_memory_limit
                - workspace_bytes)) {
        clear_cusparse_csr_prepared_state(state);
        return fail(operation_status_code::capability_rejected,
            "cuSPARSE CSR exceeds the complete persistent memory limit");
    }
    if (workspace_bytes != 0u
        && runtime::reserve_persistent(session, runtime::persistent_lifetime::plan,
            workspace_bytes, &state->preprocessing_workspace)
            != runtime::session_status::success) {
        clear_cusparse_csr_prepared_state(state);
        return vendor_failure("session plan workspace reservation failed");
    }
    state->preprocessing_workspace_bytes = workspace_bytes;
    state->costs.preprocessing_workspace_bytes = workspace_bytes;
    // CUDA 12.9 exposes an explicit SpMM preprocessing entry point. The SpMV
    // descriptor and workspace are still fully prepared here, but this
    // toolchain has no separate SpMV preprocess API to invoke.
    cusparseStatus_t preprocess_status = CUSPARSE_STATUS_SUCCESS;
    if (!vector)
        preprocess_status = cusparseSpMM_preprocess(state->handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, state->sparse,
            state->input_matrix, &beta, state->output_matrix, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, state->preprocessing_workspace);
    if (preprocess_status != CUSPARSE_STATUS_SUCCESS) {
        clear_cusparse_csr_prepared_state(state);
        return vendor_failure("cuSPARSE preprocessing failed");
    }
    state->costs.preprocess_calls = vector ? 0u : 1u;

    *prepared = {};
    prepared->persistent = {state, sizeof(*state)};
    const operation_status status = prepare_candidate(selected, problem,
        structures, projection, numeric, policy, prepared);
    if (!status) clear_cusparse_csr_prepared_state(state);
    return status;
}

void clear_cusparse_csr_prepared_state(
    cusparse_csr_prepared_state *state) noexcept {
    if (state == nullptr) return;
    if (state->output_matrix != nullptr)
        (void) cusparseDestroyDnMat(state->output_matrix);
    if (state->input_matrix != nullptr)
        (void) cusparseDestroyDnMat(state->input_matrix);
    if (state->output_vector != nullptr)
        (void) cusparseDestroyDnVec(state->output_vector);
    if (state->input_vector != nullptr)
        (void) cusparseDestroyDnVec(state->input_vector);
    if (state->sparse != nullptr)
        (void) cusparseDestroySpMat(state->sparse);
    *state = {};
}

cusparse_csr_preparation_costs cusparse_csr_costs(
    const cusparse_csr_prepared_state &state) noexcept {
    return state.costs;
}

} // namespace cellerator::compute::math::core
