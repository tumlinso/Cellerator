/*
CE-ARCH-85 native transpose/backward evidence (2026-08-25, V100 sm_70): the
focused celleratorTransposeBackwardCandidateTest compares CTP1 execution for
two mutable f16 value generations against the independent transposed-SpMM
referee at 1e-5 absolute/relative tolerance. No maintained sparse library can
consume CTP1 or share its forward value positions, so this task makes no
performance promotion claim; the direct kernel exists to prove native reverse
execution without CSR reconstruction, allocation, or order conversion.
*/

#include <Cellerator/compute/math/operation_core/transpose_backward_candidate.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::math::core {
namespace {

operation_status fail(operation_status_code code,
    const char *message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
}

bool supports_numeric(const numeric_policy &numeric) noexcept {
    return numeric.sparse_storage == execution::numeric_type::f16
        && numeric.dense_storage == execution::numeric_type::f32
        && numeric.output_storage == execution::numeric_type::f32
        && numeric.multiply == execution::numeric_type::f32
        && numeric.accumulation == execution::numeric_type::f32
        && numeric.scalar == execution::numeric_type::f32
        && numeric.bias == execution::numeric_type::invalid
        && numeric.rounding == rounding_policy::nearest_even
        && numeric.saturation == saturation_policy::none
        && numeric.quantization == quantization_granularity::none;
}

bool same_location(execution::device_location lhs,
    execution::device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

bool valid_device_projection(const transpose_projection_view &view,
    std::int32_t device) noexcept {
    const auto &header = view.header;
    return header.schema_version == transpose_projection_schema_version
        && header.payload_kind == transpose_projection_payload_kind
        && header.header_bytes == sizeof(header)
        && header.alignment == transpose_projection_alignment
        && header.payload_bytes != 0u
        && execution::valid_identity(header.structure_identity)
        && execution::valid_identity(header.projection_identity)
        && execution::valid_identity(header.forward_projection_identity)
        && !execution::same_identity(header.projection_identity,
            header.forward_projection_identity)
        && header.structure_epoch != 0u && header.nnz_count != 0u
        && header.row_count != 0u && header.feature_count != 0u
        && header.value_size_bytes == sizeof(__half)
        && execution::valid_handle(view.runtime_structure)
        && execution::valid_handle(view.runtime_projection)
        && execution::valid_handle(view.runtime_forward_projection)
        && view.payload_base != nullptr && view.feature_offsets != nullptr
        && view.execution_row_ids != nullptr
        && view.forward_value_positions != nullptr
        && view.logical_to_transpose != nullptr
        && view.transpose_to_logical != nullptr && device >= 0;
}

__global__ void transpose_backward_n1_kernel(
    transpose_projection_view projection,
    const __half *forward_values,
    const float *row_input,
    float *feature_output) {
    const std::uint32_t feature =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= projection.header.feature_count) return;
    float sum = 0.0f;
    for (std::uint32_t edge = projection.feature_offsets[feature];
         edge < projection.feature_offsets[feature + 1u]; ++edge) {
        const std::uint32_t row = projection.execution_row_ids[edge];
        const std::uint32_t forward =
            projection.forward_value_positions[edge];
        sum += __half2float(forward_values[forward]) * row_input[row];
    }
    feature_output[feature] = sum;
}

operation_status run_impl(const prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    if (prepared.persistent.data == nullptr
        || prepared.persistent.bytes
            != sizeof(transpose_backward_prepared_state))
        return fail(operation_status_code::execution_failed,
            "transpose backward prepared state is absent");
    const auto &state = *static_cast<
        const transpose_backward_prepared_state *>(prepared.persistent.data);
    if (state.schema_version != transpose_backward_candidate_schema_version
        || state.dense_width != 1u || launch.input_count != 1u
        || launch.output_count != 1u || launch.value_count != 1u
        || launch.values == nullptr
        || launch.inputs[0].kind != execution::operand_kind::dense_tensor
        || launch.outputs[0].kind != execution::operand_kind::dense_tensor)
        return fail(operation_status_code::invalid_launch_bindings,
            "transpose backward launch arity or state is incompatible");
    const auto &input = launch.inputs[0].storage.dense;
    const auto &output = launch.outputs[0].storage.dense;
    const auto &values = *launch.values[0].plane;
    const auto &structure = launch.structures[0];
    const auto &projection = state.projection;
    if (!execution::same_axis_identity(structure.source_axis,
            state.feature_axis)
        || !execution::same_axis_identity(structure.destination_axis,
            state.row_axis)
        || !execution::same_handle(structure.identity,
            projection.runtime_structure)
        || structure.epoch.value != projection.header.structure_epoch
        || structure.logical_edge_count != projection.header.nnz_count
        || input.value_type != execution::numeric_type::f32
        || input.rank != 2u || input.shape[0] != projection.header.row_count
        || input.shape[1] != 1u || input.stride[0] != 1
        || input.stride[1] != 1
        || output.value_type != execution::numeric_type::f32
        || output.rank != 2u
        || output.shape[0] != projection.header.feature_count
        || output.shape[1] != 1u || output.stride[0] != 1
        || output.stride[1] != 1
        || values.numeric.storage != execution::numeric_type::f16
        || values.numeric.dequantized != execution::numeric_type::f32
        || values.numeric.accumulation != execution::numeric_type::f32
        || values.layout != execution::value_layout_kind::projection_local_order
        || values.element_count != projection.header.nnz_count
        || values.value_bytes != values.element_count * sizeof(__half)
        || input.location.residency == execution::residency_kind::host
        || output.location.residency == execution::residency_kind::host
        || values.location.residency == execution::residency_kind::host
        || !same_location(input.location, output.location)
        || !same_location(input.location, values.location)
        || input.location.device_ordinal != state.device_ordinal
        || launch.stream.device_ordinal != state.device_ordinal)
        return fail(operation_status_code::invalid_launch_bindings,
            "transpose backward order, shape, value, or residency is incompatible");
    constexpr std::uint32_t threads = 128u;
    const std::uint32_t blocks =
        (projection.header.feature_count + threads - 1u) / threads;
    transpose_backward_n1_kernel<<<blocks, threads, 0u,
        static_cast<cudaStream_t>(launch.stream.stream)>>>(projection,
        static_cast<const __half *>(values.values),
        static_cast<const float *>(input.data),
        static_cast<float *>(output.data));
    if (cudaPeekAtLastError() != cudaSuccess)
        return fail(operation_status_code::execution_failed,
            "transpose backward kernel launch failed");
    return {};
}

operation_status prepare_impl(const operation_candidate &candidate,
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &,
    prepared_operation *prepared) noexcept {
    if (prepared == nullptr || prepared->persistent.data == nullptr
        || prepared->persistent.bytes
            != sizeof(transpose_backward_prepared_state))
        return fail(operation_status_code::preparation_failed,
            "transpose backward requires caller-owned prebound state");
    auto *state = static_cast<transpose_backward_prepared_state *>(
        const_cast<void *>(prepared->persistent.data));
    const auto &view = state->projection;
    const auto &header = view.header;
    if (!same_stable_id(candidate.identity,
            transpose_backward_n1_candidate_id)
        || state->schema_version != transpose_backward_candidate_schema_version
        || state->dense_width != 1u
        || !valid_device_projection(view, state->device_ordinal)
        || !execution::valid_axis_identity(state->feature_axis)
        || !execution::valid_axis_identity(state->row_axis)
        || !execution::valid_axis_identity(state->dense_column_axis)
        || structures.count != 1u || problem.input_count != 1u
        || problem.output_count != 1u
        || problem.logical_work_items != header.nnz_count
        || !execution::same_identity(structures.structures[0].persistent,
            header.structure_identity)
        || !execution::same_handle(structures.structures[0].runtime,
            view.runtime_structure)
        || structures.structures[0].epoch.value != header.structure_epoch
        || !execution::same_identity(projection.persistent,
            header.projection_identity)
        || !execution::same_handle(projection.runtime,
            view.runtime_projection)
        || projection.kind != projection_kind::transpose_or_backward
        || projection.schema_version != transpose_projection_schema_version
        || projection.variant != transpose_projection_variant)
        return fail(operation_status_code::unsupported_problem,
            "transpose backward preparation metadata is incompatible");

    state->input_contract = {};
    state->input_contract.kind = execution::operand_kind::dense_tensor;
    state->input_contract.rank = 2u;
    state->input_contract.axes[0] = state->row_axis;
    state->input_contract.axes[1] = state->dense_column_axis;
    state->output_contract = {};
    state->output_contract.kind = execution::operand_kind::dense_tensor;
    state->output_contract.rank = 2u;
    state->output_contract.axes[0] = state->feature_axis;
    state->output_contract.axes[1] = state->dense_column_axis;
    state->output_orders[0] = {state->feature_axis, state->feature_axis,
        execution::order_transition_kind::preserve, 0u, 0u, 0u, 1u, {}, {}};
    state->output_orders[1] = {state->dense_column_axis,
        state->dense_column_axis, execution::order_transition_kind::preserve,
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
    prepared->binding_contract.output_order_count = 2u;
    prepared->binding_contract.structure_count = 1u;
    prepared->binding_contract.output_effect_count = 1u;
    prepared->binding_contract.workspace = {0u, 1u, 0u};
    prepared->run = run_impl;
    return validate_prepared_operation(*prepared);
}

} // namespace

operation_candidate transpose_backward_n1_candidate() noexcept {
    operation_candidate candidate{};
    candidate.identity = transpose_backward_n1_candidate_id;
    candidate.name = "cpbp-transpose-backward-n1-f16-f32";
    candidate.operation = operation_kind::sparse_dense_multiply;
    candidate.projection = projection_kind::transpose_or_backward;
    candidate.backend = backend_kind::native_direct;
    candidate.capability_flags = candidate_deterministic
        | candidate_graph_capture | candidate_persistent_preprocessing;
    candidate.persistent_bytes = sizeof(transpose_backward_prepared_state);
    candidate.transient_bytes = 0u;
    candidate.supports_numeric = supports_numeric;
    candidate.prepare = prepare_impl;
    return candidate;
}

operation_status register_transpose_backward_n1_candidate(
    candidate_registry *registry) noexcept {
    return register_candidate(registry, transpose_backward_n1_candidate());
}

operation_status prepare_transpose_backward_n1_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const transpose_projection_view &device_projection,
    std::int32_t device_ordinal,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    execution::axis_identity dense_column_axis,
    transpose_backward_prepared_state *state,
    prepared_operation *prepared) noexcept {
    if (state == nullptr || prepared == nullptr)
        return fail(operation_status_code::invalid_argument,
            "transpose backward preparation output is null");
    *state = {};
    state->device_ordinal = device_ordinal;
    state->dense_width = 1u;
    state->projection = device_projection;
    state->feature_axis = feature_axis;
    state->row_axis = row_axis;
    state->dense_column_axis = dense_column_axis;
    *prepared = {};
    prepared->persistent = {state, sizeof(*state)};
    const operation_candidate candidate = transpose_backward_n1_candidate();
    return prepare_candidate(candidate, problem, structures, projection,
        numeric, policy, prepared);
}

} // namespace cellerator::compute::math::core
