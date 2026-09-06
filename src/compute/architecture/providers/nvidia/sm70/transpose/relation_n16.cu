// N16 arithmetic extracted from native_training_slice; no training epilogue.
// Expected limiter: sparse weight/index traffic and reuse with 16 accumulators.
#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/relation_n16.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstddef>
namespace cellerator::compute::math::core {
namespace {
operation_status fail(operation_status_code code, const char* message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
}
bool same_location(execution::device_location lhs,
    execution::device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

__global__ void transpose_backward_n16_kernel(
    transpose_projection_view projection,
    const __half *forward_values,
    const float *row_input,
    float *feature_output) {
    const std::uint32_t feature =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= projection.header.feature_count) return;
    float sum[16] = {};
    for (std::uint32_t edge = projection.feature_offsets[feature];
         edge < projection.feature_offsets[feature + 1u]; ++edge) {
        const std::uint32_t row = projection.execution_row_ids[edge];
        const std::uint32_t forward =
            projection.forward_value_positions[edge];
        const float weight = __half2float(forward_values[forward]);
        #pragma unroll
        for (unsigned k = 0; k < 16; ++k)
            sum[k] += weight * row_input[std::size_t(row) * 16 + k];
    }
    #pragma unroll
    for (unsigned k = 0; k < 16; ++k)
        feature_output[std::size_t(feature) * 16 + k] = sum[k];
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
        || state.dense_width != 16u || launch.input_count != 1u
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
        || input.shape[1] != 16u || input.stride[0] != 16
        || input.stride[1] != 1
        || output.value_type != execution::numeric_type::f32
        || output.rank != 2u
        || output.shape[0] != projection.header.feature_count
        || output.shape[1] != 16u || output.stride[0] != 16
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
    transpose_backward_n16_kernel<<<blocks, threads, 0u,
        static_cast<cudaStream_t>(launch.stream.stream)>>>(projection,
        static_cast<const __half *>(values.values),
        static_cast<const float *>(input.data),
        static_cast<float *>(output.data));
    if (cudaPeekAtLastError() != cudaSuccess)
        return fail(operation_status_code::execution_failed,
            "transpose backward kernel launch failed");
    return {};
}


operation_status prepare_impl(const operation_candidate& candidate,
    const operation_problem& problem, const structure_set_key& structures,
    const projection_key& projection, const numeric_policy& numeric,
    const prepare_policy& policy, prepared_operation* prepared) noexcept {
    if (!prepared || !prepared->persistent.data ||
        prepared->persistent.bytes != sizeof(transpose_backward_prepared_state))
        return fail(operation_status_code::preparation_failed, "N16 transpose state absent");
    auto* state = static_cast<transpose_backward_prepared_state*>(
        const_cast<void*>(prepared->persistent.data));
    if (!same_stable_id(candidate.identity, transpose_backward_n16_candidate_id) ||
        state->dense_width != 16 || problem.logical_work_items !=
            std::uint64_t(state->projection.header.nnz_count) * 16)
        return fail(operation_status_code::unsupported_problem, "N16 transpose work geometry incompatible");
    // Reuse the existing CTP1 identity/binding validation and contracts at cold
    // preparation only. The resulting plan has its own N16 identity and runner.
    auto scalar_problem = problem;
    scalar_problem.logical_work_items /= 16;
    state->dense_width = 1;
    auto scalar_candidate = transpose_backward_n1_candidate();
    auto status = prepare_candidate(scalar_candidate, scalar_problem, structures,
        projection, numeric, policy, prepared);
    state->dense_width = 16;
    if (!status) return status;
    prepared->problem = problem;
    prepared->kernel = candidate.identity;
    prepared->run = run_impl;
    return validate_prepared_operation(*prepared);
}
} // namespace
operation_candidate transpose_backward_n16_candidate() noexcept {
    auto candidate = transpose_backward_n1_candidate();
    candidate.identity = transpose_backward_n16_candidate_id;
    candidate.name = "cpbp-transpose-backward-n16-f16-f32";
    candidate.prepare = prepare_impl;
    return candidate;
}
operation_status register_transpose_backward_n16_candidate(candidate_registry* registry) noexcept {
    return register_candidate(registry, transpose_backward_n16_candidate());
}
operation_status prepare_transpose_backward_n16_operation(
    const operation_problem& problem, const structure_set_key& structures,
    const projection_key& projection, const numeric_policy& numeric,
    const prepare_policy& policy, const transpose_projection_view& view,
    std::int32_t device, execution::axis_identity source, execution::axis_identity destination,
    execution::axis_identity columns, transpose_backward_prepared_state* state,
    prepared_operation* prepared) noexcept {
    if (!state || !prepared)
        return fail(operation_status_code::invalid_argument, "N16 transpose output absent");
    *state = {};
    state->device_ordinal = device;
    state->dense_width = 16;
    state->projection = view;
    state->feature_axis = source;
    state->row_axis = destination;
    state->dense_column_axis = columns;
    *prepared = {};
    prepared->persistent = {state, sizeof(*state)};
    return prepare_candidate(transpose_backward_n16_candidate(), problem,
        structures, projection, numeric, policy, prepared);
}
} // namespace cellerator::compute::math::core
