#include <Cellerator/compute/candidate/csr_fallback_candidate.hh>

#include "compute/operators/sparse/primitives/common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::math::core {
namespace {

namespace primitives = compute::sparse::ops::primitives;

#include "compute/operators/sparse/kernels/base_sparse/csr_spmv_fwd_kernel_.cuh"

inline constexpr int csr_row_threads = 256;

operation_status fail(operation_status_code code, const char *message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
}

bool supports_numeric(const numeric_policy &numeric) noexcept {
    return numeric.sparse_storage == execution::numeric_type::f16
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

bool valid_preconstructed_device_csr(
    const execution_csr_view &view, std::int32_t device_ordinal) noexcept {
    return view.schema_version == execution_csr_schema_version
        && view.structure.schema_version == sparse_structure_identity_schema_version
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
        && view.nnz_count != 0u
        && view.value_size_bytes == sizeof(__half)
        && view.row_offsets != nullptr
        && view.execution_feature_ids != nullptr
        && device_ordinal >= 0;
}

bool same_location(
    execution::device_location lhs,
    execution::device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

operation_status run_csr_fallback(
    const prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    if (prepared.persistent.data == nullptr
        || prepared.persistent.bytes != sizeof(csr_fallback_prepared_state))
        return fail(operation_status_code::execution_failed,
            "CSR fallback prepared state is absent");
    const auto &state = *static_cast<const csr_fallback_prepared_state *>(
        prepared.persistent.data);
    if (state.schema_version != csr_fallback_candidate_schema_version
        || launch.input_count != 1u || launch.output_count != 1u
        || launch.value_count != 1u || launch.values == nullptr
        || launch.inputs[0].kind != execution::operand_kind::dense_tensor
        || launch.outputs[0].kind != execution::operand_kind::dense_tensor)
        return fail(operation_status_code::invalid_launch_bindings,
            "CSR fallback launch arity or state is incompatible");

    const auto &vector = launch.inputs[0].storage.dense;
    const auto &output = launch.outputs[0].storage.dense;
    const execution::value_plane &values = *launch.values[0].plane;
    const execution::relation_structure &structure = launch.structures[0];
    if (!execution::same_axis_identity(structure.source_axis, state.feature_axis)
        || !execution::same_axis_identity(structure.destination_axis, state.row_axis)
        || structure.logical_edge_count != state.projection.nnz_count
        || vector.value_type != execution::numeric_type::f32
        || vector.rank != 1u || vector.shape[0] != state.projection.feature_count
        || vector.stride[0] != 1
        || output.value_type != execution::numeric_type::f32
        || output.rank != 1u || output.shape[0] != state.projection.row_count
        || output.stride[0] != 1
        || values.numeric.storage != execution::numeric_type::f16
        || values.numeric.dequantized != execution::numeric_type::f32
        || values.numeric.accumulation != execution::numeric_type::f32
        || values.layout != execution::value_layout_kind::projection_local_order
        || values.element_count != state.projection.nnz_count
        || values.value_bytes != values.element_count * sizeof(__half)
        || vector.location.residency == execution::residency_kind::host
        || output.location.residency == execution::residency_kind::host
        || values.location.residency == execution::residency_kind::host
        || !same_location(vector.location, output.location)
        || !same_location(vector.location, values.location)
        || vector.location.device_ordinal != state.device_ordinal
        || launch.stream.device_ordinal != state.device_ordinal)
        return fail(operation_status_code::invalid_launch_bindings,
            "CSR fallback order, shape, value, or residency is incompatible");

    const int blocks = static_cast<int>((
        static_cast<std::size_t>(state.projection.row_count)
        + csr_row_threads - 1u) / csr_row_threads);
    csr_spmv_fwd_kernel_<<<blocks, csr_row_threads, 0,
        static_cast<cudaStream_t>(launch.stream.stream)>>>(
            state.projection.row_offsets,
            state.projection.execution_feature_ids,
            static_cast<const __half *>(values.values),
            state.projection.row_count,
            static_cast<const float *>(vector.data),
            static_cast<float *>(output.data));
    if (cudaGetLastError() != cudaSuccess)
        return fail(operation_status_code::execution_failed,
            "existing CSR kernel launch failed");
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
        || prepared->persistent.bytes != sizeof(csr_fallback_prepared_state))
        return fail(operation_status_code::preparation_failed,
            "CSR fallback requires caller-owned prebound state");
    auto *state = static_cast<csr_fallback_prepared_state *>(
        const_cast<void *>(prepared->persistent.data));
    if (state->schema_version != csr_fallback_candidate_schema_version
        || !valid_preconstructed_device_csr(
            state->projection, state->device_ordinal)
        || !execution::valid_axis_identity(state->feature_axis)
        || !execution::valid_axis_identity(state->row_axis)
        || structures.count != 1u || problem.input_count != 1u
        || problem.output_count != 1u
        || problem.logical_work_items != state->projection.nnz_count
        || projection.schema_version != execution_csr_schema_version
        || projection.variant != 1u)
        return fail(operation_status_code::unsupported_problem,
            "CSR fallback preparation metadata is incompatible");

    state->input_contract = {};
    state->input_contract.kind = execution::operand_kind::dense_tensor;
    state->input_contract.rank = 1u;
    state->input_contract.axes[0] = state->feature_axis;
    state->output_contract = {};
    state->output_contract.kind = execution::operand_kind::dense_tensor;
    state->output_contract.rank = 1u;
    state->output_contract.axes[0] = state->row_axis;
    state->output_order = {state->row_axis, state->row_axis,
        execution::order_transition_kind::preserve, 0u, 0u, 0u, 1u, {}, {}};
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
    prepared->binding_contract.output_orders = &state->output_order;
    prepared->binding_contract.output_effects = &state->output_effect;
    prepared->binding_contract.input_count = 1u;
    prepared->binding_contract.output_count = 1u;
    prepared->binding_contract.output_order_count = 1u;
    prepared->binding_contract.structure_count = 1u;
    prepared->binding_contract.output_effect_count = 1u;
    prepared->binding_contract.workspace = {0u, 1u, 0u};
    prepared->run = run_csr_fallback;
    return validate_prepared_operation(*prepared);
}

} // namespace

operation_candidate csr_fallback_candidate() noexcept {
    operation_candidate candidate{};
    candidate.identity = csr_fallback_candidate_id;
    candidate.name = "cellerator-csr-fallback-n1-f16-f32";
    candidate.operation = operation_kind::weighted_relation_reduce;
    candidate.projection = projection_kind::csr;
    candidate.backend = backend_kind::native_direct;
    candidate.capability_flags = candidate_deterministic
        | candidate_persistent_preprocessing;
    candidate.persistent_bytes = sizeof(csr_fallback_prepared_state);
    candidate.transient_bytes = 0u;
    candidate.supports_numeric = supports_numeric;
    candidate.prepare = prepare_impl;
    return candidate;
}

operation_status register_csr_fallback_candidate(
    candidate_registry *registry) noexcept {
    return register_candidate(registry, csr_fallback_candidate());
}

operation_status prepare_csr_fallback_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const execution_csr_view &device_csr,
    std::int32_t device_ordinal,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    csr_fallback_prepared_state *state,
    prepared_operation *prepared) noexcept {
    if (state == nullptr || prepared == nullptr)
        return fail(operation_status_code::invalid_argument,
            "CSR fallback preparation output is null");
    *state = {};
    state->projection = device_csr;
    state->device_ordinal = device_ordinal;
    state->feature_axis = feature_axis;
    state->row_axis = row_axis;
    *prepared = {};
    prepared->persistent = {state, sizeof(*state)};
    const operation_candidate candidate = csr_fallback_candidate();
    return prepare_candidate(candidate, problem, structures, projection,
        numeric, policy, prepared);
}

} // namespace cellerator::compute::math::core
