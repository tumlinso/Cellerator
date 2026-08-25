/*
CE-ARCH-75 correctness activation, 2026-08-25:
No maintained CUDA sparse library consumes the FMP1 masked feature-major ABI,
so direct native execution requires this bounded one-warp kernel. CE-ARCH-75
adds correctness evidence on the repository V100 for irregular/partial tiles
at N=1 and N=16 against the independent SpMM referee. It makes no performance
selection claim; CE-ARCH-76 owns the row-masked/feature-major/CSR comparison,
exact commands, measurements, and retention decision for N=1..16.
*/

#include <Cellerator/compute/math/operation_core/feature_major_small_n_candidate.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::math::core {
namespace {

operation_status fail(operation_status_code code, const char *message) noexcept {
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

bool same_location(
    execution::device_location lhs,
    execution::device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

bool valid_device_projection(
    const feature_major_projection_view &view,
    std::int32_t device_ordinal) noexcept {
    const auto &header = view.header;
    return header.schema_version == feature_major_projection_schema_version
        && header.payload_kind == feature_major_projection_payload_kind
        && header.header_bytes == sizeof(header)
        && header.alignment == feature_major_projection_alignment
        && header.payload_bytes != 0u
        && execution::valid_identity(header.structure_identity)
        && header.structure_epoch != 0u
        && execution::valid_identity(header.projection_identity)
        && execution::valid_handle(view.runtime_structure)
        && execution::valid_handle(view.runtime_projection)
        && view.payload_base != nullptr
        && header.feature_block_geometry_identity != 0u
        && header.ordering_identity != 0u
        && header.row_domain_identity != 0u
        && header.feature_axis_fingerprint != 0u
        && header.feature_axis_fingerprint_version != 0u
        && header.row_count != 0u && header.feature_count != 0u
        && header.nnz_count != 0u
        && header.tile_row_width != 0u && header.tile_row_width <= 32u
        && header.tile_count == header.row_count / header.tile_row_width
            + (header.row_count % header.tile_row_width != 0u ? 1u : 0u)
        && header.value_size_bytes == sizeof(__half)
        && view.tile_feature_offsets != nullptr
        && view.execution_feature_ids != nullptr
        && view.participating_row_masks != nullptr
        && view.feature_value_offsets != nullptr
        && view.source_value_positions != nullptr
        && device_ordinal >= 0;
}

__global__ void feature_major_small_n_kernel(
    feature_major_projection_view projection,
    const __half *values,
    const float *dense_rhs,
    std::uint32_t dense_width,
    float *output) {
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t tile = blockIdx.x;
    if (lane >= 32u || tile >= projection.header.tile_count) return;

    float accumulators[feature_major_small_n_maximum]{};
    __shared__ float dense_feature[feature_major_small_n_maximum];
    const std::uint32_t record_begin = projection.tile_feature_offsets[tile];
    const std::uint32_t record_end = projection.tile_feature_offsets[tile + 1u];
    for (std::uint32_t record = record_begin;
         record < record_end; ++record) {
        const std::uint32_t feature = projection.execution_feature_ids[record];
        if (lane < dense_width)
            dense_feature[lane] = dense_rhs[
                static_cast<std::size_t>(feature) * dense_width + lane];
        __syncwarp();
        const std::uint32_t row_mask =
            projection.participating_row_masks[record];
        if ((row_mask & (1u << lane)) != 0u) {
            const std::uint32_t lower_rows = lane == 0u
                ? 0u : row_mask & ((1u << lane) - 1u);
            const std::uint32_t value =
                projection.feature_value_offsets[record]
                + static_cast<std::uint32_t>(__popc(lower_rows));
            const float sparse = __half2float(values[value]);
            #pragma unroll
            for (std::uint32_t column = 0u;
                 column < feature_major_small_n_maximum; ++column)
                if (column < dense_width)
                    accumulators[column] += sparse * dense_feature[column];
        }
        __syncwarp();
    }
    const std::uint32_t row = tile * projection.header.tile_row_width + lane;
    if (lane < projection.header.tile_row_width
        && row < projection.header.row_count) {
        #pragma unroll
        for (std::uint32_t column = 0u;
             column < feature_major_small_n_maximum; ++column)
            if (column < dense_width)
                output[static_cast<std::size_t>(row) * dense_width + column]
                    = accumulators[column];
    }
}

operation_status run_feature_major_small_n(
    const prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    if (prepared.persistent.data == nullptr
        || prepared.persistent.bytes
            != sizeof(feature_major_small_n_prepared_state)) {
        return fail(operation_status_code::execution_failed,
            "feature-major small-N prepared state is absent");
    }
    const auto &state = *static_cast<
        const feature_major_small_n_prepared_state *>(prepared.persistent.data);
    if (state.schema_version != feature_major_small_n_candidate_schema_version
        || launch.input_count != 1u || launch.output_count != 1u
        || launch.value_count != 1u || launch.values == nullptr
        || launch.inputs[0].kind != execution::operand_kind::dense_tensor
        || launch.outputs[0].kind != execution::operand_kind::dense_tensor) {
        return fail(operation_status_code::invalid_launch_bindings,
            "feature-major small-N launch arity or state is incompatible");
    }
    const auto &rhs = launch.inputs[0].storage.dense;
    const auto &output = launch.outputs[0].storage.dense;
    const execution::value_plane &values = *launch.values[0].plane;
    const execution::relation_structure &structure = launch.structures[0];
    const auto &projection = state.projection;
    if (!execution::same_axis_identity(structure.source_axis, state.feature_axis)
        || !execution::same_axis_identity(
            structure.destination_axis, state.row_axis)
        || !execution::same_handle(
            structure.identity, projection.runtime_structure)
        || structure.epoch.value != projection.header.structure_epoch
        || structure.logical_edge_count != projection.header.nnz_count
        || rhs.value_type != execution::numeric_type::f32 || rhs.rank != 2u
        || rhs.shape[0] != projection.header.feature_count
        || rhs.shape[1] != state.dense_width
        || rhs.stride[0] != static_cast<std::int64_t>(state.dense_width)
        || rhs.stride[1] != 1
        || output.value_type != execution::numeric_type::f32
        || output.rank != 2u
        || output.shape[0] != projection.header.row_count
        || output.shape[1] != state.dense_width
        || output.stride[0] != static_cast<std::int64_t>(state.dense_width)
        || output.stride[1] != 1
        || values.numeric.storage != execution::numeric_type::f16
        || values.numeric.dequantized != execution::numeric_type::f32
        || values.numeric.accumulation != execution::numeric_type::f32
        || values.layout != execution::value_layout_kind::projection_local_order
        || values.element_count != projection.header.nnz_count
        || values.value_bytes != values.element_count * sizeof(__half)
        || rhs.location.residency == execution::residency_kind::host
        || output.location.residency == execution::residency_kind::host
        || values.location.residency == execution::residency_kind::host
        || !same_location(rhs.location, output.location)
        || !same_location(rhs.location, values.location)
        || rhs.location.device_ordinal != state.device_ordinal
        || launch.stream.device_ordinal != state.device_ordinal) {
        return fail(operation_status_code::invalid_launch_bindings,
            "feature-major small-N order, shape, value, or residency is incompatible");
    }
    feature_major_small_n_kernel<<<projection.header.tile_count, 32u, 0u,
        static_cast<cudaStream_t>(launch.stream.stream)>>>(projection,
        static_cast<const __half *>(values.values),
        static_cast<const float *>(rhs.data), state.dense_width,
        static_cast<float *>(output.data));
    if (cudaPeekAtLastError() != cudaSuccess) {
        return fail(operation_status_code::execution_failed,
            "feature-major small-N kernel launch failed");
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
        || prepared->persistent.bytes
            != sizeof(feature_major_small_n_prepared_state)) {
        return fail(operation_status_code::preparation_failed,
            "feature-major small-N requires caller-owned prebound state");
    }
    auto *state = static_cast<feature_major_small_n_prepared_state *>(
        const_cast<void *>(prepared->persistent.data));
    const auto &view = state->projection;
    const auto &header = view.header;
    const bool work_overflows = state->dense_width != 0u
        && header.nnz_count > std::numeric_limits<std::uint64_t>::max()
            / state->dense_width;
    if (state->schema_version != feature_major_small_n_candidate_schema_version
        || !valid_device_projection(view, state->device_ordinal)
        || !execution::valid_axis_identity(state->feature_axis)
        || !execution::valid_axis_identity(state->row_axis)
        || !execution::valid_axis_identity(state->dense_column_axis)
        || state->dense_width < feature_major_small_n_minimum
        || state->dense_width > feature_major_small_n_maximum
        || structures.count != 1u || problem.input_count != 1u
        || problem.output_count != 1u || work_overflows
        || problem.logical_work_items
            != static_cast<std::uint64_t>(header.nnz_count) * state->dense_width
        || !execution::same_identity(
            structures.structures[0].persistent, header.structure_identity)
        || !execution::same_handle(
            structures.structures[0].runtime, view.runtime_structure)
        || structures.structures[0].epoch.value != header.structure_epoch
        || !execution::same_identity(
            projection.persistent, header.projection_identity)
        || !execution::same_handle(projection.runtime, view.runtime_projection)
        || projection.schema_version != feature_major_projection_schema_version
        || projection.variant != feature_major_projection_variant) {
        return fail(operation_status_code::unsupported_problem,
            "feature-major small-N preparation metadata or N is incompatible");
    }

    state->input_contract = {};
    state->input_contract.kind = execution::operand_kind::dense_tensor;
    state->input_contract.rank = 2u;
    state->input_contract.axes[0] = state->feature_axis;
    state->input_contract.axes[1] = state->dense_column_axis;
    state->output_contract = {};
    state->output_contract.kind = execution::operand_kind::dense_tensor;
    state->output_contract.rank = 2u;
    state->output_contract.axes[0] = state->row_axis;
    state->output_contract.axes[1] = state->dense_column_axis;
    state->output_orders[0] = {state->row_axis, state->row_axis,
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
    prepared->run = run_feature_major_small_n;
    return validate_prepared_operation(*prepared);
}

} // namespace

operation_candidate feature_major_small_n_candidate() noexcept {
    operation_candidate candidate{};
    candidate.identity = feature_major_small_n_candidate_id;
    candidate.name = "cpbp-feature-major-small-n-f16-f32";
    candidate.operation = operation_kind::sparse_dense_multiply;
    candidate.projection = projection_kind::native_feature_major;
    candidate.backend = backend_kind::native_direct;
    candidate.capability_flags = candidate_deterministic
        | candidate_graph_capture | candidate_persistent_preprocessing;
    candidate.persistent_bytes = sizeof(feature_major_small_n_prepared_state);
    candidate.transient_bytes = 0u;
    candidate.supports_numeric = supports_numeric;
    candidate.prepare = prepare_impl;
    return candidate;
}

operation_status register_feature_major_small_n_candidate(
    candidate_registry *registry) noexcept {
    return register_candidate(registry, feature_major_small_n_candidate());
}

operation_status prepare_feature_major_small_n_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const feature_major_projection_view &device_projection,
    std::int32_t device_ordinal,
    std::uint32_t dense_width,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    execution::axis_identity dense_column_axis,
    feature_major_small_n_prepared_state *state,
    prepared_operation *prepared) noexcept {
    if (state == nullptr || prepared == nullptr) {
        return fail(operation_status_code::invalid_argument,
            "feature-major small-N preparation output is null");
    }
    *state = {};
    state->device_ordinal = device_ordinal;
    state->dense_width = dense_width;
    state->projection = device_projection;
    state->feature_axis = feature_axis;
    state->row_axis = row_axis;
    state->dense_column_axis = dense_column_axis;
    *prepared = {};
    prepared->persistent = {state, sizeof(*state)};
    const operation_candidate candidate = feature_major_small_n_candidate();
    return prepare_candidate(candidate, problem, structures, projection,
        numeric, policy, prepared);
}

} // namespace cellerator::compute::math::core
