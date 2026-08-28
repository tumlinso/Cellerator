#include <Cellerator/compute/candidate/tensor_core/v100_dense_fragment_candidate.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::math::tensor_core {
namespace {

namespace wmma = nvcuda::wmma;

core::operation_status fail(
    core::operation_status_code code, const char *message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
}

bool supports_numeric(const core::numeric_policy &numeric) noexcept {
    return numeric.sparse_storage == execution::numeric_type::f16
        && numeric.dense_storage == execution::numeric_type::f16
        && numeric.output_storage == execution::numeric_type::f32
        && numeric.multiply == execution::numeric_type::f16
        && numeric.accumulation == execution::numeric_type::f32
        && numeric.scalar == execution::numeric_type::f32
        && numeric.bias == execution::numeric_type::invalid
        && numeric.rounding == core::rounding_policy::nearest_even
        && numeric.saturation == core::saturation_policy::none
        && numeric.quantization == core::quantization_granularity::none;
}

bool same_location(
    execution::device_location lhs,
    execution::device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

bool valid_projection(
    const v100_dense_fragment_projection_view &view) noexcept {
    constexpr std::uint64_t slots_per_fragment =
        v100_dense_fragment_extent * v100_dense_fragment_extent;
    return view.schema_version == v100_dense_fragment_schema_version
        && view.variant == v100_dense_fragment_variant
        && view.architecture_class == 70u
        && view.fragment_count != 0u
        && view.destination_count >= v100_dense_fragment_extent
        && view.source_count >= v100_dense_fragment_extent
        && view.logical_edge_count != 0u
        && view.packed_slot_count
            == static_cast<std::uint64_t>(view.fragment_count)
                * slots_per_fragment
        && execution::valid_identity(view.persistent_structure)
        && execution::valid_handle(view.runtime_structure)
        && view.structure_epoch.value != 0u
        && execution::valid_identity(view.persistent_projection)
        && execution::valid_handle(view.runtime_projection)
        && view.fragment_destination_bases != nullptr
        && view.fragment_source_bases != nullptr;
}

__global__ void v100_dense_fragment_kernel(
    v100_dense_fragment_projection_view projection,
    const __half *fragment_values,
    const __half *dense_rhs,
    std::uint32_t dense_width,
    float *output) {
    const std::uint32_t fragment_index = blockIdx.x;
    const std::uint32_t column_tile = blockIdx.y;
    if (fragment_index >= projection.fragment_count || threadIdx.x >= 32u)
        return;

    const std::uint32_t destination_base =
        projection.fragment_destination_bases[fragment_index];
    const std::uint32_t source_base =
        projection.fragment_source_bases[fragment_index];
    const std::uint32_t column_base =
        column_tile * v100_dense_fragment_extent;
    if (destination_base + v100_dense_fragment_extent
            > projection.destination_count
        || source_base + v100_dense_fragment_extent > projection.source_count
        || column_base + v100_dense_fragment_extent > dense_width)
        return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
        wmma::row_major> relation;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
        wmma::row_major> rhs;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    wmma::load_matrix_sync(relation,
        fragment_values + static_cast<std::size_t>(fragment_index) * 256u,
        16u);
    wmma::load_matrix_sync(rhs,
        dense_rhs + static_cast<std::size_t>(source_base) * dense_width
            + column_base,
        dense_width);
    wmma::mma_sync(accumulator, relation, rhs, accumulator);

    __shared__ float tile[256];
    wmma::store_matrix_sync(tile, accumulator, 16u, wmma::mem_row_major);
    __syncwarp();
    for (std::uint32_t slot = threadIdx.x; slot < 256u; slot += 32u) {
        const std::uint32_t local_row = slot / 16u;
        const std::uint32_t local_column = slot % 16u;
        atomicAdd(output
                + static_cast<std::size_t>(destination_base + local_row)
                    * dense_width + column_base + local_column,
            tile[slot]);
    }
}

core::operation_status run_dense_fragment(
    const core::prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    if (prepared.persistent.data == nullptr
        || prepared.persistent.bytes
            != sizeof(v100_dense_fragment_prepared_state))
        return fail(core::operation_status_code::execution_failed,
            "V100 dense-fragment prepared state is absent");
    const auto &state = *static_cast<
        const v100_dense_fragment_prepared_state *>(prepared.persistent.data);
    if (state.schema_version != v100_dense_fragment_schema_version
        || launch.input_count != 1u || launch.output_count != 1u
        || launch.value_count != 1u || launch.values == nullptr
        || launch.inputs[0].kind != execution::operand_kind::dense_tensor
        || launch.outputs[0].kind != execution::operand_kind::dense_tensor)
        return fail(core::operation_status_code::invalid_launch_bindings,
            "V100 dense-fragment launch arity or state is incompatible");

    const auto &rhs = launch.inputs[0].storage.dense;
    const auto &output = launch.outputs[0].storage.dense;
    const execution::value_plane &values = *launch.values[0].plane;
    const execution::relation_structure &structure = launch.structures[0];
    const auto &projection = state.projection;
    if (!execution::same_axis_identity(structure.source_axis, state.source_axis)
        || !execution::same_axis_identity(
            structure.destination_axis, state.destination_axis)
        || !execution::same_handle(
            structure.identity, projection.runtime_structure)
        || structure.epoch.value != projection.structure_epoch.value
        || structure.logical_edge_count != projection.logical_edge_count
        || rhs.value_type != execution::numeric_type::f16 || rhs.rank != 2u
        || rhs.shape[0] != projection.source_count
        || rhs.shape[1] != state.dense_width
        || rhs.stride[0] != static_cast<std::int64_t>(state.dense_width)
        || rhs.stride[1] != 1
        || output.value_type != execution::numeric_type::f32
        || output.rank != 2u
        || output.shape[0] != projection.destination_count
        || output.shape[1] != state.dense_width
        || output.stride[0] != static_cast<std::int64_t>(state.dense_width)
        || output.stride[1] != 1
        || values.numeric.storage != execution::numeric_type::f16
        || values.numeric.dequantized != execution::numeric_type::f32
        || values.numeric.accumulation != execution::numeric_type::f32
        || values.layout != execution::value_layout_kind::projection_local_order
        || values.element_count != projection.packed_slot_count
        || values.value_bytes != values.element_count * sizeof(__half)
        || rhs.location.residency == execution::residency_kind::host
        || output.location.residency == execution::residency_kind::host
        || values.location.residency == execution::residency_kind::host
        || !same_location(rhs.location, output.location)
        || !same_location(rhs.location, values.location)
        || rhs.location.device_ordinal != state.device_ordinal
        || launch.stream.device_ordinal != state.device_ordinal)
        return fail(core::operation_status_code::invalid_launch_bindings,
            "V100 dense-fragment identity, shape, value, or residency is incompatible");

    const dim3 grid(projection.fragment_count,
        state.dense_width / v100_dense_fragment_extent);
    v100_dense_fragment_kernel<<<grid, 32u, 0u,
        static_cast<cudaStream_t>(launch.stream.stream)>>>(projection,
        static_cast<const __half *>(values.values),
        static_cast<const __half *>(rhs.data), state.dense_width,
        static_cast<float *>(output.data));
    if (cudaPeekAtLastError() != cudaSuccess)
        return fail(core::operation_status_code::execution_failed,
            "V100 dense-fragment kernel launch failed");
    return {};
}

core::operation_status prepare_impl(
    const core::operation_candidate &candidate,
    const core::operation_problem &problem,
    const core::structure_set_key &structures,
    const core::projection_key &projection,
    const core::numeric_policy &numeric,
    const core::prepare_policy &,
    core::prepared_operation *prepared) noexcept {
    if (prepared == nullptr || prepared->persistent.data == nullptr
        || prepared->persistent.bytes
            != sizeof(v100_dense_fragment_prepared_state))
        return fail(core::operation_status_code::preparation_failed,
            "V100 dense-fragment requires caller-owned prebound state");
    auto *state = static_cast<v100_dense_fragment_prepared_state *>(
        const_cast<void *>(prepared->persistent.data));
    const auto &view = state->projection;
    if (!valid_projection(view)
        || state->dense_width == 0u
        || state->dense_width % v100_dense_fragment_extent != 0u
        || state->dense_width > 64u
        || problem.kind != core::operation_kind::sparse_dense_multiply
        || problem.input_count != 1u || problem.output_count != 1u
        || problem.logical_work_items
            != view.logical_edge_count * state->dense_width
        || structures.count != 1u
        || !execution::same_identity(
            structures.structures[0].persistent,
            view.persistent_structure)
        || !execution::same_handle(
            structures.structures[0].runtime, view.runtime_structure)
        || structures.structures[0].epoch.value
            != view.structure_epoch.value
        || !execution::same_identity(
            projection.persistent, view.persistent_projection)
        || !execution::same_handle(
            projection.runtime, view.runtime_projection)
        || projection.kind != core::projection_kind::dense_fragment
        || projection.schema_version != v100_dense_fragment_schema_version
        || projection.variant != v100_dense_fragment_variant)
        return fail(core::operation_status_code::unsupported_problem,
            "V100 dense-fragment problem or projection is incompatible");

    state->input_contract = {};
    state->input_contract.kind = execution::operand_kind::dense_tensor;
    state->input_contract.rank = 2u;
    state->input_contract.axes[0] = state->source_axis;
    state->input_contract.axes[1] = state->dense_column_axis;
    state->output_contract = {};
    state->output_contract.kind = execution::operand_kind::dense_tensor;
    state->output_contract.rank = 2u;
    state->output_contract.axes[0] = state->destination_axis;
    state->output_contract.axes[1] = state->dense_column_axis;
    state->output_orders[0] = {state->destination_axis,
        state->destination_axis, execution::order_transition_kind::preserve,
        0u, 0u, 0u, 1u, {}, {}};
    state->output_orders[1] = {state->dense_column_axis,
        state->dense_column_axis, execution::order_transition_kind::preserve,
        1u, 0u, 0u, 1u, {}, {}};
    state->output_effect = {execution::output_update_kind::accumulate,
        true, false, 0u, execution::invalid_scalar_binding_id,
        execution::invalid_scalar_binding_id};

    const core::persistent_kernel_state persistent = prepared->persistent;
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
    prepared->run = run_dense_fragment;
    return core::validate_prepared_operation(*prepared);
}

} // namespace

core::operation_candidate v100_dense_fragment_candidate() noexcept {
    core::operation_candidate candidate{};
    candidate.identity = v100_dense_fragment_candidate_id;
    candidate.name = "v100-wmma-dense-fragment-f16-f32";
    candidate.operation = core::operation_kind::sparse_dense_multiply;
    candidate.projection = core::projection_kind::dense_fragment;
    candidate.backend = core::backend_kind::composed;
    candidate.capability_flags = core::candidate_graph_capture
        | core::candidate_persistent_preprocessing
        | core::candidate_composed_epilogue;
    candidate.persistent_bytes = sizeof(v100_dense_fragment_prepared_state);
    candidate.transient_bytes = 0u;
    candidate.supports_numeric = supports_numeric;
    candidate.prepare = prepare_impl;
    return candidate;
}

core::operation_status register_v100_dense_fragment_candidate(
    core::candidate_registry *registry) noexcept {
    return core::register_candidate(registry,
        v100_dense_fragment_candidate());
}

core::operation_status prepare_v100_dense_fragment_operation(
    const core::operation_problem &problem,
    const core::structure_set_key &structures,
    const core::projection_key &projection,
    const core::numeric_policy &numeric,
    const core::prepare_policy &policy,
    const v100_dense_fragment_projection_view &device_projection,
    std::int32_t device_ordinal,
    std::uint32_t dense_width,
    execution::axis_identity source_axis,
    execution::axis_identity destination_axis,
    execution::axis_identity dense_column_axis,
    v100_dense_fragment_prepared_state *state,
    core::prepared_operation *prepared) noexcept {
    if (state == nullptr || prepared == nullptr || device_ordinal < 0)
        return fail(core::operation_status_code::invalid_argument,
            "V100 dense-fragment typed preparation arguments are incomplete");
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device_ordinal) != cudaSuccess
        || properties.major != 7 || properties.minor != 0)
        return fail(core::operation_status_code::capability_rejected,
            "V100 dense-fragment candidate requires sm_70");

    *state = {};
    state->device_ordinal = device_ordinal;
    state->dense_width = dense_width;
    state->projection = device_projection;
    state->source_axis = source_axis;
    state->destination_axis = destination_axis;
    state->dense_column_axis = dense_column_axis;
    prepared->persistent = {state, sizeof(*state)};
    return core::prepare_candidate(v100_dense_fragment_candidate(), problem,
        structures, projection, numeric, policy, prepared);
}

} // namespace cellerator::compute::math::tensor_core
