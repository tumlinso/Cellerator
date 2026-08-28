#include <Cellerator/compute/candidate/row_masked_n1_candidate.hh>

#include <Cellerator/geometry/feature_weighted_row_reduction_cuda.hh>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math::core {
namespace {

operation_status fail(operation_status_code code, const char *message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
}

template<typename T>
constexpr execution::numeric_type execution_numeric_type() noexcept {
    if constexpr (std::is_same<T, __half>::value)
        return execution::numeric_type::f16;
    else if constexpr (std::is_same<T, float>::value)
        return execution::numeric_type::f32;
    else if constexpr (std::is_same<T, double>::value)
        return execution::numeric_type::f64;
#if CELLERATOR_REAL_HAS_CUDA_BF16_HEADER
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
        return execution::numeric_type::bf16;
#endif
    else
        return execution::numeric_type::invalid;
}

constexpr execution::numeric_type sparse_type =
    execution_numeric_type<cellerator::real::storage_t>();
constexpr execution::numeric_type compute_type =
    execution_numeric_type<cellerator::real::compute_t>();
constexpr execution::numeric_type accumulation_type =
    execution_numeric_type<cellerator::real::accum_t>();

bool supports_numeric(const numeric_policy &numeric) noexcept {
    return numeric.sparse_storage == sparse_type
        && numeric.dense_storage == compute_type
        && numeric.output_storage == accumulation_type
        && numeric.multiply == compute_type
        && numeric.accumulation == accumulation_type
        && numeric.scalar == execution::numeric_type::u32
        && numeric.bias == execution::numeric_type::invalid
        && numeric.rounding == rounding_policy::nearest_even
        && numeric.saturation == saturation_policy::none
        && numeric.quantization == quantization_granularity::none;
}

bool valid_device_cpk1(
    const cellpack::persistent_packing_payload_view &payload) noexcept {
    const auto &plan = payload.plan;
    const auto &order = payload.order;
    const auto &tiles = payload.tiles;
    return payload.payload_schema_version
            == cellpack::persistent_packing_payload_schema_version
        && payload.payload_kind == cellpack::persistent_packing_payload_kind
        && payload.payload_identity != 0u
        && payload.image_base != nullptr && payload.image_bytes != 0u
        && plan.semantic_plan_schema_version
            == cellpack::packing_plan_semantic_schema_version
        && plan.geometry_identity_version
            == cellpack::feature_block_geometry_identity_version
        && plan.feature_count == tiles.feature_count
        && plan.feature_block_count == tiles.feature_block_count
        && plan.feature_block_geometry_identity
            == tiles.feature_block_geometry_identity
        && order.order_schema_version == cellpack::local_cell_order_schema_version
        && order.signature_algorithm_version
            == cellpack::local_cell_signature_algorithm_version
        && order.ordering_identity == tiles.ordering_identity
        && order.row_count == tiles.row_count
        && order.row_domain_identity == tiles.row_domain_identity
        && tiles.tile_schema_version == cellpack::warp_tile_schema_version
        && tiles.record_schema_version == cellpack::cell_block_record_schema_version
        && tiles.value_size_bytes == sizeof(cellerator::real::storage_t)
        && plan.feature_block_offsets != nullptr
        && (plan.feature_count == 0u || plan.feature_permutation != nullptr)
        && (order.row_count == 0u || order.row_permutation != nullptr)
        && tiles.tile_block_offsets != nullptr
        && tiles.block_row_entry_offsets != nullptr
        && tiles.row_block_value_offsets != nullptr
        && (tiles.tile_block_count == 0u
            || (tiles.tile_block_ids != nullptr
                && tiles.tile_block_cell_masks != nullptr))
        && (tiles.row_block_entry_count == 0u
            || tiles.row_block_gene_masks != nullptr)
        && (tiles.nnz_count == 0u || tiles.values != nullptr);
}

bool same_location(
    execution::device_location lhs,
    execution::device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

const execution::scalar_binding *find_feature_weight_generation(
    const execution::scalar_bindings &bindings) noexcept {
    for (std::uint32_t index = 0u; index < bindings.count; ++index)
        if (bindings.values[index].binding_id
                == row_masked_n1_feature_weight_generation_binding
            && bindings.values[index].type == execution::numeric_type::u32
            && bindings.values[index].bits != 0u)
            return &bindings.values[index];
    return nullptr;
}

operation_status run_row_masked_n1(
    const prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    if (prepared.persistent.data == nullptr
        || prepared.persistent.bytes != sizeof(row_masked_n1_prepared_state))
        return fail(operation_status_code::execution_failed,
            "row-masked N=1 prepared state is absent");
    const auto &state = *static_cast<const row_masked_n1_prepared_state *>(
        prepared.persistent.data);
    if (state.schema_version != row_masked_n1_candidate_schema_version
        || launch.input_count != 1u || launch.output_count != 1u
        || launch.value_count != 1u || launch.values == nullptr
        || launch.inputs[0].kind != execution::operand_kind::dense_tensor
        || launch.outputs[0].kind != execution::operand_kind::dense_tensor)
        return fail(operation_status_code::invalid_launch_bindings,
            "row-masked N=1 launch arity or state is incompatible");

    const auto &weights = launch.inputs[0].storage.dense;
    const auto &output = launch.outputs[0].storage.dense;
    const execution::value_plane &values = *launch.values[0].plane;
    const execution::relation_structure &structure = launch.structures[0];
    const execution::scalar_binding *weight_generation =
        find_feature_weight_generation(launch.scalars);
    if (weight_generation == nullptr
        || !execution::same_axis_identity(structure.source_axis, state.feature_axis)
        || !execution::same_axis_identity(structure.destination_axis, state.row_axis)
        || weights.value_type != compute_type || weights.rank != 1u
        || weights.shape[0] != state.projection.plan.feature_count
        || weights.stride[0] != 1
        || output.value_type != accumulation_type || output.rank != 1u
        || output.shape[0] != state.projection.tiles.row_count
        || output.stride[0] != 1
        || values.numeric.storage != sparse_type
        || values.numeric.dequantized != compute_type
        || values.numeric.accumulation != accumulation_type
        || values.layout != execution::value_layout_kind::projection_local_order
        || values.element_count != state.projection.tiles.nnz_count
        || values.value_bytes != values.element_count
            * sizeof(cellerator::real::storage_t)
        || weights.location.residency == execution::residency_kind::host
        || output.location.residency == execution::residency_kind::host
        || values.location.residency == execution::residency_kind::host
        || !same_location(weights.location, output.location)
        || !same_location(weights.location, values.location)
        || weights.location.device_ordinal != launch.stream.device_ordinal)
        return fail(operation_status_code::invalid_launch_bindings,
            "row-masked N=1 launch order, shape, value, or residency is incompatible");

    auto direct = cellpack::make_persistent_feature_weighted_row_reduction_view(
        state.projection, weight_generation->bits,
        static_cast<std::size_t>(weights.shape[0]),
        static_cast<const cellerator::real::compute_t *>(weights.data));
    // CPK1 v1 carries initial values for compatibility. Mutable relation values
    // are launch-bound and replace only that pointer; geometry is unchanged.
    direct.tiles.values = values.values;
    cellpack::feature_weighted_row_reduction_result_view result{};
    const cellpack::feature_weighted_row_reduction_buffers buffers{
        static_cast<std::size_t>(output.shape[0]),
        static_cast<cellerator::real::accum_t *>(output.data)};
    const cellpack::validation_result status =
        cellpack::evaluate_feature_weighted_row_reduction_tiles_cuda(
            direct, state.projection.order, buffers,
            static_cast<cudaStream_t>(launch.stream.stream), &result);
    if (!status)
        return fail(operation_status_code::execution_failed,
            "direct CP-BP row-masked N=1 execution failed");
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
        || prepared->persistent.bytes != sizeof(row_masked_n1_prepared_state))
        return fail(operation_status_code::preparation_failed,
            "row-masked N=1 requires caller-owned prebound state");
    auto *state = static_cast<row_masked_n1_prepared_state *>(
        const_cast<void *>(prepared->persistent.data));
    if (state->schema_version != row_masked_n1_candidate_schema_version
        || !valid_device_cpk1(state->projection)
        || !execution::valid_axis_identity(state->feature_axis)
        || !execution::valid_axis_identity(state->row_axis)
        || structures.count != 1u || problem.input_count != 1u
        || problem.output_count != 1u
        || problem.logical_work_items != state->projection.tiles.nnz_count
        || projection.schema_version
            != cellpack::persistent_packing_payload_schema_version
        || projection.variant != 1u)
        return fail(operation_status_code::unsupported_problem,
            "row-masked N=1 preparation metadata is incompatible");

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
    prepared->run = run_row_masked_n1;
    return validate_prepared_operation(*prepared);
}

} // namespace

operation_candidate row_masked_n1_candidate() noexcept {
    operation_candidate candidate{};
    candidate.identity = row_masked_n1_candidate_id;
    candidate.name = "cpbp-cpk1-row-masked-n1";
    candidate.operation = operation_kind::weighted_relation_reduce;
    candidate.projection = projection_kind::native_row_masked;
    candidate.backend = backend_kind::native_direct;
    candidate.capability_flags = candidate_deterministic
        | candidate_graph_capture | candidate_persistent_preprocessing;
    candidate.persistent_bytes = sizeof(row_masked_n1_prepared_state);
    candidate.transient_bytes = 0u;
    candidate.supports_numeric = supports_numeric;
    candidate.prepare = prepare_impl;
    return candidate;
}

operation_status register_row_masked_n1_candidate(
    candidate_registry *registry) noexcept {
    return register_candidate(registry, row_masked_n1_candidate());
}

operation_status prepare_row_masked_n1_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const cellpack::persistent_packing_payload_view &device_cpk1,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    row_masked_n1_prepared_state *state,
    prepared_operation *prepared) noexcept {
    if (state == nullptr || prepared == nullptr)
        return fail(operation_status_code::invalid_argument,
            "row-masked N=1 preparation output is null");
    *state = {};
    state->projection = device_cpk1;
    state->feature_axis = feature_axis;
    state->row_axis = row_axis;
    *prepared = {};
    prepared->persistent = {state, sizeof(*state)};
    const operation_candidate candidate = row_masked_n1_candidate();
    return prepare_candidate(candidate, problem, structures, projection,
        numeric, policy, prepared);
}

} // namespace cellerator::compute::math::core
