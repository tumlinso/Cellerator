#include <Cellerator/execution/sequence/regulatory.cuh>

#include <Cellerator/execution/validation.hh>

#include <cuda_runtime.h>

#include <stdint.h>

namespace cellerator::execution::sequence {
namespace {

constexpr operation_core::stable_id sequence_operation_id{
    0x7365712d72656731ULL, 0x62617365706c616eULL};
constexpr operation_core::stable_id materialized_kernel_id{
    0x6d61736b2d726567ULL, 0x7365712d76310001ULL};
constexpr operation_core::stable_id relation_materialized_kernel_id{
    0x72656c2d72656731ULL, 0x7365712d76330001ULL};
constexpr operation_core::stable_id fused_kernel_id{
    0x667573652d726567ULL, 0x7365712d76310001ULL};

operation_core::operation_status fail(
    operation_core::operation_status_code code,
    const char *message) noexcept {
    return {code, execution::binding_validation_code::ok, message};
}

bool same_location(
    const execution::device_location &left,
    const execution::device_location &right) noexcept {
    return left.residency == right.residency
        && left.device_ordinal == right.device_ordinal
        && left.address_space == right.address_space;
}

bool same_cache_key(
    const predicate_materialization_key &left,
    const predicate_materialization_key &right) noexcept {
    return left.sequence_generation.value == right.sequence_generation.value
        && left.predicate_semantic_hash == right.predicate_semantic_hash
        && ::cellerator::structure::same_identity(
            left.coordinate_structure, right.coordinate_structure)
        && ::cellerator::structure::same_epoch(left.coordinate_epoch, right.coordinate_epoch)
        && ::cellerator::structure::same_identity(left.coordinate_order, right.coordinate_order)
        && left.predicate_id == right.predicate_id
        && left.output_flags == right.output_flags;
}

bool same_relation_cache_key(
    const regulatory_relation_materialization_key &left,
    const regulatory_relation_materialization_key &right) noexcept {
    return left.sequence_generation.value == right.sequence_generation.value
        && left.predicate_semantic_hash == right.predicate_semantic_hash
        && ::cellerator::structure::same_identity(
            left.coordinate_structure, right.coordinate_structure)
        && ::cellerator::structure::same_epoch(left.coordinate_epoch, right.coordinate_epoch)
        && ::cellerator::structure::same_identity(left.coordinate_order, right.coordinate_order)
        && ::cellerator::structure::same_identity(
            left.regulatory_projection, right.regulatory_projection)
        && left.predicate_id == right.predicate_id
        && left.output_flags == right.output_flags;
}

bool plans_match(
    const ops::sequence::prepared_predicate_plan &left,
    const ops::sequence::prepared_predicate_plan &right) noexcept {
    if (left.version != right.version || left.node_count != right.node_count
        || left.live_node_count != right.live_node_count
        || left.output_count != right.output_count
        || left.lookbehind != right.lookbehind
        || left.lookahead != right.lookahead
        || left.scratch_mask_words_per_window
            != right.scratch_mask_words_per_window
        || left.family != right.family
        || left.semantic_hash != right.semantic_hash)
        return false;
    for (uint16_t index = 0u;
         index < ops::sequence::sequence_program_max_nodes; ++index)
        if (left.original_to_prepared[index]
            != right.original_to_prepared[index]) return false;
    return true;
}

bool supported_exact_mask_program(
    const ops::sequence::sequence_predicate_program &program,
    const ops::sequence::prepared_predicate_plan &plan) noexcept {
    if (program.node_count != 1u || program.output_count != 1u
        || program.exact_motif_count != 1u || program.allowed_motif_count != 0u
        || plan.family != ops::sequence::predicate_lowering_family::exact_scan)
        return false;
    const ops::sequence::predicate_instruction &node = program.nodes[0];
    const ops::sequence::predicate_output &output = program.outputs[0];
    return node.opcode == ops::sequence::predicate_opcode::exact_motif
        && node.result_kind == ops::sequence::predicate_value_kind::mask
        && node.input_a == ops::sequence::sequence_program_no_input
        && node.input_b == ops::sequence::sequence_program_no_input
        && node.immediate == 0u && output.node == 0u
        && output.mode == ops::sequence::sequence_output_mode::mask
        && (output.flags & ops::sequence::sequence_event_reverse_strand) == 0u;
}

__device__ __forceinline__ bool predicate_matches(
    tensor::sequence::BitPlanesView input,
    const execution::sequence_domain &domain,
    uint32_t anchor,
    ops::sequence::motif32_exact motif) noexcept {
    return anchor >= domain.owned_begin && anchor < domain.owned_end
        && ops::sequence::match_exact_at(input, anchor, motif);
}

__device__ __forceinline__ uint32_t find_regulatory_element(
    regulatory_projection_view projection,
    uint32_t anchor,
    uint16_t predicate_id) noexcept {
    uint32_t lower = 0u, upper = projection.interval_count;
    while (lower < upper) {
        const uint32_t middle = lower + (upper - lower) / 2u;
        if (projection.intervals[middle].begin <= anchor)
            lower = middle + 1u;
        else
            upper = middle;
    }
    if (lower == 0u) return 0xffffffffu;
    const regulatory_interval &interval = projection.intervals[lower - 1u];
    return anchor < interval.end && interval.predicate_id == predicate_id
        ? interval.regulatory_element : 0xffffffffu;
}

__device__ __forceinline__ void accumulate_element(
    regulatory_projection_view projection,
    uint32_t element,
    const float *weights,
    float *gene_state) noexcept {
    if (element >= projection.regulatory_element_count) return;
    const uint32_t begin = projection.element_offsets[element];
    const uint32_t end = projection.element_offsets[element + 1u];
    for (uint32_t edge = begin; edge < end; ++edge)
        atomicAdd(gene_state + projection.gene_indices[edge], weights[edge]);
}

__global__ void materialize_predicate_mask_kernel(
    tensor::sequence::BitPlanesView input,
    execution::sequence_domain domain,
    ops::sequence::motif32_exact motif,
    uint32_t *mask) {
    const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= input.word_count) return;
    const uint32_t first = word << 5u;
    uint32_t result = 0u;
    for (uint32_t lane = 0u; lane < 32u; ++lane) {
        const uint32_t anchor = first + lane;
        if (anchor < input.base_count
            && predicate_matches(input, domain, anchor, motif))
            result |= 1u << lane;
    }
    mask[word] = result;
}

__global__ void accumulate_materialized_mask_kernel(
    const uint32_t *mask,
    uint32_t base_count,
    regulatory_projection_view projection,
    uint16_t predicate_id,
    const float *weights,
    float *gene_state) {
    const uint32_t anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= base_count
        || ((mask[anchor >> 5u] >> (anchor & 31u)) & 1u) == 0u) return;
    const uint32_t element = find_regulatory_element(
        projection, anchor, predicate_id);
    if (element != 0xffffffffu)
        accumulate_element(projection, element, weights, gene_state);
}

__global__ void materialize_regulatory_relation_kernel(
    tensor::sequence::BitPlanesView input,
    execution::sequence_domain domain,
    ops::sequence::motif32_exact motif,
    regulatory_projection_view projection,
    uint16_t predicate_id,
    uint32_t *elements) {
    const uint32_t anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= input.base_count) return;
    elements[anchor] = predicate_matches(input, domain, anchor, motif)
        ? find_regulatory_element(projection, anchor, predicate_id)
        : 0xffffffffu;
}

__global__ void accumulate_materialized_relation_kernel(
    const uint32_t *elements,
    uint32_t base_count,
    regulatory_projection_view projection,
    const float *weights,
    float *gene_state) {
    const uint32_t anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= base_count) return;
    const uint32_t element = elements[anchor];
    if (element != 0xffffffffu)
        accumulate_element(projection, element, weights, gene_state);
}

__global__ void fused_predicate_accumulate_kernel(
    tensor::sequence::BitPlanesView input,
    execution::sequence_domain domain,
    ops::sequence::motif32_exact motif,
    regulatory_projection_view projection,
    uint16_t predicate_id,
    const float *weights,
    float *gene_state) {
    const uint32_t anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= input.base_count
        || !predicate_matches(input, domain, anchor, motif)) return;
    const uint32_t element = find_regulatory_element(
        projection, anchor, predicate_id);
    if (element != 0xffffffffu)
        accumulate_element(projection, element, weights, gene_state);
}

struct validated_sequence_launch {
    const prepared_sequence_state *state = nullptr;
    execution::bit_plane_view input{};
    execution::dense_tensor_view output{};
    const execution::value_plane *values = nullptr;
    uint32_t *materialized = nullptr;
};

operation_core::operation_status validate_sequence_launch(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch,
    validated_sequence_launch *validated) noexcept {
    if (validated == nullptr || prepared.persistent.data == nullptr
        || prepared.persistent.bytes != sizeof(prepared_sequence_state)
        || launch.inputs == nullptr || launch.outputs == nullptr
        || launch.values == nullptr)
        return fail(operation_core::operation_status_code::execution_failed,
            "prepared sequence state or launch arrays are absent");
    const auto &state = *static_cast<const prepared_sequence_state *>(
        prepared.persistent.data);
    if (state.schema_version != sequence_regulatory_schema_version
        || launch.input_count != 1u || launch.output_count == 0u
        || launch.value_count != 1u || launch.values[0].plane == nullptr
        || launch.inputs[0].kind != execution::operand_kind::bit_plane
        || launch.outputs[0].kind != execution::operand_kind::dense_tensor)
        return fail(operation_core::operation_status_code::invalid_launch_bindings,
            "sequence launch has wrong state or operand kinds");
    const execution::bit_plane_view input = launch.inputs[0].storage.bits;
    const execution::dense_tensor_view output = launch.outputs[0].storage.dense;
    const execution::value_plane &values = *launch.values[0].plane;
    if (input.base_count != state.source_domain.local_base_count
        || prepared.structures.count != 2u
        || !execution::same_structure_handle(
            values.structure, prepared.structures.structures[1].runtime)
        || output.value_type != execution::numeric_type::f32
        || output.rank != 1u || output.shape[0] != state.regulatory.gene_count
        || output.stride[0] != 1
        || values.numeric.storage != execution::numeric_type::f32
        || values.element_count != state.regulatory.edge_count
        || input.location.residency == execution::residency_kind::host
        || output.location.residency == execution::residency_kind::host
        || !same_location(input.location, output.location)
        || input.location.device_ordinal != launch.stream.device_ordinal
        || input.location.device_ordinal
            != state.regulatory.location.device_ordinal
        || values.location.device_ordinal != input.location.device_ordinal)
        return fail(operation_core::operation_status_code::invalid_launch_bindings,
            "sequence launch residency, shape, or values are incompatible");

    uint32_t *materialized = nullptr;
    if (state.strategy == sequence_strategy::materialize_mask
        || state.strategy == sequence_strategy::materialize_relation) {
        if (launch.output_count != 2u
            || launch.outputs[1].kind != execution::operand_kind::dense_tensor)
            return fail(operation_core::operation_status_code::invalid_launch_bindings,
                "materialized strategy requires a caller-owned relation output");
        const execution::dense_tensor_view mask_output =
            launch.outputs[1].storage.dense;
        const uint32_t expected_count = state.strategy
                == sequence_strategy::materialize_mask
            ? input.word_count : input.base_count;
        const execution::axis_identity expected_axis = state.strategy
                == sequence_strategy::materialize_mask
            ? state.output_contracts[1].axes[0]
            : state.regulatory_relation_axis;
        if (mask_output.value_type != execution::numeric_type::u32
            || mask_output.rank != 1u
            || mask_output.shape[0] != expected_count
            || mask_output.stride[0] != 1
            || !execution::same_axis_identity(
                mask_output.axes[0], expected_axis)
            || !same_location(mask_output.location, input.location))
            return fail(operation_core::operation_status_code::invalid_launch_bindings,
                "materialized sequence relation output is incompatible");
        materialized = static_cast<uint32_t *>(mask_output.data);
    } else if (state.strategy == sequence_strategy::fuse_predicate) {
        if (launch.output_count != 1u)
            return fail(operation_core::operation_status_code::invalid_launch_bindings,
                "fused strategy has an unexpected materialized output");
    } else {
        return fail(operation_core::operation_status_code::execution_failed,
            "prepared sequence strategy is invalid");
    }
    *validated = {&state, input, output, &values, materialized};
    return {};
}

operation_core::operation_status enqueue_materialization(
    const validated_sequence_launch &launch,
    cudaStream_t stream) noexcept {
    constexpr uint32_t block_size = 128u;
    if (launch.state->strategy == sequence_strategy::materialize_mask) {
        const uint32_t blocks =
            (launch.input.word_count + block_size - 1u) / block_size;
        materialize_predicate_mask_kernel<<<blocks, block_size, 0, stream>>>(
            sequence_planes(launch.input), launch.state->source_domain,
            launch.state->motif, launch.materialized);
    } else if (launch.state->strategy
            == sequence_strategy::materialize_relation) {
        const uint32_t blocks =
            (launch.input.base_count + block_size - 1u) / block_size;
        materialize_regulatory_relation_kernel<<<blocks, block_size, 0, stream>>>(
            sequence_planes(launch.input), launch.state->source_domain,
            launch.state->motif, launch.state->regulatory,
            launch.state->predicate_id, launch.materialized);
    } else {
        return fail(operation_core::operation_status_code::capability_rejected,
            "sequence materialization strategy is unavailable");
    }
    return cudaPeekAtLastError() == cudaSuccess ? operation_core::operation_status{}
        : fail(operation_core::operation_status_code::execution_failed,
            "sequence relation materialization launch failed");
}

operation_core::operation_status enqueue_materialized_accumulation(
    const validated_sequence_launch &launch,
    cudaStream_t stream) noexcept {
    constexpr uint32_t block_size = 128u;
    const uint32_t blocks =
        (launch.input.base_count + block_size - 1u) / block_size;
    if (launch.state->strategy == sequence_strategy::materialize_mask) {
        accumulate_materialized_mask_kernel<<<blocks, block_size, 0, stream>>>(
            launch.materialized, launch.input.base_count,
            launch.state->regulatory, launch.state->predicate_id,
            static_cast<const float *>(launch.values->values),
            static_cast<float *>(launch.output.data));
    } else if (launch.state->strategy
            == sequence_strategy::materialize_relation) {
        accumulate_materialized_relation_kernel<<<blocks, block_size, 0, stream>>>(
            launch.materialized, launch.input.base_count,
            launch.state->regulatory,
            static_cast<const float *>(launch.values->values),
            static_cast<float *>(launch.output.data));
    } else {
        return fail(operation_core::operation_status_code::capability_rejected,
            "sequence materialized consumer is unavailable");
    }
    return cudaPeekAtLastError() == cudaSuccess ? operation_core::operation_status{}
        : fail(operation_core::operation_status_code::execution_failed,
            "cached predicate accumulation launch failed");
}

operation_core::operation_status order_cached_materialization(
    cudaStream_t stream, cudaEvent_t ready, bool cache_hit) noexcept {
    cudaStreamCaptureStatus capture{};
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess
        || capture == cudaStreamCaptureStatusInvalidated)
        return fail(operation_core::operation_status_code::execution_failed,
            "sequence cache stream capture status is unavailable");
    if (!cache_hit) {
        if (capture == cudaStreamCaptureStatusActive)
            return fail(operation_core::operation_status_code::capability_rejected,
                "publish sequence cache materialization before graph capture");
        return {};
    }
    const unsigned int flags = capture == cudaStreamCaptureStatusActive
        ? cudaEventWaitExternal : cudaEventWaitDefault;
    if (cudaStreamWaitEvent(stream, ready, flags) != cudaSuccess)
        return fail(operation_core::operation_status_code::execution_failed,
            "waiting for cached sequence materialization failed");
    return {};
}

operation_core::operation_status enqueue_sequence_operation(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept;

} // namespace

bool validate_regulatory_projection_host(
    const regulatory_projection_view &projection,
    uint32_t local_base_count) noexcept {
    if (projection.location.residency != execution::residency_kind::host
        || projection.location.device_ordinal != -1
        || projection.regulatory_element_count == 0u
        || projection.gene_count == 0u
        || (projection.interval_count != 0u && projection.intervals == nullptr)
        || projection.element_offsets == nullptr
        || (projection.edge_count != 0u && projection.gene_indices == nullptr)
        || projection.element_offsets[0] != 0u)
        return false;
    uint32_t previous_end = 0u;
    for (uint32_t index = 0u; index < projection.interval_count; ++index) {
        const regulatory_interval &interval = projection.intervals[index];
        if (interval.begin >= interval.end || interval.end > local_base_count
            || interval.begin < previous_end
            || interval.regulatory_element
                >= projection.regulatory_element_count)
            return false;
        previous_end = interval.end;
    }
    for (uint32_t element = 0u;
         element < projection.regulatory_element_count; ++element)
        if (projection.element_offsets[element]
                > projection.element_offsets[element + 1u]
            || projection.element_offsets[element + 1u]
                > projection.edge_count)
            return false;
    if (projection.element_offsets[projection.regulatory_element_count]
        != projection.edge_count) return false;
    for (uint32_t edge = 0u; edge < projection.edge_count; ++edge)
        if (projection.gene_indices[edge] >= projection.gene_count) return false;
    return true;
}

operation_core::operation_status prepare_sequence_regulatory_operation(
    const sequence_prepare_request &request,
    const sequence_prepare_policy &policy,
    prepared_sequence_state *state,
    operation_core::prepared_operation *prepared) noexcept {
    if (state == nullptr || prepared == nullptr || request.program == nullptr
        || request.predicate_plan == nullptr)
        return fail(operation_core::operation_status_code::invalid_argument,
            "sequence preparation requires program, predicate plan, and outputs");
    if (ops::sequence::verify_sequence_predicate_program(*request.program)
        != ops::sequence::predicate_plan_status::ok)
        return fail(operation_core::operation_status_code::unsupported_problem,
            "sequence predicate program is invalid");
    ops::sequence::prepared_predicate_plan expected{};
    if (ops::sequence::prepare_sequence_predicate_program(
            *request.program, &expected)
            != ops::sequence::predicate_plan_status::ok
        || !plans_match(expected, *request.predicate_plan))
        return fail(operation_core::operation_status_code::unsupported_problem,
            "prepared predicate plan is stale");
    if (!supported_exact_mask_program(*request.program, expected))
        return fail(operation_core::operation_status_code::unsupported_problem,
            "v1 integration supports one precompiled forward exact-mask predicate");
    if (execution::validate_relation_structure(
            request.coordinate_to_regulatory)
            != execution::lifetime_validation_code::ok
        || execution::validate_relation_structure(request.regulatory_to_gene)
            != execution::lifetime_validation_code::ok
        || execution::validate_sequence_domain(request.source_domain)
            != execution::biological_validation_code::ok
        || !::cellerator::structure::valid_identity(
            request.persistent_coordinate_structure)
        || !::cellerator::structure::valid_identity(
            request.persistent_regulatory_structure)
        || !::cellerator::structure::valid_identity(request.persistent_coordinate_order)
        || !::cellerator::structure::valid_identity(request.persistent_projection)
        || !execution::valid_handle(request.projection)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.domain,
            request.source_domain.genome_domain)
        || !execution::same_axis_identity(
            request.coordinate_to_regulatory.destination_axis,
            request.regulatory_axis)
        || !execution::same_axis_identity(
            request.regulatory_to_gene.source_axis,
            request.regulatory_axis)
        || !execution::valid_axis_identity(request.predicate_mask_axis)
        || !execution::valid_axis_identity(request.regulatory_relation_axis)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.domain,
            request.predicate_mask_axis.domain)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.order,
            request.predicate_mask_axis.order)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.partition,
            request.predicate_mask_axis.partition)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.domain,
            request.regulatory_relation_axis.domain)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.order,
            request.regulatory_relation_axis.order)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.partition,
            request.regulatory_relation_axis.partition)
        || request.source_domain.local_base_count == 0u
        || request.source_domain.local_base_count > tensor::sequence::max_chunk_bases
        || request.regulatory.interval_count
            != request.coordinate_to_regulatory.logical_edge_count
        || request.regulatory.edge_count
            != request.regulatory_to_gene.logical_edge_count
        || !execution::valid_location(request.regulatory.location)
        || request.regulatory.location.residency == execution::residency_kind::host
        || request.regulatory.regulatory_element_count == 0u
        || request.regulatory.gene_count == 0u
        || (request.regulatory.interval_count != 0u
            && request.regulatory.intervals == nullptr)
        || request.regulatory.element_offsets == nullptr
        || (request.regulatory.edge_count != 0u
            && request.regulatory.gene_indices == nullptr))
        return fail(operation_core::operation_status_code::unsupported_projection,
            "sequence regulatory projection metadata is invalid");
    const sequence_measurement_key measurement_key{
        expected.semantic_hash, request.persistent_coordinate_order,
        request.persistent_projection, policy.device,
        policy.runtime_build_identity, request.source_domain.local_base_count,
        request.program->outputs[0].predicate_id,
        request.program->outputs[0].flags};
    const sequence_strategy_decision decision =
        select_sequence_strategy(measurement_key, policy);
    const sequence_strategy strategy = decision.strategy;
    if (strategy == sequence_strategy::automatic)
        return fail(operation_core::operation_status_code::capability_rejected,
            decision.reason);

    *state = prepared_sequence_state{};
    state->strategy = strategy;
    state->predicate_semantic_hash = expected.semantic_hash;
    state->motif = request.program->exact_motifs[0];
    state->predicate_id = request.program->outputs[0].predicate_id;
    state->output_flags = request.program->outputs[0].flags;
    state->persistent_coordinate_structure =
        request.persistent_coordinate_structure;
    state->persistent_coordinate_order = request.persistent_coordinate_order;
    state->source_domain = request.source_domain;
    state->regulatory_relation_axis = request.regulatory_relation_axis;
    state->regulatory = request.regulatory;
    state->input_contracts[0].kind = execution::operand_kind::bit_plane;
    state->input_contracts[0].rank = 1u;
    state->input_contracts[0].axes[0] =
        request.coordinate_to_regulatory.source_axis;
    state->output_contracts[0].kind = execution::operand_kind::dense_tensor;
    state->output_contracts[0].rank = 1u;
    state->output_contracts[0].axes[0] =
        request.regulatory_to_gene.destination_axis;
    state->output_orders[0] = execution::output_axis_contract{
        request.regulatory_to_gene.destination_axis,
        request.regulatory_to_gene.destination_axis,
        execution::order_transition_kind::preserve, 0u, 0u, 0u, 1u, {}, {}};
    state->output_effects[0] = execution::output_effect_contract{
        execution::output_update_kind::accumulate, true, false, 0u,
        execution::invalid_scalar_binding_id,
        execution::invalid_scalar_binding_id};
    const bool materialized = strategy == sequence_strategy::materialize_mask
        || strategy == sequence_strategy::materialize_relation;
    const uint32_t output_count = materialized ? 2u : 1u;
    if (output_count == 2u) {
        state->output_contracts[1].kind = execution::operand_kind::dense_tensor;
        state->output_contracts[1].rank = 1u;
        const execution::axis_identity materialized_axis = strategy
                == sequence_strategy::materialize_mask
            ? request.predicate_mask_axis : request.regulatory_relation_axis;
        state->output_contracts[1].axes[0] = materialized_axis;
        state->output_orders[1] = execution::output_axis_contract{
            materialized_axis, materialized_axis,
            execution::order_transition_kind::preserve, 0u, 1u,
            0u, 1u, {}, {}};
        state->output_effects[1] = execution::output_effect_contract{
            execution::output_update_kind::overwrite, false, false, 0u,
            execution::invalid_scalar_binding_id,
            execution::invalid_scalar_binding_id};
    }

    *prepared = operation_core::prepared_operation{};
    prepared->problem = operation_core::operation_problem{
        operation_core::operation_core_schema_version,
        operation_core::operation_kind::sequence_predicate_accumulate, 0u,
        sequence_operation_id, 1u, output_count,
        request.source_domain.local_base_count};
    prepared->structures.count = 2u;
    prepared->structures.structures[0] = operation_core::structure_key{
        request.persistent_coordinate_structure,
        request.coordinate_to_regulatory.identity,
        request.coordinate_to_regulatory.epoch};
    prepared->structures.structures[1] = operation_core::structure_key{
        request.persistent_regulatory_structure,
        request.regulatory_to_gene.identity,
        request.regulatory_to_gene.epoch};
    prepared->projection = operation_core::projection_key{
        request.persistent_projection, request.projection,
        operation_core::projection_kind::native_feature_major,
        sequence_regulatory_schema_version,
        static_cast<uint32_t>(strategy)};
    prepared->numeric = operation_core::numeric_policy{
        execution::numeric_type::f32, execution::numeric_type::bit,
        execution::numeric_type::f32, execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32,
        execution::numeric_type::invalid,
        operation_core::rounding_policy::nearest_even,
        operation_core::saturation_policy::none,
        operation_core::quantization_granularity::none, {}};
    prepared->kernel = strategy == sequence_strategy::materialize_mask
        ? materialized_kernel_id
        : strategy == sequence_strategy::materialize_relation
            ? relation_materialized_kernel_id : fused_kernel_id;
    prepared->backend = operation_core::backend_kind::native_direct;
    prepared->capability_flags = operation_core::candidate_graph_capture;
    prepared->persistent = {state, sizeof(*state)};
    prepared->binding_contract.structures[0] = {
        request.coordinate_to_regulatory.identity,
        request.coordinate_to_regulatory.epoch};
    prepared->binding_contract.structures[1] = {
        request.regulatory_to_gene.identity,
        request.regulatory_to_gene.epoch};
    prepared->binding_contract.inputs = state->input_contracts;
    prepared->binding_contract.outputs = state->output_contracts;
    prepared->binding_contract.output_orders = state->output_orders;
    prepared->binding_contract.output_effects = state->output_effects;
    prepared->binding_contract.input_count = 1u;
    prepared->binding_contract.output_count = output_count;
    prepared->binding_contract.output_order_count = output_count;
    prepared->binding_contract.structure_count = 2u;
    prepared->binding_contract.output_effect_count = output_count;
    prepared->binding_contract.workspace = {0u, 1u, 0u};
    prepared->run = enqueue_sequence_operation;
    return operation_core::validate_prepared_operation(*prepared);
}

namespace {

operation_core::operation_status enqueue_sequence_operation(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    validated_sequence_launch validated{};
    const operation_core::operation_status valid =
        validate_sequence_launch(prepared, launch, &validated);
    if (!valid) return valid;
    auto stream = static_cast<cudaStream_t>(launch.stream.stream);
    constexpr uint32_t block_size = 128u;
    const uint32_t anchor_blocks =
        (validated.input.base_count + block_size - 1u) / block_size;

    if (validated.state->strategy == sequence_strategy::materialize_mask
        || validated.state->strategy
            == sequence_strategy::materialize_relation) {
        const operation_core::operation_status materialized =
            enqueue_materialization(validated, stream);
        if (!materialized) return materialized;
        return enqueue_materialized_accumulation(validated, stream);
    }
    if (validated.state->strategy == sequence_strategy::fuse_predicate) {
        fused_predicate_accumulate_kernel<<<anchor_blocks, block_size, 0, stream>>>(
            sequence_planes(validated.input), validated.state->source_domain,
            validated.state->motif, validated.state->regulatory,
            validated.state->predicate_id,
            static_cast<const float *>(validated.values->values),
            static_cast<float *>(validated.output.data));
        if (cudaPeekAtLastError() != cudaSuccess)
            return fail(operation_core::operation_status_code::execution_failed,
                "sequence regulatory launch failed");
        return {};
    }
    return fail(operation_core::operation_status_code::execution_failed,
        "prepared sequence strategy is invalid");
}

} // namespace

operation_core::operation_status run_sequence_regulatory_operation_cached(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch,
    execution::value_generation sequence_generation,
    predicate_mask_cache_entry *cache,
    predicate_cache_run_result *result) noexcept {
    if (cache == nullptr || result == nullptr || sequence_generation.value == 0u)
        return fail(operation_core::operation_status_code::invalid_argument,
            "cached sequence execution requires generation, cache, and result");
    *result = {};
    const operation_core::operation_status prepared_status =
        operation_core::validate_prepared_operation(prepared);
    if (!prepared_status) return prepared_status;
    const execution::binding_validation_code binding =
        execution::validate_launch_bindings(prepared.binding_contract, launch);
    if (binding != execution::binding_validation_code::ok)
        return {binding == execution::binding_validation_code::stale_structure
                    ? operation_core::operation_status_code::stale_structure
                    : operation_core::operation_status_code::invalid_launch_bindings,
            binding, "launch bindings do not satisfy prepared contract"};
    validated_sequence_launch validated{};
    const operation_core::operation_status valid =
        validate_sequence_launch(prepared, launch, &validated);
    if (!valid) return valid;
    if (validated.state->strategy != sequence_strategy::materialize_mask)
        return fail(operation_core::operation_status_code::capability_rejected,
            "predicate cache execution requires materialized strategy");
    if (cache->words == nullptr || cache->ready_event == nullptr
        || cache->word_capacity < validated.input.word_count
        || cache->words != validated.materialized
        || !same_location(cache->location, validated.input.location))
        return fail(operation_core::operation_status_code::invalid_launch_bindings,
            "predicate cache storage or completion event is incompatible");

    const predicate_materialization_key key{
        sequence_generation, validated.state->predicate_semantic_hash,
        validated.state->persistent_coordinate_structure,
        prepared.structures.structures[0].epoch,
        validated.state->persistent_coordinate_order,
        validated.state->predicate_id, validated.state->output_flags, 0u};
    auto stream = static_cast<cudaStream_t>(launch.stream.stream);
    const cudaEvent_t ready = static_cast<cudaEvent_t>(cache->ready_event);
    const bool cache_hit = cache->occupied && same_cache_key(cache->key, key);
    const auto ordered = order_cached_materialization(stream, ready, cache_hit);
    if (!ordered) return ordered;
    if (cache_hit) {
        result->cache_hit = true;
        result->launches_enqueued = 1u;
    } else {
        cache->occupied = false;
        const operation_core::operation_status materialized =
            enqueue_materialization(validated, stream);
        if (!materialized) return materialized;
        if (cudaEventRecord(ready, stream) != cudaSuccess)
            return fail(operation_core::operation_status_code::execution_failed,
                "recording cached predicate readiness failed");
        cache->key = key;
        cache->occupied = true;
        result->launches_enqueued = 2u;
    }
    return enqueue_materialized_accumulation(validated, stream);
}

operation_core::operation_status run_sequence_regulatory_relation_cached(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch,
    execution::value_generation sequence_generation,
    regulatory_relation_cache_entry *cache,
    predicate_cache_run_result *result) noexcept {
    if (cache == nullptr || result == nullptr || sequence_generation.value == 0u)
        return fail(operation_core::operation_status_code::invalid_argument,
            "cached relation execution requires generation, cache, and result");
    *result = {};
    const operation_core::operation_status prepared_status =
        operation_core::validate_prepared_operation(prepared);
    if (!prepared_status) return prepared_status;
    const execution::binding_validation_code binding =
        execution::validate_launch_bindings(prepared.binding_contract, launch);
    if (binding != execution::binding_validation_code::ok)
        return {binding == execution::binding_validation_code::stale_structure
                    ? operation_core::operation_status_code::stale_structure
                    : operation_core::operation_status_code::invalid_launch_bindings,
            binding, "launch bindings do not satisfy prepared contract"};
    validated_sequence_launch validated{};
    const operation_core::operation_status valid =
        validate_sequence_launch(prepared, launch, &validated);
    if (!valid) return valid;
    if (validated.state->strategy
            != sequence_strategy::materialize_relation)
        return fail(operation_core::operation_status_code::capability_rejected,
            "direct relation cache requires materialize_relation strategy");
    if (cache->elements == nullptr || cache->ready_event == nullptr
        || cache->element_capacity < validated.input.base_count
        || cache->elements != validated.materialized
        || !same_location(cache->location, validated.input.location))
        return fail(operation_core::operation_status_code::invalid_launch_bindings,
            "direct relation cache storage or completion event is incompatible");

    const regulatory_relation_materialization_key key{
        sequence_generation, validated.state->predicate_semantic_hash,
        validated.state->persistent_coordinate_structure,
        prepared.structures.structures[0].epoch,
        validated.state->persistent_coordinate_order,
        prepared.projection.persistent,
        validated.state->predicate_id, validated.state->output_flags};
    auto stream = static_cast<cudaStream_t>(launch.stream.stream);
    const cudaEvent_t ready = static_cast<cudaEvent_t>(cache->ready_event);
    const bool cache_hit = cache->occupied
        && same_relation_cache_key(cache->key, key);
    const auto ordered = order_cached_materialization(stream, ready, cache_hit);
    if (!ordered) return ordered;
    if (cache_hit) {
        result->cache_hit = true;
        result->launches_enqueued = 1u;
    } else {
        cache->occupied = false;
        const operation_core::operation_status materialized =
            enqueue_materialization(validated, stream);
        if (!materialized) return materialized;
        if (cudaEventRecord(ready, stream) != cudaSuccess)
            return fail(operation_core::operation_status_code::execution_failed,
                "recording cached regulatory relation readiness failed");
        cache->key = key;
        cache->occupied = true;
        result->launches_enqueued = 2u;
    }
    return enqueue_materialized_accumulation(validated, stream);
}

sequence_execution_accounting sequence_accounting(
    const prepared_sequence_state &state,
    uint32_t base_count) noexcept {
    const uint64_t words = (static_cast<uint64_t>(base_count) + 31u) / 32u;
    sequence_execution_accounting result{};
    result.packed_sequence_bytes = words * sizeof(uint64_t);
    result.plane_and_validity_bytes = words * 3u * sizeof(uint32_t);
    result.materialized_mask_bytes =
        state.strategy == sequence_strategy::materialize_mask
            ? words * sizeof(uint32_t) : 0u;
    result.materialized_relation_bytes =
        state.strategy == sequence_strategy::materialize_relation
            ? static_cast<uint64_t>(base_count) * sizeof(uint32_t)
            : 0u;
    result.immutable_relation_bytes =
        static_cast<uint64_t>(state.regulatory.interval_count)
            * sizeof(regulatory_interval)
        + static_cast<uint64_t>(state.regulatory.regulatory_element_count + 1u)
            * sizeof(uint32_t)
        + static_cast<uint64_t>(state.regulatory.edge_count)
            * sizeof(uint32_t);
    result.mutable_value_bytes =
        static_cast<uint64_t>(state.regulatory.edge_count) * sizeof(float);
    result.output_bytes =
        static_cast<uint64_t>(state.regulatory.gene_count) * sizeof(float);
    result.launch_count = state.strategy == sequence_strategy::materialize_mask
            || state.strategy == sequence_strategy::materialize_relation
        ? 2u : 1u;
    return result;
}

} // namespace cellerator::execution::sequence
