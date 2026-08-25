/*
CE-ARCH-78 custom sequence-path evidence (2026-08-25, V100 sm_70):
`python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py run
--spec bench/architecture_evidence/ce_arch_78_v100_spec.json --json` compared
the fused exact-predicate accumulation with first-use and cached materialization
on the same 70-base validity-aware fixture. Nine medians of 100 uses after four
warmups measured 3.645 us fused, 13.148 us first materialized, and 3.154 us
cached (MAD 0%, 0%, 0.325%). With the declared 2% practical tolerance, cached
materialization first wins at reuse 24. Exact referee tolerance was 1e-6. This
supports a measured, replaceable choice; it does not make either kernel a
universal default. Evidence: bench/architecture_evidence/ce_arch_78_v100.json.
*/
#include <Cellerator/compute/sequence/baseplane_integration.cuh>

#include <Baseplane/seq/dna2_ops.hh>
#include <Cellerator/execution/validation.hh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cellerator::compute::sequence {
namespace {

constexpr operation_core::stable_id sequence_operation_id{
    0x7365712d72656731ULL, 0x62617365706c616eULL};
constexpr operation_core::stable_id materialized_kernel_id{
    0x6d61736b2d726567ULL, 0x7365712d76310001ULL};
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

bool same_device_class(
    const execution::device_performance_class &left,
    const execution::device_performance_class &right) noexcept {
    return left.vendor == right.vendor
        && left.architecture_major == right.architecture_major
        && left.architecture_minor == right.architecture_minor
        && left.build_identity == right.build_identity;
}

bool same_measurement_key(
    const sequence_measurement_key &left,
    const sequence_measurement_key &right) noexcept {
    return left.predicate_semantic_hash == right.predicate_semantic_hash
        && execution::same_identity(left.coordinate_order, right.coordinate_order)
        && execution::same_identity(
            left.regulatory_projection, right.regulatory_projection)
        && same_device_class(left.device, right.device)
        && left.runtime_build_identity == right.runtime_build_identity
        && left.local_base_count == right.local_base_count
        && left.predicate_id == right.predicate_id
        && left.output_flags == right.output_flags;
}

bool same_cache_key(
    const predicate_materialization_key &left,
    const predicate_materialization_key &right) noexcept {
    return left.sequence_generation.value == right.sequence_generation.value
        && left.predicate_semantic_hash == right.predicate_semantic_hash
        && execution::same_identity(
            left.coordinate_structure, right.coordinate_structure)
        && execution::same_identity(left.coordinate_order, right.coordinate_order)
        && left.predicate_id == right.predicate_id
        && left.output_flags == right.output_flags;
}

bool plans_match(
    const baseplane::seq::prepared_predicate_plan &left,
    const baseplane::seq::prepared_predicate_plan &right) noexcept {
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
    for (std::uint16_t index = 0u;
         index < baseplane::seq::sequence_program_max_nodes; ++index)
        if (left.original_to_prepared[index]
            != right.original_to_prepared[index]) return false;
    return true;
}

bool supported_exact_mask_program(
    const baseplane::seq::sequence_predicate_program &program,
    const baseplane::seq::prepared_predicate_plan &plan) noexcept {
    if (program.node_count != 1u || program.output_count != 1u
        || program.exact_motif_count != 1u || program.allowed_motif_count != 0u
        || plan.family != baseplane::seq::predicate_lowering_family::exact_scan)
        return false;
    const baseplane::seq::predicate_instruction &node = program.nodes[0];
    const baseplane::seq::predicate_output &output = program.outputs[0];
    return node.opcode == baseplane::seq::predicate_opcode::exact_motif
        && node.result_kind == baseplane::seq::predicate_value_kind::mask
        && node.input_a == baseplane::seq::sequence_program_no_input
        && node.input_b == baseplane::seq::sequence_program_no_input
        && node.immediate == 0u && output.node == 0u
        && output.mode == baseplane::seq::sequence_output_mode::mask
        && (output.flags & baseplane::seq::sequence_event_reverse_strand) == 0u;
}

execution::device_location adapt_location(
    baseplane::seq::sequence_buffer_residency residency,
    std::int32_t device_ordinal) noexcept {
    execution::residency_kind mapped = execution::residency_kind::host;
    switch (residency) {
    case baseplane::seq::sequence_buffer_residency::host:
        mapped = execution::residency_kind::host;
        break;
    case baseplane::seq::sequence_buffer_residency::device:
        mapped = execution::residency_kind::device;
        break;
    case baseplane::seq::sequence_buffer_residency::managed:
        mapped = execution::residency_kind::managed;
        break;
    case baseplane::seq::sequence_buffer_residency::peer_device:
        mapped = execution::residency_kind::peer_device;
        break;
    }
    return {mapped, {}, device_ordinal, 0u};
}

__device__ __forceinline__ std::uint32_t shifted_word(
    const std::uint32_t *words,
    std::uint32_t word_count,
    std::uint32_t anchor) noexcept {
    const std::uint32_t word = anchor >> 5u, shift = anchor & 31u;
    std::uint32_t result = words[word] >> shift;
    if (shift != 0u && word + 1u < word_count)
        result |= words[word + 1u] << (32u - shift);
    return result;
}

__device__ __forceinline__ std::uint32_t active_mask(
    std::uint32_t length) noexcept {
    return length == 32u ? 0xffffffffu : ((1u << length) - 1u);
}

__device__ __forceinline__ bool predicate_matches(
    const execution::bit_plane_view &input,
    const execution::sequence_domain &domain,
    std::uint32_t anchor,
    baseplane::seq::motif32_exact motif) noexcept {
    const std::uint32_t length = motif.length;
    if (anchor < domain.owned_begin || anchor >= domain.owned_end
        || length == 0u || anchor > input.base_count
        || length > input.base_count - anchor)
        return false;
    const std::uint32_t valid = shifted_word(
        input.validity, input.word_count, anchor);
    const std::uint32_t active = active_mask(length);
    if ((valid & active) != active) return false;
    const baseplane::seq::dna2_planes32 sequence{
        shifted_word(input.low, input.word_count, anchor),
        shifted_word(input.high, input.word_count, anchor)};
    const baseplane::seq::dna2_planes32 expected =
        baseplane::seq::unpack_word64_to_planes32(
            baseplane::seq::dna2_word64{motif.packed});
    return baseplane::seq::planes32_mismatches(sequence, expected, active)
        <= motif.max_mismatches;
}

__device__ __forceinline__ std::uint32_t find_regulatory_element(
    regulatory_projection_view projection,
    std::uint32_t anchor,
    std::uint16_t predicate_id) noexcept {
    std::uint32_t lower = 0u, upper = projection.interval_count;
    while (lower < upper) {
        const std::uint32_t middle = lower + (upper - lower) / 2u;
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
    std::uint32_t element,
    const float *weights,
    float *gene_state) noexcept {
    if (element >= projection.regulatory_element_count) return;
    const std::uint32_t begin = projection.element_offsets[element];
    const std::uint32_t end = projection.element_offsets[element + 1u];
    for (std::uint32_t edge = begin; edge < end; ++edge)
        atomicAdd(gene_state + projection.gene_indices[edge], weights[edge]);
}

__global__ void materialize_predicate_mask_kernel(
    execution::bit_plane_view input,
    execution::sequence_domain domain,
    baseplane::seq::motif32_exact motif,
    std::uint32_t *mask) {
    const std::uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= input.word_count) return;
    const std::uint32_t first = word << 5u;
    std::uint32_t result = 0u;
    for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
        const std::uint32_t anchor = first + lane;
        if (anchor < input.base_count
            && predicate_matches(input, domain, anchor, motif))
            result |= 1u << lane;
    }
    mask[word] = result;
}

__global__ void accumulate_materialized_mask_kernel(
    const std::uint32_t *mask,
    std::uint32_t base_count,
    regulatory_projection_view projection,
    std::uint16_t predicate_id,
    const float *weights,
    float *gene_state) {
    const std::uint32_t anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= base_count
        || ((mask[anchor >> 5u] >> (anchor & 31u)) & 1u) == 0u) return;
    const std::uint32_t element = find_regulatory_element(
        projection, anchor, predicate_id);
    if (element != 0xffffffffu)
        accumulate_element(projection, element, weights, gene_state);
}

__global__ void fused_predicate_accumulate_kernel(
    execution::bit_plane_view input,
    execution::sequence_domain domain,
    baseplane::seq::motif32_exact motif,
    regulatory_projection_view projection,
    std::uint16_t predicate_id,
    const float *weights,
    float *gene_state) {
    const std::uint32_t anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= input.base_count
        || !predicate_matches(input, domain, anchor, motif)) return;
    const std::uint32_t element = find_regulatory_element(
        projection, anchor, predicate_id);
    if (element != 0xffffffffu)
        accumulate_element(projection, element, weights, gene_state);
}

struct validated_sequence_launch {
    const prepared_sequence_state *state = nullptr;
    execution::bit_plane_view input{};
    execution::dense_tensor_view output{};
    const execution::value_plane *values = nullptr;
    std::uint32_t *mask = nullptr;
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
    if (state.schema_version != baseplane_integration_schema_version
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

    std::uint32_t *mask = nullptr;
    if (state.strategy == sequence_strategy::materialize_mask) {
        if (launch.output_count != 2u
            || launch.outputs[1].kind != execution::operand_kind::dense_tensor)
            return fail(operation_core::operation_status_code::invalid_launch_bindings,
                "materialized strategy requires a caller-owned mask output");
        const execution::dense_tensor_view mask_output =
            launch.outputs[1].storage.dense;
        if (mask_output.value_type != execution::numeric_type::u32
            || mask_output.rank != 1u
            || mask_output.shape[0] != input.word_count
            || mask_output.stride[0] != 1
            || !same_location(mask_output.location, input.location))
            return fail(operation_core::operation_status_code::invalid_launch_bindings,
                "materialized predicate mask output is incompatible");
        mask = static_cast<std::uint32_t *>(mask_output.data);
    } else if (state.strategy == sequence_strategy::fuse_predicate) {
        if (launch.output_count != 1u)
            return fail(operation_core::operation_status_code::invalid_launch_bindings,
                "fused strategy has an unexpected materialized output");
    } else {
        return fail(operation_core::operation_status_code::execution_failed,
            "prepared sequence strategy is invalid");
    }
    *validated = {&state, input, output, &values, mask};
    return {};
}

operation_core::operation_status enqueue_materialization(
    const validated_sequence_launch &launch,
    cudaStream_t stream) noexcept {
    constexpr std::uint32_t block_size = 128u;
    const std::uint32_t blocks =
        (launch.input.word_count + block_size - 1u) / block_size;
    materialize_predicate_mask_kernel<<<blocks, block_size, 0, stream>>>(
        launch.input, launch.state->source_domain, launch.state->motif,
        launch.mask);
    return cudaPeekAtLastError() == cudaSuccess ? operation_core::operation_status{}
        : fail(operation_core::operation_status_code::execution_failed,
            "predicate-mask launch failed");
}

operation_core::operation_status enqueue_materialized_accumulation(
    const validated_sequence_launch &launch,
    cudaStream_t stream) noexcept {
    constexpr std::uint32_t block_size = 128u;
    const std::uint32_t blocks =
        (launch.input.base_count + block_size - 1u) / block_size;
    accumulate_materialized_mask_kernel<<<blocks, block_size, 0, stream>>>(
        launch.mask, launch.input.base_count, launch.state->regulatory,
        launch.state->predicate_id,
        static_cast<const float *>(launch.values->values),
        static_cast<float *>(launch.output.data));
    return cudaPeekAtLastError() == cudaSuccess ? operation_core::operation_status{}
        : fail(operation_core::operation_status_code::execution_failed,
            "cached predicate accumulation launch failed");
}

} // namespace

bool adapt_baseplane_chunk(
    const baseplane::seq::dna2_chunk_coordinates &source,
    execution::domain_handle genome_domain,
    execution::sequence_domain *destination) noexcept {
    if (destination == nullptr || !execution::valid_handle(genome_domain)
        || !baseplane::seq::dna2_valid_chunk_coordinates(source)
        || source.identity.contig > std::numeric_limits<std::uint32_t>::max()
        || source.identity.chunk > std::numeric_limits<std::uint32_t>::max()
        || source.halo_left > std::numeric_limits<std::uint16_t>::max()
        || source.halo_right > std::numeric_limits<std::uint16_t>::max())
        return false;
    *destination = execution::sequence_domain{
        genome_domain,
        static_cast<std::uint32_t>(source.identity.contig),
        static_cast<std::uint32_t>(source.identity.chunk),
        source.global_base_begin, source.base_count,
        source.owned_begin, source.owned_end,
        static_cast<std::uint16_t>(source.halo_left),
        static_cast<std::uint16_t>(source.halo_right)};
    return execution::validate_sequence_domain(*destination)
        == execution::biological_validation_code::ok;
}

bool adapt_baseplane_planes(
    const baseplane::seq::dna2_planes32_valid_stream_view &source,
    execution::axis_identity coordinate_axis,
    baseplane::seq::sequence_buffer_residency residency,
    std::int32_t device_ordinal,
    execution::biological_operand_view *destination) noexcept {
    if (destination == nullptr || !baseplane::seq::dna2_valid_view(source)
        || source.base_count > std::numeric_limits<std::uint32_t>::max()
        || source.planes.n_words > std::numeric_limits<std::uint32_t>::max()
        || (source.base_count != 0u && source.validity_masks == nullptr))
        return false;
    execution::biological_operand_view result{};
    result.kind = execution::operand_kind::bit_plane;
    result.storage.bits = execution::bit_plane_view{
        coordinate_axis, source.planes.lo_words, source.planes.hi_words,
        source.validity_masks, adapt_location(residency, device_ordinal),
        static_cast<std::uint32_t>(source.planes.n_words),
        static_cast<std::uint32_t>(source.base_count)};
    if (execution::validate_operand(result)
        != execution::biological_validation_code::ok) return false;
    *destination = result;
    return true;
}

bool validate_regulatory_projection_host(
    const regulatory_projection_view &projection,
    std::uint32_t local_base_count) noexcept {
    if (projection.location.residency != execution::residency_kind::host
        || projection.location.device_ordinal != -1
        || projection.regulatory_element_count == 0u
        || projection.gene_count == 0u
        || (projection.interval_count != 0u && projection.intervals == nullptr)
        || projection.element_offsets == nullptr
        || (projection.edge_count != 0u && projection.gene_indices == nullptr)
        || projection.element_offsets[0] != 0u)
        return false;
    std::uint32_t previous_end = 0u;
    for (std::uint32_t index = 0u; index < projection.interval_count; ++index) {
        const regulatory_interval &interval = projection.intervals[index];
        if (interval.begin >= interval.end || interval.end > local_base_count
            || interval.begin < previous_end
            || interval.regulatory_element
                >= projection.regulatory_element_count)
            return false;
        previous_end = interval.end;
    }
    for (std::uint32_t element = 0u;
         element < projection.regulatory_element_count; ++element)
        if (projection.element_offsets[element]
                > projection.element_offsets[element + 1u]
            || projection.element_offsets[element + 1u]
                > projection.edge_count)
            return false;
    if (projection.element_offsets[projection.regulatory_element_count]
        != projection.edge_count) return false;
    for (std::uint32_t edge = 0u; edge < projection.edge_count; ++edge)
        if (projection.gene_indices[edge] >= projection.gene_count) return false;
    return true;
}

sequence_strategy select_sequence_strategy(
    const sequence_prepare_policy &policy) noexcept {
    if (policy.requested == sequence_strategy::materialize_mask)
        return policy.allow_materialization
            ? sequence_strategy::materialize_mask : sequence_strategy::automatic;
    if (policy.requested == sequence_strategy::fuse_predicate)
        return policy.allow_fusion
            ? sequence_strategy::fuse_predicate : sequence_strategy::automatic;
    if (policy.allow_fusion != policy.allow_materialization)
        return policy.allow_fusion ? sequence_strategy::fuse_predicate
            : sequence_strategy::materialize_mask;
    return sequence_strategy::automatic;
}

sequence_strategy_decision select_sequence_strategy(
    const sequence_measurement_key &key,
    const sequence_prepare_policy &policy) noexcept {
    sequence_strategy_decision decision{};
    if (policy.requested == sequence_strategy::materialize_mask) {
        decision.strategy = policy.allow_materialization
            ? sequence_strategy::materialize_mask : sequence_strategy::automatic;
        decision.empirical_measurement_required = false;
        decision.reason = policy.allow_materialization
            ? "materialization explicitly requested"
            : "requested materialization is unavailable";
        return decision;
    }
    if (policy.requested == sequence_strategy::fuse_predicate) {
        decision.strategy = policy.allow_fusion
            ? sequence_strategy::fuse_predicate : sequence_strategy::automatic;
        decision.empirical_measurement_required = false;
        decision.reason = policy.allow_fusion
            ? "fusion explicitly requested" : "requested fusion is unavailable";
        return decision;
    }
    if (!policy.allow_materialization && !policy.allow_fusion) {
        decision.reason = "materialization and fusion are both unavailable";
        return decision;
    }
    if (!policy.allow_materialization || !policy.allow_fusion) {
        decision.strategy = policy.allow_materialization
            ? sequence_strategy::materialize_mask
            : sequence_strategy::fuse_predicate;
        decision.empirical_measurement_required = false;
        decision.reason = "only one capable sequence strategy remains";
        return decision;
    }

    const sequence_strategy_evidence *evidence = policy.evidence;
    const bool valid_policy = std::isfinite(policy.practical_tolerance_percent)
        && policy.practical_tolerance_percent >= 0.0
        && policy.practical_tolerance_percent < 100.0
        && std::isfinite(policy.maximum_spread_percent)
        && policy.maximum_spread_percent >= 0.0;
    const bool valid_evidence = evidence != nullptr && valid_policy
        && same_measurement_key(evidence->key, key)
        && evidence->sample_count >= 3u
        && std::isfinite(evidence->fused_per_use_ns)
        && std::isfinite(evidence->first_materialized_use_ns)
        && std::isfinite(evidence->cached_materialized_use_ns)
        && std::isfinite(evidence->fused_spread_percent)
        && std::isfinite(evidence->materialized_spread_percent)
        && evidence->fused_per_use_ns > 0.0
        && evidence->first_materialized_use_ns > 0.0
        && evidence->cached_materialized_use_ns > 0.0
        && evidence->fused_spread_percent >= 0.0
        && evidence->materialized_spread_percent >= 0.0
        && evidence->fused_spread_percent <= policy.maximum_spread_percent
        && evidence->materialized_spread_percent
            <= policy.maximum_spread_percent;
    if (!valid_evidence) {
        decision.reason = "current comparable measurement is required";
        return decision;
    }

    const double reuse = static_cast<double>(
        policy.expected_predicate_reuse == 0u
            ? 1u : policy.expected_predicate_reuse);
    decision.fused_total_ns = evidence->fused_per_use_ns * reuse;
    decision.materialized_total_ns = evidence->first_materialized_use_ns
        + evidence->cached_materialized_use_ns * (reuse - 1.0);
    const double required_ratio =
        1.0 - policy.practical_tolerance_percent / 100.0;
    decision.strategy = decision.materialized_total_ns
            < decision.fused_total_ns * required_ratio
        ? sequence_strategy::materialize_mask
        : sequence_strategy::fuse_predicate;
    decision.empirical_measurement_required = false;
    decision.reason = decision.strategy == sequence_strategy::materialize_mask
        ? "measured amortized materialization cost is lower"
        : "fusion wins or is within practical tolerance";
    return decision;
}

operation_core::operation_status prepare_sequence_regulatory_operation(
    const sequence_prepare_request &request,
    const sequence_prepare_policy &policy,
    prepared_sequence_state *state,
    operation_core::prepared_operation *prepared) noexcept {
    if (state == nullptr || prepared == nullptr || request.program == nullptr
        || request.baseplane_plan == nullptr)
        return fail(operation_core::operation_status_code::invalid_argument,
            "sequence preparation requires program, Baseplane plan, and outputs");
    if (baseplane::seq::verify_sequence_predicate_program(*request.program)
        != baseplane::seq::predicate_plan_status::ok)
        return fail(operation_core::operation_status_code::unsupported_problem,
            "Baseplane predicate program is invalid");
    baseplane::seq::prepared_predicate_plan expected{};
    if (baseplane::seq::prepare_sequence_predicate_program(
            *request.program, &expected)
            != baseplane::seq::predicate_plan_status::ok
        || !plans_match(expected, *request.baseplane_plan))
        return fail(operation_core::operation_status_code::unsupported_problem,
            "Baseplane prepared predicate plan is stale");
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
        || !execution::valid_identity(
            request.persistent_coordinate_structure)
        || !execution::valid_identity(
            request.persistent_regulatory_structure)
        || !execution::valid_identity(request.persistent_coordinate_order)
        || !execution::valid_identity(request.persistent_projection)
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
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.domain,
            request.predicate_mask_axis.domain)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.order,
            request.predicate_mask_axis.order)
        || !execution::same_handle(
            request.coordinate_to_regulatory.source_axis.partition,
            request.predicate_mask_axis.partition)
        || request.source_domain.local_base_count == 0u
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
    const std::uint32_t output_count =
        strategy == sequence_strategy::materialize_mask ? 2u : 1u;
    if (output_count == 2u) {
        state->output_contracts[1].kind = execution::operand_kind::dense_tensor;
        state->output_contracts[1].rank = 1u;
        state->output_contracts[1].axes[0] = request.predicate_mask_axis;
        state->output_orders[1] = execution::output_axis_contract{
            request.predicate_mask_axis, request.predicate_mask_axis,
            execution::order_transition_kind::preserve, 0u, 1u,
            0u, 1u, {}, {}};
        state->output_effects[1] = execution::output_effect_contract{
            execution::output_update_kind::partial_write, true, false, 0u,
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
        baseplane_integration_schema_version,
        static_cast<std::uint32_t>(strategy)};
    prepared->numeric = operation_core::numeric_policy{
        execution::numeric_type::f32, execution::numeric_type::bit,
        execution::numeric_type::f32, execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32,
        execution::numeric_type::invalid,
        operation_core::rounding_policy::nearest_even,
        operation_core::saturation_policy::none,
        operation_core::quantization_granularity::none, {}};
    prepared->kernel = strategy == sequence_strategy::materialize_mask
        ? materialized_kernel_id : fused_kernel_id;
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
    prepared->run = run_sequence_regulatory_operation;
    return operation_core::validate_prepared_operation(*prepared);
}

operation_core::operation_status run_sequence_regulatory_operation(
    const operation_core::prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    validated_sequence_launch validated{};
    const operation_core::operation_status valid =
        validate_sequence_launch(prepared, launch, &validated);
    if (!valid) return valid;
    auto stream = static_cast<cudaStream_t>(launch.stream.stream);
    constexpr std::uint32_t block_size = 128u;
    const std::uint32_t anchor_blocks =
        (validated.input.base_count + block_size - 1u) / block_size;

    if (validated.state->strategy == sequence_strategy::materialize_mask) {
        const operation_core::operation_status materialized =
            enqueue_materialization(validated, stream);
        if (!materialized) return materialized;
        return enqueue_materialized_accumulation(validated, stream);
    }
    if (validated.state->strategy == sequence_strategy::fuse_predicate) {
        fused_predicate_accumulate_kernel<<<anchor_blocks, block_size, 0, stream>>>(
            validated.input, validated.state->source_domain,
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
        || cache->words != validated.mask
        || !same_location(cache->location, validated.input.location))
        return fail(operation_core::operation_status_code::invalid_launch_bindings,
            "predicate cache storage or completion event is incompatible");

    const predicate_materialization_key key{
        sequence_generation, validated.state->predicate_semantic_hash,
        validated.state->persistent_coordinate_structure,
        validated.state->persistent_coordinate_order,
        validated.state->predicate_id, validated.state->output_flags, 0u};
    auto stream = static_cast<cudaStream_t>(launch.stream.stream);
    const cudaEvent_t ready = static_cast<cudaEvent_t>(cache->ready_event);
    if (cache->occupied && same_cache_key(cache->key, key)) {
        if (cudaStreamWaitEvent(stream, ready, 0u) != cudaSuccess)
            return fail(operation_core::operation_status_code::execution_failed,
                "waiting for cached predicate mask failed");
        result->cache_hit = true;
        result->launches_enqueued = 1u;
    } else {
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

sequence_execution_accounting sequence_accounting(
    const prepared_sequence_state &state,
    std::uint32_t base_count) noexcept {
    const std::uint64_t words = (static_cast<std::uint64_t>(base_count) + 31u) / 32u;
    sequence_execution_accounting result{};
    result.packed_sequence_bytes = words * sizeof(std::uint64_t);
    result.plane_and_validity_bytes = words * 3u * sizeof(std::uint32_t);
    result.materialized_mask_bytes =
        state.strategy == sequence_strategy::materialize_mask
            ? words * sizeof(std::uint32_t) : 0u;
    result.immutable_relation_bytes =
        static_cast<std::uint64_t>(state.regulatory.interval_count)
            * sizeof(regulatory_interval)
        + static_cast<std::uint64_t>(state.regulatory.regulatory_element_count + 1u)
            * sizeof(std::uint32_t)
        + static_cast<std::uint64_t>(state.regulatory.edge_count)
            * sizeof(std::uint32_t);
    result.mutable_value_bytes =
        static_cast<std::uint64_t>(state.regulatory.edge_count) * sizeof(float);
    result.output_bytes =
        static_cast<std::uint64_t>(state.regulatory.gene_count) * sizeof(float);
    result.launch_count = state.strategy == sequence_strategy::materialize_mask
        ? 2u : 1u;
    return result;
}

} // namespace cellerator::compute::sequence
