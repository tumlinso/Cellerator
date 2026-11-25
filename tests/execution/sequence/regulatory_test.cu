#include <Cellerator/execution/sequence/regulatory.cuh>
#include <Cellerator/execution/sequence/bindings.hh>
#include <Cellerator/sequence/cuda.hh>
#include <Cellerator/execution/validation.hh>

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace cs = cellerator::execution::sequence;
namespace ce = cellerator::execution;
namespace co = cellerator::compute::math::core;
namespace sq = cellerator::sequence;

namespace {

template<typename T>
void require(T condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "%s\n", message);
    exit(1);
}

void cuda_require(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
    exit(1);
}

template<typename T>
T *allocate_device(size_t count) {
    T *data = nullptr;
    cuda_require(cudaMalloc(&data, count * sizeof(T)), "allocate fixture storage");
    return data;
}

ce::axis_identity axis(
    ce::u32 domain, ce::u32 order, ce::u32 geometry, ce::u32 partition) {
    return {{domain, 1u}, {order, 1u}, {geometry, 1u}, {partition, 1u}};
}

void place_motif(char *bases, uint32_t position, char middle = 'C') {
    bases[position] = 'A';
    bases[position + 1u] = middle;
    bases[position + 2u] = 'G';
}

sq::sequence_predicate_program make_program() {
    sq::sequence_predicate_program program{};
    program.version = sq::sequence_program_version;
    program.node_count = 1u;
    program.output_count = 1u;
    program.exact_motif_count = 1u;
    const uint64_t acg = (1u << 2u) | (2u << 4u);
    require(sq::dna2_normalize_motif32_exact(
        acg, 3u, 0u, 7u, &program.exact_motifs[0]),
        "failed to normalize test motif");
    program.nodes[0] = sq::predicate_instruction{
        sq::predicate_opcode::exact_motif, sq::predicate_value_kind::mask,
        0u, 0u, sq::sequence_program_no_input,
        sq::sequence_program_no_input, 0u, 0u};
    program.outputs[0] = sq::predicate_output{
        0u, 7u, sq::sequence_output_mode::mask, 0u,
        sq::sequence_event_forward_strand, 0u};
    return program;
}

void scalar_reference(
    const char *bases,
    const sq::SequenceChunk &chunk,
    const cs::regulatory_interval (&intervals)[3],
    const uint32_t *offsets,
    const uint32_t *genes,
    const float *weights,
    float (&result)[3]) {
    for (uint32_t anchor = chunk.owned_begin; anchor < chunk.owned_end; ++anchor) {
        if (3u > chunk.base_count - anchor
            || bases[anchor] != 'A' || bases[anchor + 1u] != 'C'
            || bases[anchor + 2u] != 'G') continue;
        for (const cs::regulatory_interval &interval : intervals) {
            if (anchor < interval.begin || anchor >= interval.end
                || interval.predicate_id != 7u) continue;
            for (uint32_t edge = offsets[interval.regulatory_element];
                 edge < offsets[interval.regulatory_element + 1u]; ++edge)
                result[genes[edge]] += weights[edge];
            break;
        }
    }
}

ce::biological_operand_view output_view(void *data, ce::axis_identity axis,
    ce::device_location location, ce::numeric_type type, uint32_t count) {
    ce::biological_operand_view output{};
    output.kind = ce::operand_kind::dense_tensor;
    auto &dense = output.storage.dense;
    dense.data = data;
    dense.location = location;
    dense.value_type = type;
    dense.rank = 1u;
    dense.axes[0] = axis;
    dense.shape[0] = count;
    dense.stride[0] = 1;
    return output;
}

ce::launch_bindings launch_bindings(
    const ce::relation_structure *structures,
    const ce::biological_operand_view *input,
    ce::biological_operand_view *outputs,
    uint32_t output_count,
    const ce::value_binding *values,
    cudaStream_t stream,
    int device) {
    ce::launch_bindings launch{};
    launch.structures = structures;
    launch.inputs = input;
    launch.outputs = outputs;
    launch.values = values;
    launch.input_count = 1u;
    launch.output_count = output_count;
    launch.value_count = 1u;
    launch.structure_count = 2u;
    launch.stream = {stream, device, 0u};
    launch.workspace = {nullptr, 0u,
        {ce::residency_kind::device, {}, device, 0u}};
    return launch;
}

void compare_output(
    const float *actual,
    const float *expected,
    const char *path) {
    for (size_t index = 0u; index < 3u; ++index)
        require(__builtin_fabsf(actual[index] - expected[index]) < 1e-6f, path);
}

void sort_samples(double *x, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        const double value = x[i];
        size_t j = i;
        while (j && x[j - 1] > value) {
            x[j] = x[j - 1];
            --j;
        }
        x[j] = value;
    }
}

struct timing_summary {
    double median_ns = 0.0;
    double spread_percent = 0.0;
};

template<typename Enqueue>
timing_summary measure_candidate(
    cudaStream_t stream,
    Enqueue enqueue,
    const char *label) {
    constexpr uint32_t warmups = 4u;
    constexpr uint32_t samples = 9u;
    constexpr uint32_t uses_per_sample = 100u;
    for (uint32_t index = 0u;
         index < warmups * uses_per_sample; ++index)
        require(enqueue(index), label);
    cuda_require(cudaStreamSynchronize(stream), "finish benchmark warmup");
    cudaEvent_t begin = nullptr, end = nullptr;
    cuda_require(cudaEventCreate(&begin), "cudaEventCreate begin");
    cuda_require(cudaEventCreate(&end), "cudaEventCreate end");
    double elapsed[samples]{};
    for (uint32_t index = 0u; index < samples; ++index) {
        cuda_require(cudaEventRecord(begin, stream), "record benchmark begin");
        for (uint32_t use = 0u; use < uses_per_sample; ++use)
            require(enqueue((warmups + index) * uses_per_sample + use), label);
        cuda_require(cudaEventRecord(end, stream), "record benchmark end");
        cuda_require(cudaEventSynchronize(end), "finish benchmark sample");
        float milliseconds = 0.0f;
        cuda_require(cudaEventElapsedTime(&milliseconds, begin, end),
            "measure benchmark sample");
        elapsed[index] = static_cast<double>(milliseconds) * 1.0e6
            / static_cast<double>(uses_per_sample);
    }
    cuda_require(cudaEventDestroy(end), "cudaEventDestroy end");
    cuda_require(cudaEventDestroy(begin), "cudaEventDestroy begin");
    sort_samples(elapsed, samples);
    const double median = elapsed[samples / 2u];
    double deviation[samples]{};
    for (uint32_t index = 0u; index < samples; ++index)
        deviation[index] = __builtin_fabs(elapsed[index] - median);
    sort_samples(deviation, samples);
    return {median, median == 0.0 ? 0.0
        : deviation[samples / 2u] * 100.0 / median};
}

} // namespace

int main(int argc, char **argv) {
    const bool benchmark = argc == 2 && __builtin_strcmp(argv[1], "--benchmark") == 0;
    require(argc == 1 || benchmark, "usage: celleratorSequenceRegulatoryTest [--benchmark]");
    int device_count = 0;
    cuda_require(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    require(device_count > 0, "no CUDA device available");
    int device = 0;
    cuda_require(cudaGetDevice(&device), "cudaGetDevice");
    cudaStream_t stream = nullptr;
    cuda_require(cudaStreamCreate(&stream), "create fixture stream");
    cudaEvent_t cache_ready = nullptr;
    cuda_require(cudaEventCreateWithFlags(&cache_ready, cudaEventDisableTiming),
        "create relation readiness event");

    constexpr uint32_t base_count = 70u, word_count = 3u, validity_words = 2u;
    char bases[base_count];
    __builtin_memset(bases, 'T', sizeof(bases));
    place_motif(bases, 0u);
    place_motif(bases, 5u);
    place_motif(bases, 30u);
    place_motif(bases, 40u, 'N');
    place_motif(bases, 58u);
    place_motif(bases, 62u);
    uint64_t packed[word_count]{}, validity[validity_words]{};
    for (uint32_t i = 0u; i < base_count; ++i) {
        const char base = bases[i];
        // Make the invalid payload itself match so that only validity rejects it.
        const uint64_t code = base == 'A' ? 0u : base == 'G' ? 2u
            : base == 'T' ? 3u : 1u;
        packed[i / 32u] |= code << (2u * (i % 32u));
        if (base != 'N') validity[i / 64u] |= 1ull << (i % 64u);
    }

    const sq::sequence_predicate_program program = make_program();
    sq::prepared_predicate_plan predicate_plan{};
    require(sq::prepare_sequence_predicate_program(program, &predicate_plan)
            == sq::predicate_plan_status::ok,
        "sequence plan preparation failed");
    const sq::SequenceChunk chunk{
        {17u, 1u}, 2u, 9u, 1ull << 40u, base_count,
        4u, 60u, 4u, 10u};
    ce::sequence_domain source_domain{};
    require(cs::bind_sequence_chunk(chunk, {41u, 1u}, &source_domain),
        "chunk adapter rejected bounded coordinates");
    sq::SequenceChunk oversized_chunk = chunk;
    oversized_chunk.contig = UINT64_MAX;
    require(!cs::bind_sequence_chunk(
            oversized_chunk, {41u, 1u}, &source_domain),
        "unrepresentable contig identity was truncated");

    const ce::axis_identity coordinate_axis = axis(41u, 42u, 43u, 44u);
    const ce::axis_identity regulatory_axis = axis(46u, 47u, 48u, 49u);
    const ce::axis_identity predicate_mask_axis = axis(41u, 42u, 45u, 44u);
    const ce::axis_identity regulatory_relation_axis =
        axis(41u, 42u, 50u, 44u);
    const ce::axis_identity gene_axis = axis(51u, 52u, 53u, 54u);
    const cs::regulatory_interval intervals[3]{
        {4u, 20u, 0u, 7u, 0u},
        {20u, 50u, 1u, 7u, 0u},
        {50u, 60u, 2u, 7u, 0u}};
    const uint32_t offsets[4]{0u, 1u, 3u, 3u};
    const uint32_t genes[3]{2u, 0u, 1u};
    const float weights[3]{1.5f, 2.0f, 0.5f};
    const cs::regulatory_projection_view host_projection{
        intervals, offsets, genes, 3u, 3u, 3u, 3u,
        {ce::residency_kind::host, {}, -1, 0u}};
    require(cs::validate_regulatory_projection_host(host_projection, base_count),
        "valid host regulatory projection rejected");
    cs::regulatory_interval overlapping[3];
    __builtin_memcpy(overlapping, intervals, sizeof(intervals));
    overlapping[1].begin = 19u;
    require(!cs::validate_regulatory_projection_host(
            {overlapping, offsets, genes,
             3u, 3u, 3u, 3u,
             {ce::residency_kind::host, {}, -1, 0u}}, base_count),
        "overlapping interval projection accepted");

    uint64_t *device_packed = allocate_device<uint64_t>(word_count);
    uint64_t *device_source_validity = allocate_device<uint64_t>(validity_words);
    uint32_t *device_low = allocate_device<uint32_t>(word_count);
    uint32_t *device_high = allocate_device<uint32_t>(word_count);
    uint32_t *device_validity = allocate_device<uint32_t>(word_count);
    uint32_t *device_mask = allocate_device<uint32_t>(word_count);
    uint32_t *device_relation = allocate_device<uint32_t>(base_count);
    uint32_t *device_offsets = allocate_device<uint32_t>(4u);
    uint32_t *device_genes = allocate_device<uint32_t>(3u);
    cs::regulatory_interval *device_intervals = allocate_device<cs::regulatory_interval>(3u);
    float *device_weights = allocate_device<float>(3u);
    float *device_output = allocate_device<float>(3u);

    cuda_require(cudaMemcpyAsync(device_packed, packed, sizeof(packed),
        cudaMemcpyHostToDevice, stream), "copy packed sequence");
    cuda_require(cudaMemcpyAsync(device_source_validity, validity, sizeof(validity),
        cudaMemcpyHostToDevice, stream), "copy packed validity");
    cuda_require(cudaMemcpyAsync(device_intervals, intervals,
        sizeof(intervals), cudaMemcpyHostToDevice, stream), "copy intervals");
    cuda_require(cudaMemcpyAsync(device_offsets, offsets,
        sizeof(offsets), cudaMemcpyHostToDevice, stream), "copy offsets");
    cuda_require(cudaMemcpyAsync(device_genes, genes,
        sizeof(genes), cudaMemcpyHostToDevice, stream), "copy genes");
    cuda_require(cudaMemcpyAsync(device_weights, weights,
        sizeof(weights), cudaMemcpyHostToDevice, stream), "copy weights");
    cuda_require(sq::pack_planes_cuda(
        {device_packed, device_source_validity, word_count, base_count, validity_words},
        {device_low, device_high, device_validity, word_count}, stream),
        "prepare device sequence planes");
    const ce::device_location location{ce::residency_kind::device, {}, device, 0u};
    ce::biological_operand_view input{};
    const sq::BitPlanesView planes{
        device_low, device_high, device_validity, word_count, base_count};
    require(cs::bind_sequence_planes(
            planes, coordinate_axis, location, &input),
        "bind device sequence planes");
    require(input.storage.bits.low == device_low && input.storage.bits.high == device_high
            && input.storage.bits.validity == device_validity,
        "binding did not borrow prepared plane arrays");
    ce::biological_operand_view missing_validity{};
    require(!cs::bind_sequence_planes(
            {device_low, device_high, nullptr, word_count, base_count},
            coordinate_axis, location, &missing_validity),
        "implicit nonempty validity was accepted at the Cellerator seam");
    ce::relation_structure relations[2]{
        {{61u, 1u}, {1u}, coordinate_axis, regulatory_axis,
            {62u, 1u}, 3u},
        {{64u, 1u}, {2u}, regulatory_axis, gene_axis,
            {65u, 1u}, 3u}};
    const cs::regulatory_projection_view device_projection{
        device_intervals, device_offsets, device_genes,
        3u, 3u, 3u, 3u,
        {ce::residency_kind::device, {}, device, 0u}};
    cs::sequence_prepare_request request{};
    request.program = &program;
    request.predicate_plan = &predicate_plan;
    request.persistent_coordinate_structure = {71u, 72u};
    request.persistent_regulatory_structure = {75u, 76u};
    request.persistent_coordinate_order = {77u, 78u};
    request.persistent_projection = {73u, 74u};
    request.projection = {63u, 1u};
    request.coordinate_to_regulatory = relations[0];
    request.regulatory_to_gene = relations[1];
    request.source_domain = source_domain;
    request.regulatory_axis = regulatory_axis;
    request.predicate_mask_axis = predicate_mask_axis;
    request.regulatory_relation_axis = regulatory_relation_axis;
    request.regulatory = device_projection;
    ce::value_plane value_plane{
        relations[1].identity, relations[1].epoch, device_weights,
        {ce::residency_kind::device, {}, device, 0u},
        {ce::numeric_type::f32, ce::numeric_type::f32,
         ce::numeric_type::f32, 0u},
        {ce::quantization_kind::none, ce::numeric_type::invalid,
         ce::numeric_type::invalid, 0u, nullptr, nullptr, 0u},
        ce::value_layout_kind::logical_edge_order, {}, {1u}, 3u,
        3u * sizeof(float)};
    ce::value_binding value_binding{&value_plane, {1u}};

    float contribution[3]{};
    scalar_reference(bases, chunk, intervals, offsets, genes, weights, contribution);
    require(contribution[0] == 2.0f && contribution[1] == 0.5f
            && contribution[2] == 1.5f,
        "scalar fixture does not cover expected regulatory mapping");

    const ce::device_performance_class measurement_device{
        1u, 7u, 0u, 0x5a17u};
    const cs::sequence_measurement_key measurement_key{
        predicate_plan.semantic_hash, request.persistent_coordinate_order,
        request.persistent_projection, measurement_device, 0xce780001u,
        base_count, program.outputs[0].predicate_id,
        program.outputs[0].flags};
    cs::sequence_strategy_evidence evidence{
        measurement_key, 1000.0, 1800.0, 100.0, 1.0, 1.0, 7u, 0u};
    cs::sequence_prepare_policy fused_policy{};
    fused_policy.expected_predicate_reuse = 1u;
    fused_policy.evidence = &evidence;
    fused_policy.device = measurement_device;
    fused_policy.runtime_build_identity = 0xce780001u;
    cs::sequence_prepare_policy materialized_policy = fused_policy;
    materialized_policy.expected_cell_state_count = 4u;

    cs::prepared_sequence_state rejected_state{};
    co::prepared_operation rejected{};
    require(!cs::prepare_sequence_regulatory_operation(
            request, {}, &rejected_state, &rejected),
        "automatic selection accepted missing empirical evidence");

    cs::prepared_sequence_state fused_state{};
    co::prepared_operation fused{};
    require(cs::prepare_sequence_regulatory_operation(
            request, fused_policy,
            &fused_state, &fused),
        "fused operation did not prepare");
    require(fused_state.strategy == cs::sequence_strategy::fuse_predicate,
        "one-shot policy did not select fusion");
    require(fused.binding_contract.output_effect_count == 1u
            && fused.binding_contract.output_effects[0].update
                == ce::output_update_kind::accumulate
            && fused.binding_contract.output_effects[0]
                .requires_initialized_destination,
        "gene-state accumulation effect was not declared");
    cs::sequence_prepare_request wrong_coordinate_relation = request;
    wrong_coordinate_relation.coordinate_to_regulatory.destination_axis =
        axis(46u, 47u, 148u, 49u);
    require(!cs::prepare_sequence_regulatory_operation(
            wrong_coordinate_relation, fused_policy, &rejected_state, &rejected),
        "coordinate-to-regulatory relation with wrong biology was accepted");
    cs::sequence_prepare_request wrong_regulatory_relation = request;
    wrong_regulatory_relation.regulatory_to_gene.source_axis =
        axis(46u, 47u, 48u, 149u);
    require(!cs::prepare_sequence_regulatory_operation(
            wrong_regulatory_relation, fused_policy, &rejected_state, &rejected),
        "regulatory-to-gene relation with wrong biology was accepted");
    ce::biological_operand_view fused_output =
        output_view(device_output, gene_axis, location, ce::numeric_type::f32, 3u);
    const float initial[3]{10.0f, 20.0f, 30.0f};
    float accumulated_expected[3]{};
    for (size_t index = 0u; index < 3u; ++index)
        accumulated_expected[index] = initial[index] + contribution[index];
    cuda_require(cudaMemcpyAsync(device_output, initial,
        sizeof(initial), cudaMemcpyHostToDevice, stream),
        "initialize fused accumulated output");
    ce::launch_bindings fused_launch = launch_bindings(
        relations, &input, &fused_output, 1u, &value_binding, stream, device);
    require(co::run_prepared_operation(fused, fused_launch),
        "fused operation launch rejected");
    ++relations[0].epoch.value;
    require(!co::run_prepared_operation(fused, fused_launch),
        "stale coordinate relation epoch was accepted");
    relations[0] = request.coordinate_to_regulatory;
    ++relations[1].epoch.value;
    require(!co::run_prepared_operation(fused, fused_launch),
        "stale regulatory relation epoch was accepted");
    relations[1] = request.regulatory_to_gene;
    ce::value_plane wrong_relation_value_plane = value_plane;
    wrong_relation_value_plane.structure = relations[0].identity;
    wrong_relation_value_plane.structure_epoch_value = relations[0].epoch;
    const ce::value_binding wrong_relation_value{
        &wrong_relation_value_plane, {1u}};
    fused_launch.values = &wrong_relation_value;
    require(!co::run_prepared_operation(fused, fused_launch),
        "regulatory weights bound to coordinate relation were accepted");
    fused_launch.values = &value_binding;
    float fused_result[3]{};
    cuda_require(cudaMemcpyAsync(fused_result, device_output,
        sizeof(fused_result), cudaMemcpyDeviceToHost, stream),
        "copy fused output");
    cuda_require(cudaStreamSynchronize(stream), "finish fused path");
    compare_output(fused_result, accumulated_expected,
        "fused accumulation did not preserve initial gene state");

    // A new coordinate epoch can exclude an anchor while preserving the
    // sequence words, predicate, coordinate IDs, order, and projection.
    sq::SequenceChunk narrowed_chunk = chunk;
    narrowed_chunk.owned_begin = narrowed_chunk.halo_left = 6u;
    cs::sequence_prepare_request narrowed_request = request;
    ++narrowed_request.coordinate_to_regulatory.epoch.value;
    require(cs::bind_sequence_chunk(narrowed_chunk, source_domain.genome_domain,
            &narrowed_request.source_domain), "bind a narrower owned sequence span");
    const ce::relation_structure narrowed_relations[2]{
        narrowed_request.coordinate_to_regulatory, narrowed_request.regulatory_to_gene};

    {
        cs::sequence_prepare_policy policy{};
        policy.requested = cs::sequence_strategy::materialize_mask;
        cs::prepared_sequence_state state{};
        co::prepared_operation prepared{};
        require(cs::prepare_sequence_regulatory_operation(request, policy, &state, &prepared)
                && state.strategy == cs::sequence_strategy::materialize_mask,
            "explicit mask materialization did not prepare");
        ce::biological_operand_view outputs[2]{
            fused_output,
            output_view(device_mask, predicate_mask_axis, location, ce::numeric_type::u32, word_count)};
        const ce::launch_bindings launch =
            launch_bindings(relations, &input, outputs, 2u, &value_binding, stream, device);
        cudaEvent_t mask_ready = nullptr;
        cuda_require(cudaEventCreateWithFlags(&mask_ready, cudaEventDisableTiming),
            "create mask readiness event");
        cs::predicate_mask_cache_entry mask{};
        mask.words = device_mask;
        mask.ready_event = mask_ready;
        mask.word_capacity = word_count - 1u;
        mask.location = location;
        cs::predicate_cache_run_result run{};
        require(!cs::run_sequence_regulatory_operation_cached(prepared, launch, {11u}, &mask, &run)
                && !mask.occupied && run.launches_enqueued == 0u,
            "short mask capacity changed or launched the cache");
        mask.word_capacity = word_count;

        // Uncached materialization, cache publication, and reuse all compute the same output.
        for (uint32_t i = 0; i < 3u; ++i) {
            cuda_require(cudaMemsetAsync(device_output, 0, sizeof(contribution), stream),
                "zero mask path output");
            if (i == 0u) {
                require(co::run_prepared_operation(prepared, launch), "launch mask path");
            } else {
                require(cs::run_sequence_regulatory_operation_cached(
                        prepared, launch, {11u}, &mask, &run)
                        && run.cache_hit == (i == 2u) && run.launches_enqueued == (i == 1u ? 2u : 1u),
                    "mask cache did not publish and reuse once");
            }
            float output[3]{};
            uint32_t words[word_count]{};
            cuda_require(cudaMemcpyAsync(output, device_output,
                sizeof(output), cudaMemcpyDeviceToHost, stream), "read mask path output");
            cuda_require(cudaMemcpyAsync(words, device_mask,
                sizeof(words), cudaMemcpyDeviceToHost, stream), "read materialized mask");
            cuda_require(cudaStreamSynchronize(stream), "finish mask path");
            compare_output(output, contribution, "materialized mask scalar parity failed");
            require(words[0] == ((1u << 5u) | (1u << 30u))
                    && words[1] == (1u << (58u - 32u)) && words[2] == 0u,
                "materialized mask lost seam, halo, or validity semantics");
        }
        require(cs::prepare_sequence_regulatory_operation(
                narrowed_request, policy, &state, &prepared),
            "narrowed mask reprepare was rejected");
        ce::launch_bindings narrowed_launch = launch;
        narrowed_launch.structures = narrowed_relations;
        cuda_require(cudaMemsetAsync(device_output, 0, sizeof(contribution), stream),
            "zero narrowed mask output");
        require(cs::run_sequence_regulatory_operation_cached(
                prepared, narrowed_launch, {11u}, &mask, &run)
                && !run.cache_hit && run.launches_enqueued == 2u
                && mask.key.coordinate_epoch.value == narrowed_relations[0].epoch.value,
            "narrowed mask reused an older coordinate epoch");
        float narrowed_output[3]{}, expected[3]{};
        uint32_t first_mask_word = 0u;
        scalar_reference(bases, narrowed_chunk, intervals, offsets, genes, weights, expected);
        cuda_require(cudaMemcpyAsync(narrowed_output, device_output,
            sizeof(narrowed_output), cudaMemcpyDeviceToHost, stream), "read narrowed mask output");
        cuda_require(cudaMemcpyAsync(&first_mask_word, device_mask,
            sizeof(first_mask_word), cudaMemcpyDeviceToHost, stream), "read narrowed first mask word");
        cuda_require(cudaStreamSynchronize(stream), "finish narrowed mask path");
        require(first_mask_word == (1u << 30u), "narrowed mask retained an unowned anchor");
        compare_output(narrowed_output, expected, "narrowed mask scalar parity failed");
        cuda_require(cudaEventDestroy(mask_ready), "destroy mask readiness event");
    }

    cs::prepared_sequence_state materialized_state{};
    co::prepared_operation materialized{};
    require(cs::prepare_sequence_regulatory_operation(
            request, materialized_policy,
            &materialized_state, &materialized),
        "materialized operation did not prepare");
    require(materialized_state.strategy
            == cs::sequence_strategy::materialize_relation,
        "reuse policy did not select direct relation materialization");
    require(materialized.binding_contract.output_effect_count == 2u
            && materialized.binding_contract.output_effects[0].update
                == ce::output_update_kind::accumulate
            && materialized.binding_contract.output_effects[1].update
                == ce::output_update_kind::overwrite
            && !materialized.binding_contract.output_effects[1]
                .requires_initialized_destination,
        "materialized output effects are incomplete");
    require(ce::same_handle(
                materialized.binding_contract.output_orders[1]
                    .output_axis.domain,
                coordinate_axis.domain)
            && ce::same_handle(
                materialized.binding_contract.output_orders[1]
                    .output_axis.order,
                coordinate_axis.order),
        "direct relation output lost sequence coordinate identity");
    ce::biological_operand_view materialized_outputs[2]{
        fused_output,
        output_view(device_relation, regulatory_relation_axis, location,
            ce::numeric_type::u32, base_count)};
    cuda_require(cudaMemsetAsync(device_output, 0, 3u * sizeof(float), stream),
        "zero materialized output");
    ce::launch_bindings materialized_launch = launch_bindings(
        relations, &input, materialized_outputs, 2u,
        &value_binding, stream, device);
    require(co::run_prepared_operation(materialized, materialized_launch),
        "materialized operation launch rejected");
    float materialized_result[3]{};
    uint32_t relation_result[base_count]{};
    cuda_require(cudaMemcpyAsync(materialized_result, device_output,
        sizeof(materialized_result), cudaMemcpyDeviceToHost, stream),
        "copy materialized output");
    cuda_require(cudaMemcpyAsync(relation_result, device_relation,
        sizeof(relation_result), cudaMemcpyDeviceToHost, stream),
        "copy direct regulatory relation");
    cuda_require(cudaStreamSynchronize(stream), "finish materialized path");
    compare_output(materialized_result, contribution,
        "materialized scalar parity failed");
    for (uint32_t anchor = 0u; anchor < base_count; ++anchor) {
        const uint32_t expected_element = anchor == 5u ? 0u
            : anchor == 30u ? 1u : anchor == 58u ? 2u : UINT32_MAX;
        require(relation_result[anchor] == expected_element,
            "direct relation identity, validity, halo, or tail mismatch");
    }

    cs::regulatory_relation_cache_entry cache{};
    cache.elements = device_relation;
    cache.ready_event = cache_ready;
    cache.element_capacity = base_count - 1u;
    cache.location = {ce::residency_kind::device, {}, device, 0u};
    cs::predicate_cache_run_result cache_result{};
    require(!cs::run_sequence_regulatory_relation_cached(
            materialized, materialized_launch, {11u}, &cache, &cache_result)
            && !cache.occupied && cache_result.launches_enqueued == 0u,
        "short relation capacity changed or launched the cache");
    cache.element_capacity = base_count;
    cuda_require(cudaMemsetAsync(
        device_output, 0, 3u * sizeof(float), stream),
        "zero first cached output");
    require(cs::run_sequence_regulatory_relation_cached(
            materialized, materialized_launch, {11u}, &cache,
            &cache_result)
            && !cache_result.cache_hit
            && cache_result.launches_enqueued == 2u,
        "first cached execution did not materialize once");
    float first_cached_result[3]{};
    cuda_require(cudaMemcpyAsync(first_cached_result,
        device_output, sizeof(first_cached_result),
        cudaMemcpyDeviceToHost, stream), "copy first cached output");
    cuda_require(cudaStreamSynchronize(stream), "finish first cached use");
    compare_output(first_cached_result, contribution,
        "first cached execution failed referee parity");

    cuda_require(cudaMemsetAsync(
        device_output, 0, 3u * sizeof(float), stream),
        "zero reused cached output");
    require(cs::run_sequence_regulatory_relation_cached(
            materialized, materialized_launch, {11u}, &cache,
            &cache_result)
            && cache_result.cache_hit
            && cache_result.launches_enqueued == 1u,
        "same generation/order/projection did not reuse the relation");
    float reused_cached_result[3]{};
    cuda_require(cudaMemcpyAsync(reused_cached_result,
        device_output, sizeof(reused_cached_result),
        cudaMemcpyDeviceToHost, stream), "copy reused cached output");
    cuda_require(cudaStreamSynchronize(stream), "finish reused cached use");
    compare_output(reused_cached_result, contribution,
        "reused cached execution failed referee parity");

    const float second_weights[3]{3.0f, 4.0f, 1.0f};
    float second_contribution[3]{};
    scalar_reference(bases, chunk, intervals, offsets, genes, second_weights, second_contribution);
    cuda_require(cudaMemcpyAsync(device_weights, second_weights,
        sizeof(second_weights), cudaMemcpyHostToDevice, stream),
        "upload second dynamic cell state");
    value_plane.generation = {2u};
    value_binding.expected_generation = {2u};
    size_t free_before_state = 0u, total_before_state = 0u;
    cuda_require(cudaMemGetInfo(&free_before_state, &total_before_state),
        "memory before reused cell state");
    cuda_require(cudaMemsetAsync(device_output, 0,
        3u * sizeof(float), stream), "zero second cell-state output");
    require(cs::run_sequence_regulatory_relation_cached(
            materialized, materialized_launch, {11u}, &cache,
            &cache_result)
            && cache_result.cache_hit
            && cache_result.launches_enqueued == 1u,
        "new regulatory values rebuilt static sequence relation");
    float second_state_result[3]{};
    cuda_require(cudaMemcpyAsync(second_state_result,
        device_output, sizeof(second_state_result),
        cudaMemcpyDeviceToHost, stream), "copy second cell-state output");
    cuda_require(cudaStreamSynchronize(stream), "finish second cell state");
    size_t free_after_state = 0u, total_after_state = 0u;
    cuda_require(cudaMemGetInfo(&free_after_state, &total_after_state),
        "memory after reused cell state");
    compare_output(second_state_result, second_contribution,
        "reused sequence relation failed second cell-state parity");
    require(free_before_state == free_after_state
            && total_before_state == total_after_state,
        "reused cell-state execution allocated steady-state storage");

    cuda_require(cudaMemsetAsync(device_output, 0, sizeof(second_state_result), stream),
        "zero graph replay output");
    cudaGraph_t graph = nullptr;
    cuda_require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "begin warmed reuse capture");
    ce::biological_operand_view rebound_input{};
    require(cs::bind_sequence_planes(planes, coordinate_axis, location, &rebound_input),
        "bind prepacked planes again");
    ce::launch_bindings warmed_launch = materialized_launch;
    warmed_launch.inputs = &rebound_input;
    const co::operation_status cold_capture =
        cs::run_sequence_regulatory_relation_cached(
            materialized, warmed_launch, {12u}, &cache, &cache_result);
    require(cold_capture.code == co::operation_status_code::capability_rejected
            && !cache_result.cache_hit && cache_result.launches_enqueued == 0u
            && cache.occupied && cache.key.sequence_generation.value == 11u,
        "capture published a cold relation or discarded the warm cache");
    const co::operation_status captured = cs::run_sequence_regulatory_relation_cached(
        materialized, warmed_launch, {11u}, &cache, &cache_result);
    require(captured, captured.message);
    require(cache_result.cache_hit && cache_result.launches_enqueued == 1u,
        "warmed capture did not reuse the published relation");
    cuda_require(cudaStreamEndCapture(stream, &graph), "end warmed reuse capture");
    cudaGraphNode_t nodes[2]{};
    size_t node_count = 0u;
    cuda_require(cudaGraphGetNodes(graph, nullptr, &node_count), "size warmed capture");
    require(node_count == 2u, "warmed capture must have one kernel and one wait");
    cuda_require(cudaGraphGetNodes(graph, nodes, &node_count), "inspect warmed capture");
    size_t kernels = 0u;
    for (size_t i = 0u; i < node_count; ++i) {
        cudaGraphNodeType type{};
        cuda_require(cudaGraphNodeGetType(nodes[i], &type), "inspect reuse node");
        require(type == cudaGraphNodeTypeKernel || type == cudaGraphNodeTypeWaitEvent,
            "warmed capture contains unexpected work");
        kernels += type == cudaGraphNodeTypeKernel;
    }
    require(kernels == 1u, "warmed capture did not contain exactly one accumulation");
    cudaGraphExec_t replay = nullptr;
    cuda_require(cudaGraphInstantiateWithFlags(&replay, graph, 0u), "instantiate warmed capture");
    cuda_require(cudaGraphLaunch(replay, stream), "replay warmed capture");
    float graph_result[3]{};
    cuda_require(cudaMemcpyAsync(graph_result, device_output,
        sizeof(graph_result), cudaMemcpyDeviceToHost, stream), "copy replay output");
    cuda_require(cudaStreamSynchronize(stream), "finish graph replay");
    compare_output(graph_result, second_contribution, "warmed graph replay failed scalar parity");
    cuda_require(cudaGraphExecDestroy(replay), "destroy warmed replay");
    cuda_require(cudaGraphDestroy(graph), "destroy warmed reuse capture");

    cuda_require(cudaMemsetAsync(device_output, 0,
        3u * sizeof(float), stream), "zero next-sequence output");
    require(cs::run_sequence_regulatory_relation_cached(
            materialized, materialized_launch, {12u}, &cache,
            &cache_result)
            && !cache_result.cache_hit
            && cache_result.launches_enqueued == 2u,
        "new sequence generation reused a stale relation");
    cuda_require(cudaStreamSynchronize(stream), "finish new sequence use");

    cs::sequence_prepare_request reprojected_request = request;
    reprojected_request.persistent_projection = {173u, 174u};
    cs::sequence_prepare_policy forced_relation{};
    forced_relation.requested = cs::sequence_strategy::materialize_relation;
    cs::prepared_sequence_state reprojected_state{};
    co::prepared_operation reprojected{};
    require(cs::prepare_sequence_regulatory_operation(
            reprojected_request, forced_relation,
            &reprojected_state, &reprojected),
        "alternate regulatory projection did not prepare");
    require(cs::run_sequence_regulatory_relation_cached(
            reprojected, materialized_launch, {12u}, &cache,
            &cache_result)
            && !cache_result.cache_hit
            && cellerator::structure::same_identity(cache.key.regulatory_projection,
                reprojected_request.persistent_projection),
        "regulatory projection change reused a stale direct relation");
    cuda_require(cudaStreamSynchronize(stream),
        "finish reprojected relation use");

    cs::sequence_prepare_request reordered_request = request;
    reordered_request.persistent_coordinate_order = {79u, 80u};
    cs::prepared_sequence_state reordered_state{};
    co::prepared_operation reordered{};
    cs::sequence_prepare_policy forced_materialization{};
    forced_materialization.requested =
        cs::sequence_strategy::materialize_relation;
    require(cs::prepare_sequence_regulatory_operation(
            reordered_request, forced_materialization,
            &reordered_state, &reordered),
        "alternate persistent coordinate order did not prepare");
    require(cs::run_sequence_regulatory_relation_cached(
            reordered, materialized_launch, {12u}, &cache,
            &cache_result)
            && !cache_result.cache_hit
            && cellerator::structure::same_identity(cache.key.coordinate_order,
                reordered_request.persistent_coordinate_order),
        "coordinate-order change reused a stale relation");
    cuda_require(cudaStreamSynchronize(stream), "finish reordered use");

    uint32_t *replacement_relation = allocate_device<uint32_t>(base_count);
    ce::biological_operand_view rebound_outputs[2]{
        materialized_outputs[0], materialized_outputs[1]};
    rebound_outputs[1].storage.dense.data = replacement_relation;
    ce::launch_bindings rebound_launch = launch_bindings(
        relations, &input, rebound_outputs, 2u,
        &value_binding, stream, device);
    require(!cs::run_sequence_regulatory_relation_cached(
            reordered, rebound_launch, {12u}, &cache, &cache_result),
        "occupied cache silently rebound by pointer identity");

    narrowed_request.persistent_coordinate_order = reordered_request.persistent_coordinate_order;
    require(cs::prepare_sequence_regulatory_operation(
            narrowed_request, forced_materialization, &reordered_state, &reordered),
        "narrowed direct relation reprepare was rejected");
    ce::launch_bindings narrowed_launch = materialized_launch;
    narrowed_launch.structures = narrowed_relations;
    cuda_require(cudaMemsetAsync(device_output, 0, sizeof(contribution), stream),
        "zero narrowed direct relation output");
    require(cs::run_sequence_regulatory_relation_cached(
            reordered, narrowed_launch, {12u}, &cache, &cache_result)
            && !cache_result.cache_hit && cache_result.launches_enqueued == 2u
            && cache.key.coordinate_epoch.value == narrowed_relations[0].epoch.value,
        "narrowed direct relation reused an older coordinate epoch");
    float narrowed_output[3]{}, narrowed_expected[3]{};
    uint32_t excluded_element = 0u;
    scalar_reference(bases, narrowed_chunk, intervals, offsets, genes, second_weights, narrowed_expected);
    cuda_require(cudaMemcpyAsync(narrowed_output, device_output,
        sizeof(narrowed_output), cudaMemcpyDeviceToHost, stream), "read narrowed direct relation output");
    cuda_require(cudaMemcpyAsync(&excluded_element, device_relation + 5u,
        sizeof(excluded_element), cudaMemcpyDeviceToHost, stream), "read excluded regulatory element");
    cuda_require(cudaStreamSynchronize(stream), "finish narrowed direct relation path");
    require(excluded_element == UINT32_MAX, "narrowed direct relation retained an unowned anchor");
    compare_output(narrowed_output, narrowed_expected, "narrowed direct relation scalar parity failed");

    cs::sequence_strategy_evidence stale_evidence = evidence;
    stale_evidence.key.coordinate_order = {99u, 100u};
    cs::sequence_prepare_policy stale_policy = fused_policy;
    stale_policy.evidence = &stale_evidence;
    require(!cs::prepare_sequence_regulatory_operation(
            request, stale_policy, &rejected_state, &rejected),
        "stale measurement identity selected a strategy");

    const cs::sequence_execution_accounting fused_bytes =
        cs::sequence_accounting(fused_state, base_count);
    const cs::sequence_execution_accounting materialized_bytes =
        cs::sequence_accounting(materialized_state, base_count);
    require(fused_bytes.launch_count == 1u
            && fused_bytes.materialized_mask_bytes == 0u
            && materialized_bytes.launch_count == 2u
            && materialized_bytes.materialized_mask_bytes == 0u
            && materialized_bytes.materialized_relation_bytes
                == base_count * sizeof(uint32_t),
        "strategy accounting is incomplete");

    ce::biological_operand_view stale_input = input;
    stale_input.storage.bits.coordinate_axis.order.slot += 1u;
    ce::launch_bindings stale_launch = launch_bindings(
        relations, &stale_input, &fused_output, 1u,
        &value_binding, stream, device);
    const co::operation_status stale_status =
        co::run_prepared_operation(fused, stale_launch);
    require(!stale_status
            && stale_status.binding
                == ce::binding_validation_code::operand_axis_mismatch,
        "stale sequence execution order was accepted");

    sq::prepared_predicate_plan stale_plan = predicate_plan;
    ++stale_plan.semantic_hash;
    cs::sequence_prepare_request stale_request = request;
    stale_request.predicate_plan = &stale_plan;
    require(!cs::prepare_sequence_regulatory_operation(
            stale_request, fused_policy, &rejected_state, &rejected),
        "stale sequence semantic hash was accepted");

    sq::sequence_predicate_program reverse_program = program;
    reverse_program.outputs[0].flags = sq::sequence_event_reverse_strand;
    sq::prepared_predicate_plan reverse_plan{};
    require(sq::prepare_sequence_predicate_program(
            reverse_program, &reverse_plan) == sq::predicate_plan_status::ok,
        "sequence reverse fixture did not prepare");
    cs::sequence_prepare_request reverse_request = request;
    reverse_request.program = &reverse_program;
    reverse_request.predicate_plan = &reverse_plan;
    require(!cs::prepare_sequence_regulatory_operation(
            reverse_request, fused_policy, &rejected_state, &rejected),
        "unsupported reverse-strand primitive was silently accepted");

    if (benchmark) {
        const timing_summary fused_timing = measure_candidate(stream,
            [&](uint32_t) {
                return static_cast<bool>(
                    co::run_prepared_operation(fused, fused_launch));
            }, "fused benchmark launch failed");
        const timing_summary first_materialized_timing = measure_candidate(stream,
            [&](uint32_t sample) {
                cache.occupied = false;
                return static_cast<bool>(
                    cs::run_sequence_regulatory_relation_cached(
                        materialized, materialized_launch,
                        {1000u + sample}, &cache, &cache_result));
            }, "first materialized benchmark launch failed");
        cache.occupied = false;
        require(cs::run_sequence_regulatory_relation_cached(
                materialized, materialized_launch, {2000u}, &cache,
                &cache_result),
            "cached benchmark preparation failed");
        cuda_require(cudaStreamSynchronize(stream),
            "finish cached benchmark preparation");
        const timing_summary cached_timing = measure_candidate(stream,
            [&](uint32_t) {
                return static_cast<bool>(
                    cs::run_sequence_regulatory_relation_cached(
                        materialized, materialized_launch, {2000u}, &cache,
                        &cache_result));
            }, "cached materialized benchmark launch failed");

        cudaDeviceProp properties{};
        cuda_require(cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
        const ce::device_performance_class actual_device{
            0x10deu, static_cast<uint16_t>(properties.major),
            static_cast<uint16_t>(properties.minor), 0xce780002u};
        const cs::sequence_measurement_key actual_key{
            predicate_plan.semantic_hash, request.persistent_coordinate_order,
            request.persistent_projection, actual_device, 0xce780002u,
            base_count, program.outputs[0].predicate_id,
            program.outputs[0].flags};
        const cs::sequence_strategy_evidence actual_evidence{
            actual_key, fused_timing.median_ns,
            first_materialized_timing.median_ns, cached_timing.median_ns,
            fused_timing.spread_percent,
            first_materialized_timing.spread_percent > cached_timing.spread_percent
                ? first_materialized_timing.spread_percent : cached_timing.spread_percent,
            9u, 0u};
        uint32_t first_materialized_state_count = 0u;
        for (uint32_t reuse = 1u; reuse <= 64u; ++reuse) {
            cs::sequence_prepare_policy measured_policy{};
            measured_policy.expected_cell_state_count = reuse;
            measured_policy.evidence = &actual_evidence;
            measured_policy.device = actual_device;
            measured_policy.runtime_build_identity = 0xce780002u;
            measured_policy.maximum_spread_percent = 100.0;
            const cs::sequence_strategy_decision measured_decision =
                cs::select_sequence_strategy(actual_key, measured_policy);
            require(!measured_decision.empirical_measurement_required,
                "fresh measured evidence was rejected");
            if (first_materialized_state_count == 0u
                && measured_decision.strategy
                    == cs::sequence_strategy::materialize_relation)
                first_materialized_state_count = reuse;
        }

        printf("sequence_regulatory_timing device=%s cc=%d.%d bases=%u"
               " samples=9 warmups=4 uses_per_sample=100"
               " fused_ns=%.9g fused_spread_pct=%.9g"
               " first_materialized_ns=%.9g first_materialized_spread_pct=%.9g"
               " cached_ns=%.9g cached_spread_pct=%.9g first_materialized_state_count=%u\n",
            properties.name, properties.major, properties.minor, base_count,
            fused_timing.median_ns, fused_timing.spread_percent,
            first_materialized_timing.median_ns, first_materialized_timing.spread_percent,
            cached_timing.median_ns, cached_timing.spread_percent, first_materialized_state_count);

    }

    cuda_require(cudaEventDestroy(cache_ready), "destroy relation readiness event");
    cuda_require(cudaStreamDestroy(stream), "destroy fixture stream");
    cuda_require(cudaFree(replacement_relation), "free replacement relation");
    cuda_require(cudaFree(device_output), "free gene output");
    cuda_require(cudaFree(device_weights), "free weights");
    cuda_require(cudaFree(device_genes), "free genes");
    cuda_require(cudaFree(device_offsets), "free offsets");
    cuda_require(cudaFree(device_intervals), "free intervals");
    cuda_require(cudaFree(device_relation), "free materialized relation");
    cuda_require(cudaFree(device_mask), "free materialized mask");
    cuda_require(cudaFree(device_validity), "free validity plane");
    cuda_require(cudaFree(device_high), "free high plane");
    cuda_require(cudaFree(device_low), "free low plane");
    cuda_require(cudaFree(device_source_validity), "free packed validity");
    cuda_require(cudaFree(device_packed), "free packed sequence");
    printf("celleratorSequenceRegulatoryTest passed"
           " predicate_hash=%llu fused_launches=%u materialized_launches=%u\n",
        static_cast<unsigned long long>(predicate_plan.semantic_hash),
        fused_bytes.launch_count, materialized_bytes.launch_count);
    return 0;
}
