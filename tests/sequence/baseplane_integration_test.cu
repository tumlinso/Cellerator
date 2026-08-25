/*
CE-ARCH-89 CUDA evidence (2026-08-25, V100 sm_70, evidence
c9162454-63ad-462f-bc61-c65141d44a2e): for 70 validity-aware bases, the complete
fused path measured 3.645 us/state, direct relation first use 5.386 us, and
cached relation consumption 2.775 us. Nine samples x 100 states after four
warmups place the 2%-tolerance crossover at four dynamic cell states. Command:
cuda_controller.py run --spec /tmp/ce_arch_89_cuda_spec.json --json. Both paths
match the scalar referee at 1e-6 and include all per-state launches.
*/

#include <Cellerator/compute/sequence/baseplane_integration.cuh>

#include <Baseplane/seq/dna2_scan.cuh>
#include <Cellerator/execution/validation.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs = cellerator::compute::sequence;
namespace ce = cellerator::execution;
namespace co = cellerator::compute::math::core;
namespace bp = baseplane::seq;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char *message) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(message) + ": "
            + cudaGetErrorString(status));
}

template<typename T>
class device_buffer {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        if (count != 0u)
            cuda_require(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (data_ != nullptr) cudaFree(data_); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    T *get() const { return data_; }
    std::size_t count() const { return count_; }

private:
    T *data_ = nullptr;
    std::size_t count_ = 0u;
};

ce::axis_identity axis(
    ce::u32 domain, ce::u32 order, ce::u32 geometry, ce::u32 partition) {
    return {{domain, 1u}, {order, 1u}, {geometry, 1u}, {partition, 1u}};
}

void set_sequence_base(
    std::vector<std::uint64_t> &words,
    std::vector<std::uint32_t> &validity,
    std::uint32_t position,
    char base) {
    const bp::dna2_encoded_base encoded = bp::dna2_encode_base_with_validity(base);
    bp::dna2_word64 word{words[position / 32u]};
    bp::set_base(word, static_cast<int>(position % 32u), encoded.code);
    words[position / 32u] = word.packed;
    if (encoded.valid) validity[position / 32u] |= 1u << (position % 32u);
    else validity[position / 32u] &= ~(1u << (position % 32u));
}

void place_motif(
    std::vector<std::uint64_t> &words,
    std::vector<std::uint32_t> &validity,
    std::uint32_t position,
    bool invalidate_middle = false) {
    set_sequence_base(words, validity, position, 'A');
    set_sequence_base(words, validity, position + 1u,
        invalidate_middle ? 'N' : 'C');
    if (invalidate_middle) {
        bp::dna2_word64 word{words[(position + 1u) / 32u]};
        bp::set_base(word, static_cast<int>((position + 1u) % 32u), 1u);
        words[(position + 1u) / 32u] = word.packed;
    }
    set_sequence_base(words, validity, position + 2u, 'G');
}

bp::sequence_predicate_program make_program() {
    bp::sequence_predicate_program program{};
    program.version = bp::sequence_program_version;
    program.node_count = 1u;
    program.output_count = 1u;
    program.exact_motif_count = 1u;
    bp::dna2_word64 motif{};
    bp::set_base(motif, 0, 0u);
    bp::set_base(motif, 1, 1u);
    bp::set_base(motif, 2, 2u);
    require(bp::dna2_normalize_motif32_exact(
        motif, 3u, 0u, 7u, &program.exact_motifs[0]),
        "failed to normalize test motif");
    program.nodes[0] = bp::predicate_instruction{
        bp::predicate_opcode::exact_motif, bp::predicate_value_kind::mask,
        0u, 0u, bp::sequence_program_no_input,
        bp::sequence_program_no_input, 0u, 0u};
    program.outputs[0] = bp::predicate_output{
        0u, 7u, bp::sequence_output_mode::mask, 0u,
        bp::sequence_event_forward_strand, 0u};
    return program;
}

std::array<float, 3> scalar_reference(
    const std::vector<std::uint64_t> &words,
    const std::vector<std::uint32_t> &validity,
    const bp::dna2_chunk_coordinates &chunk,
    const bp::motif32_exact &motif,
    const std::array<cs::regulatory_interval, 3> &intervals,
    const std::array<std::uint32_t, 4> &offsets,
    const std::array<std::uint32_t, 3> &genes,
    const std::array<float, 3> &weights) {
    std::array<float, 3> result{};
    for (std::uint32_t anchor = chunk.owned_begin;
         anchor < chunk.owned_end; ++anchor) {
        if (motif.length > chunk.base_count - anchor) continue;
        bool valid = true;
        bp::dna2_word64 window{};
        for (std::uint32_t lane = 0u; lane < motif.length; ++lane) {
            const std::uint32_t position = anchor + lane;
            valid = valid && ((validity[position / 32u]
                >> (position % 32u)) & 1u) != 0u;
            bp::set_base(window, static_cast<int>(lane), bp::get_base(
                bp::dna2_word64{words[position / 32u]},
                static_cast<int>(position % 32u)));
        }
        if (!valid || bp::word64_mismatches(
                window, bp::dna2_word64{motif.packed},
                (1u << motif.length) - 1u) > motif.max_mismatches) continue;
        for (const cs::regulatory_interval &interval : intervals) {
            if (anchor < interval.begin || anchor >= interval.end
                || interval.predicate_id != 7u) continue;
            for (std::uint32_t edge = offsets[interval.regulatory_element];
                 edge < offsets[interval.regulatory_element + 1u]; ++edge)
                result[genes[edge]] += weights[edge];
            break;
        }
    }
    return result;
}

ce::dense_tensor_view gene_output(
    float *data,
    ce::axis_identity gene_axis,
    ce::device_location location) {
    ce::dense_tensor_view output{};
    output.data = data;
    output.location = location;
    output.value_type = ce::numeric_type::f32;
    output.rank = 1u;
    output.axes[0] = gene_axis;
    output.shape[0] = 3u;
    output.stride[0] = 1;
    return output;
}

ce::launch_bindings launch_bindings(
    const ce::relation_structure *structures,
    const ce::biological_operand_view *input,
    ce::biological_operand_view *outputs,
    std::uint32_t output_count,
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
    const std::array<float, 3> &actual,
    const std::array<float, 3> &expected,
    const char *path) {
    for (std::size_t index = 0u; index < actual.size(); ++index)
        require(std::fabs(actual[index] - expected[index]) < 1e-6f, path);
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
    constexpr std::uint32_t warmups = 4u;
    constexpr std::uint32_t samples = 9u;
    constexpr std::uint32_t uses_per_sample = 100u;
    for (std::uint32_t index = 0u;
         index < warmups * uses_per_sample; ++index)
        require(enqueue(index), label);
    cuda_require(cudaStreamSynchronize(stream), "finish benchmark warmup");
    cudaEvent_t begin = nullptr, end = nullptr;
    cuda_require(cudaEventCreate(&begin), "cudaEventCreate begin");
    cuda_require(cudaEventCreate(&end), "cudaEventCreate end");
    std::array<double, samples> elapsed{};
    for (std::uint32_t index = 0u; index < samples; ++index) {
        cuda_require(cudaEventRecord(begin, stream), "record benchmark begin");
        for (std::uint32_t use = 0u; use < uses_per_sample; ++use)
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
    std::sort(elapsed.begin(), elapsed.end());
    const double median = elapsed[samples / 2u];
    std::array<double, samples> deviation{};
    for (std::uint32_t index = 0u; index < samples; ++index)
        deviation[index] = std::fabs(elapsed[index] - median);
    std::sort(deviation.begin(), deviation.end());
    return {median, median == 0.0 ? 0.0
        : deviation[samples / 2u] * 100.0 / median};
}

} // namespace

int main() {
    try {
        int device_count = 0;
        cuda_require(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device_count > 0, "no CUDA device available");
        int device = 0;
        cuda_require(cudaGetDevice(&device), "cudaGetDevice");
        cudaStream_t stream = nullptr;
        cuda_require(cudaStreamCreate(&stream), "cudaStreamCreate");
        cudaEvent_t cache_ready = nullptr;
        cuda_require(cudaEventCreateWithFlags(
            &cache_ready, cudaEventDisableTiming), "cudaEventCreate cache_ready");

        constexpr std::uint32_t base_count = 70u;
        constexpr std::uint32_t word_count = 3u;
        std::vector<std::uint64_t> packed(word_count, ~std::uint64_t{0});
        std::vector<std::uint32_t> validity(word_count, 0xffffffffu);
        validity.back() = (1u << (base_count - 64u)) - 1u;
        place_motif(packed, validity, 0u);
        place_motif(packed, validity, 5u);
        place_motif(packed, validity, 30u);
        place_motif(packed, validity, 40u, true);
        place_motif(packed, validity, 62u);

        const bp::sequence_predicate_program program = make_program();
        bp::prepared_predicate_plan baseplane_plan{};
        require(bp::prepare_sequence_predicate_program(program, &baseplane_plan)
                == bp::predicate_plan_status::ok,
            "Baseplane plan preparation failed");
        bp::dna2_chunk_coordinates chunk{
            {17u, 2u, 9u}, 1ull << 40u, base_count,
            4u, 60u, 4u, 10u};
        ce::sequence_domain source_domain{};
        require(cs::adapt_baseplane_chunk(chunk, {41u, 1u}, &source_domain),
            "chunk adapter rejected bounded coordinates");
        bp::dna2_chunk_coordinates oversized_chunk = chunk;
        oversized_chunk.identity.contig =
            std::numeric_limits<std::uint64_t>::max();
        require(!cs::adapt_baseplane_chunk(
                oversized_chunk, {41u, 1u}, &source_domain),
            "unrepresentable contig identity was truncated");

        const ce::axis_identity coordinate_axis = axis(41u, 42u, 43u, 44u);
        const ce::axis_identity regulatory_axis = axis(46u, 47u, 48u, 49u);
        const ce::axis_identity predicate_mask_axis = axis(41u, 42u, 45u, 44u);
        const ce::axis_identity regulatory_relation_axis =
            axis(41u, 42u, 50u, 44u);
        const ce::axis_identity gene_axis = axis(51u, 52u, 53u, 54u);
        const std::array<cs::regulatory_interval, 3> intervals{{
            {4u, 20u, 0u, 7u, 0u},
            {20u, 50u, 1u, 7u, 0u},
            {50u, 60u, 2u, 7u, 0u}}};
        const std::array<std::uint32_t, 4> offsets{{0u, 1u, 3u, 3u}};
        const std::array<std::uint32_t, 3> genes{{2u, 0u, 1u}};
        const std::array<float, 3> weights{{1.5f, 2.0f, 0.5f}};
        const cs::regulatory_projection_view host_projection{
            intervals.data(), offsets.data(), genes.data(), 3u, 3u, 3u, 3u,
            {ce::residency_kind::host, {}, -1, 0u}};
        require(cs::validate_regulatory_projection_host(host_projection, base_count),
            "valid host regulatory projection rejected");
        auto overlapping = intervals;
        overlapping[1].begin = 19u;
        require(!cs::validate_regulatory_projection_host(
                {overlapping.data(), offsets.data(), genes.data(),
                 3u, 3u, 3u, 3u,
                 {ce::residency_kind::host, {}, -1, 0u}}, base_count),
            "overlapping interval projection accepted");

        device_buffer<std::uint64_t> device_packed(word_count);
        device_buffer<std::uint32_t> device_low(word_count),
            device_high(word_count), device_validity(word_count),
            device_mask(word_count), device_relation(base_count),
            device_offsets(offsets.size()),
            device_genes(genes.size());
        device_buffer<cs::regulatory_interval> device_intervals(intervals.size());
        device_buffer<float> device_weights(weights.size()), device_output(3u);
        cuda_require(cudaMemcpyAsync(device_packed.get(), packed.data(),
            packed.size() * sizeof(packed[0]), cudaMemcpyHostToDevice, stream),
            "copy packed sequence");
        cuda_require(cudaMemcpyAsync(device_validity.get(), validity.data(),
            validity.size() * sizeof(validity[0]), cudaMemcpyHostToDevice, stream),
            "copy validity");
        cuda_require(cudaMemcpyAsync(device_intervals.get(), intervals.data(),
            sizeof(intervals), cudaMemcpyHostToDevice, stream), "copy intervals");
        cuda_require(cudaMemcpyAsync(device_offsets.get(), offsets.data(),
            sizeof(offsets), cudaMemcpyHostToDevice, stream), "copy offsets");
        cuda_require(cudaMemcpyAsync(device_genes.get(), genes.data(),
            sizeof(genes), cudaMemcpyHostToDevice, stream), "copy genes");
        cuda_require(cudaMemcpyAsync(device_weights.get(), weights.data(),
            sizeof(weights), cudaMemcpyHostToDevice, stream), "copy weights");
        std::uint64_t scalar_hit_count = 0u;
        require(baseplane::is_ok(bp::scan_exact_count_cpu(
                {packed.data(), base_count, word_count}, program.exact_motifs[0],
                &scalar_hit_count)),
            "Baseplane scalar exact scanner rejected fixture");
        require(baseplane::is_ok(bp::dna2_to_planes32_stream_cuda(
                stream, {device_packed.get(), base_count, word_count},
                {device_low.get(), device_high.get(), word_count})),
            "Baseplane device plane conversion failed");
        ce::biological_operand_view input{};
        require(cs::adapt_baseplane_planes(
                {{device_low.get(), device_high.get(), word_count},
                 device_validity.get(), base_count}, coordinate_axis,
                bp::sequence_buffer_residency::device, device, &input),
            "Baseplane device plane adapter failed");
        ce::biological_operand_view missing_validity{};
        require(!cs::adapt_baseplane_planes(
                {{device_low.get(), device_high.get(), word_count},
                 nullptr, base_count}, coordinate_axis,
                bp::sequence_buffer_residency::device, device,
                &missing_validity),
            "implicit nonempty validity was accepted at the Cellerator seam");
        ce::relation_structure relations[2]{
            {{61u, 1u}, {1u}, coordinate_axis, regulatory_axis,
                {62u, 1u}, 3u},
            {{64u, 1u}, {2u}, regulatory_axis, gene_axis,
                {65u, 1u}, 3u}};
        const cs::regulatory_projection_view device_projection{
            device_intervals.get(), device_offsets.get(), device_genes.get(),
            3u, 3u, 3u, 3u,
            {ce::residency_kind::device, {}, device, 0u}};
        cs::sequence_prepare_request request{};
        request.program = &program;
        request.baseplane_plan = &baseplane_plan;
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
            relations[1].identity, relations[1].epoch, device_weights.get(),
            {ce::residency_kind::device, {}, device, 0u},
            {ce::numeric_type::f32, ce::numeric_type::f32,
             ce::numeric_type::f32, 0u},
            {ce::quantization_kind::none, ce::numeric_type::invalid,
             ce::numeric_type::invalid, 0u, nullptr, nullptr, 0u},
            ce::value_layout_kind::logical_edge_order, {}, {1u}, 3u,
            3u * sizeof(float)};
        ce::value_binding value_binding{&value_plane, {1u}};

        const std::array<float, 3> contribution = scalar_reference(
            packed, validity, chunk, program.exact_motifs[0],
            intervals, offsets, genes, weights);
        require(contribution[0] == 2.0f && contribution[1] == 0.5f
                && contribution[2] == 1.5f,
            "scalar fixture does not cover expected regulatory mapping");

        const ce::device_performance_class measurement_device{
            1u, 7u, 0u, 0x5a17u};
        const cs::sequence_measurement_key measurement_key{
            baseplane_plan.semantic_hash, request.persistent_coordinate_order,
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
        require(static_cast<bool>(cs::prepare_sequence_regulatory_operation(
                request, fused_policy,
                &fused_state, &fused)),
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
                wrong_coordinate_relation, {}, &rejected_state, &rejected),
            "coordinate-to-regulatory relation with wrong biology was accepted");
        cs::sequence_prepare_request wrong_regulatory_relation = request;
        wrong_regulatory_relation.regulatory_to_gene.source_axis =
            axis(46u, 47u, 48u, 149u);
        require(!cs::prepare_sequence_regulatory_operation(
                wrong_regulatory_relation, {}, &rejected_state, &rejected),
            "regulatory-to-gene relation with wrong biology was accepted");
        ce::biological_operand_view fused_output{};
        fused_output.kind = ce::operand_kind::dense_tensor;
        fused_output.storage.dense = gene_output(device_output.get(), gene_axis,
            {ce::residency_kind::device, {}, device, 0u});
        const std::array<float, 3> initial{{10.0f, 20.0f, 30.0f}};
        std::array<float, 3> accumulated_expected{};
        for (std::size_t index = 0u; index < initial.size(); ++index)
            accumulated_expected[index] = initial[index] + contribution[index];
        cuda_require(cudaMemcpyAsync(device_output.get(), initial.data(),
            sizeof(initial), cudaMemcpyHostToDevice, stream),
            "initialize fused accumulated output");
        ce::launch_bindings fused_launch = launch_bindings(
            relations, &input, &fused_output, 1u, &value_binding, stream, device);
        require(static_cast<bool>(co::run_prepared_operation(fused, fused_launch)),
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
        std::array<float, 3> fused_result{};
        cuda_require(cudaMemcpyAsync(fused_result.data(), device_output.get(),
            sizeof(fused_result), cudaMemcpyDeviceToHost, stream),
            "copy fused output");
        cuda_require(cudaStreamSynchronize(stream), "finish fused path");
        compare_output(fused_result, accumulated_expected,
            "fused accumulation did not preserve initial gene state");

        cs::prepared_sequence_state materialized_state{};
        co::prepared_operation materialized{};
        require(static_cast<bool>(cs::prepare_sequence_regulatory_operation(
                request, materialized_policy,
                &materialized_state, &materialized)),
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
            "direct relation output lost Baseplane coordinate identity");
        ce::biological_operand_view materialized_outputs[2]{};
        materialized_outputs[0].kind = ce::operand_kind::dense_tensor;
        materialized_outputs[0].storage.dense = gene_output(
            device_output.get(), gene_axis,
            {ce::residency_kind::device, {}, device, 0u});
        materialized_outputs[1].kind = ce::operand_kind::dense_tensor;
        materialized_outputs[1].storage.dense = ce::dense_tensor_view{};
        materialized_outputs[1].storage.dense.data = device_relation.get();
        materialized_outputs[1].storage.dense.location =
            {ce::residency_kind::device, {}, device, 0u};
        materialized_outputs[1].storage.dense.value_type = ce::numeric_type::u32;
        materialized_outputs[1].storage.dense.rank = 1u;
        materialized_outputs[1].storage.dense.axes[0] = regulatory_relation_axis;
        materialized_outputs[1].storage.dense.shape[0] = base_count;
        materialized_outputs[1].storage.dense.stride[0] = 1;
        cuda_require(cudaMemsetAsync(device_output.get(), 0, 3u * sizeof(float), stream),
            "zero materialized output");
        ce::launch_bindings materialized_launch = launch_bindings(
            relations, &input, materialized_outputs, 2u,
            &value_binding, stream, device);
        require(static_cast<bool>(
                co::run_prepared_operation(materialized, materialized_launch)),
            "materialized operation launch rejected");
        std::array<float, 3> materialized_result{};
        std::array<std::uint32_t, base_count> relation_result{};
        cuda_require(cudaMemcpyAsync(materialized_result.data(), device_output.get(),
            sizeof(materialized_result), cudaMemcpyDeviceToHost, stream),
            "copy materialized output");
        cuda_require(cudaMemcpyAsync(relation_result.data(), device_relation.get(),
            sizeof(relation_result), cudaMemcpyDeviceToHost, stream),
            "copy direct regulatory relation");
        cuda_require(cudaStreamSynchronize(stream), "finish materialized path");
        compare_output(materialized_result, contribution,
            "materialized scalar parity failed");
        for (std::uint32_t anchor = 0u; anchor < base_count; ++anchor) {
            const std::uint32_t expected_element = anchor == 5u ? 0u
                : anchor == 30u ? 1u : 0xffffffffu;
            require(relation_result[anchor] == expected_element,
                "direct relation identity, validity, halo, or tail mismatch");
        }

        cs::regulatory_relation_cache_entry cache{};
        cache.elements = device_relation.get();
        cache.ready_event = cache_ready;
        cache.element_capacity = base_count;
        cache.location = {ce::residency_kind::device, {}, device, 0u};
        cs::predicate_cache_run_result cache_result{};
        cuda_require(cudaMemsetAsync(
            device_output.get(), 0, 3u * sizeof(float), stream),
            "zero first cached output");
        require(static_cast<bool>(cs::run_sequence_regulatory_relation_cached(
                materialized, materialized_launch, {11u}, &cache,
                &cache_result))
                && !cache_result.cache_hit
                && cache_result.launches_enqueued == 2u,
            "first cached execution did not materialize once");
        std::array<float, 3> first_cached_result{};
        cuda_require(cudaMemcpyAsync(first_cached_result.data(),
            device_output.get(), sizeof(first_cached_result),
            cudaMemcpyDeviceToHost, stream), "copy first cached output");
        cuda_require(cudaStreamSynchronize(stream), "finish first cached use");
        compare_output(first_cached_result, contribution,
            "first cached execution failed referee parity");

        cuda_require(cudaMemsetAsync(
            device_output.get(), 0, 3u * sizeof(float), stream),
            "zero reused cached output");
        require(static_cast<bool>(cs::run_sequence_regulatory_relation_cached(
                materialized, materialized_launch, {11u}, &cache,
                &cache_result))
                && cache_result.cache_hit
                && cache_result.launches_enqueued == 1u,
            "same generation/order/projection did not reuse the relation");
        std::array<float, 3> reused_cached_result{};
        cuda_require(cudaMemcpyAsync(reused_cached_result.data(),
            device_output.get(), sizeof(reused_cached_result),
            cudaMemcpyDeviceToHost, stream), "copy reused cached output");
        cuda_require(cudaStreamSynchronize(stream), "finish reused cached use");
        compare_output(reused_cached_result, contribution,
            "reused cached execution failed referee parity");

        const std::array<float, 3> second_weights{{3.0f, 4.0f, 1.0f}};
        const std::array<float, 3> second_contribution{{4.0f, 1.0f, 3.0f}};
        cuda_require(cudaMemcpyAsync(device_weights.get(), second_weights.data(),
            sizeof(second_weights), cudaMemcpyHostToDevice, stream),
            "upload second dynamic cell state");
        value_plane.generation = {2u};
        value_binding.expected_generation = {2u};
        std::size_t free_before_state = 0u, total_before_state = 0u;
        cuda_require(cudaMemGetInfo(&free_before_state, &total_before_state),
            "memory before reused cell state");
        cuda_require(cudaMemsetAsync(device_output.get(), 0,
            3u * sizeof(float), stream), "zero second cell-state output");
        require(static_cast<bool>(cs::run_sequence_regulatory_relation_cached(
                materialized, materialized_launch, {11u}, &cache,
                &cache_result))
                && cache_result.cache_hit
                && cache_result.launches_enqueued == 1u,
            "new regulatory values rebuilt static sequence relation");
        std::array<float, 3> second_state_result{};
        cuda_require(cudaMemcpyAsync(second_state_result.data(),
            device_output.get(), sizeof(second_state_result),
            cudaMemcpyDeviceToHost, stream), "copy second cell-state output");
        cuda_require(cudaStreamSynchronize(stream), "finish second cell state");
        std::size_t free_after_state = 0u, total_after_state = 0u;
        cuda_require(cudaMemGetInfo(&free_after_state, &total_after_state),
            "memory after reused cell state");
        compare_output(second_state_result, second_contribution,
            "reused sequence relation failed second cell-state parity");
        require(free_before_state == free_after_state
                && total_before_state == total_after_state,
            "reused cell-state execution allocated steady-state storage");

        cuda_require(cudaMemsetAsync(device_output.get(), 0,
            3u * sizeof(float), stream), "zero next-sequence output");
        require(static_cast<bool>(cs::run_sequence_regulatory_relation_cached(
                materialized, materialized_launch, {12u}, &cache,
                &cache_result))
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
        require(static_cast<bool>(cs::prepare_sequence_regulatory_operation(
                reprojected_request, forced_relation,
                &reprojected_state, &reprojected)),
            "alternate regulatory projection did not prepare");
        require(static_cast<bool>(cs::run_sequence_regulatory_relation_cached(
                reprojected, materialized_launch, {12u}, &cache,
                &cache_result))
                && !cache_result.cache_hit
                && ce::same_identity(cache.key.regulatory_projection,
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
        require(static_cast<bool>(cs::prepare_sequence_regulatory_operation(
                reordered_request, forced_materialization,
                &reordered_state, &reordered)),
            "alternate persistent coordinate order did not prepare");
        require(static_cast<bool>(cs::run_sequence_regulatory_relation_cached(
                reordered, materialized_launch, {12u}, &cache,
                &cache_result))
                && !cache_result.cache_hit
                && ce::same_identity(cache.key.coordinate_order,
                    reordered_request.persistent_coordinate_order),
            "coordinate-order change reused a stale relation");
        cuda_require(cudaStreamSynchronize(stream), "finish reordered use");

        device_buffer<std::uint32_t> replacement_relation(base_count);
        ce::biological_operand_view rebound_outputs[2]{
            materialized_outputs[0], materialized_outputs[1]};
        rebound_outputs[1].storage.dense.data = replacement_relation.get();
        ce::launch_bindings rebound_launch = launch_bindings(
            relations, &input, rebound_outputs, 2u,
            &value_binding, stream, device);
        require(!cs::run_sequence_regulatory_relation_cached(
                reordered, rebound_launch, {12u}, &cache, &cache_result),
            "occupied cache silently rebound by pointer identity");

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
                    == base_count * sizeof(std::uint32_t),
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

        bp::prepared_predicate_plan stale_plan = baseplane_plan;
        ++stale_plan.semantic_hash;
        cs::sequence_prepare_request stale_request = request;
        stale_request.baseplane_plan = &stale_plan;
        require(!cs::prepare_sequence_regulatory_operation(
                stale_request, {}, &rejected_state, &rejected),
            "stale Baseplane semantic hash was accepted");

        bp::sequence_predicate_program reverse_program = program;
        reverse_program.outputs[0].flags = bp::sequence_event_reverse_strand;
        bp::prepared_predicate_plan reverse_plan{};
        require(bp::prepare_sequence_predicate_program(
                reverse_program, &reverse_plan) == bp::predicate_plan_status::ok,
            "Baseplane reverse fixture did not prepare");
        cs::sequence_prepare_request reverse_request = request;
        reverse_request.program = &reverse_program;
        reverse_request.baseplane_plan = &reverse_plan;
        require(!cs::prepare_sequence_regulatory_operation(
                reverse_request, {}, &rejected_state, &rejected),
            "unsupported reverse-strand primitive was silently accepted");

        const timing_summary fused_timing = measure_candidate(stream,
            [&](std::uint32_t) {
                return static_cast<bool>(
                    co::run_prepared_operation(fused, fused_launch));
            }, "fused benchmark launch failed");
        const timing_summary first_materialized_timing = measure_candidate(stream,
            [&](std::uint32_t sample) {
                cache.occupied = false;
                return static_cast<bool>(
                    cs::run_sequence_regulatory_relation_cached(
                        materialized, materialized_launch,
                        {1000u + sample}, &cache, &cache_result));
            }, "first materialized benchmark launch failed");
        cache.occupied = false;
        require(static_cast<bool>(cs::run_sequence_regulatory_relation_cached(
                materialized, materialized_launch, {2000u}, &cache,
                &cache_result)),
            "cached benchmark preparation failed");
        cuda_require(cudaStreamSynchronize(stream),
            "finish cached benchmark preparation");
        const timing_summary cached_timing = measure_candidate(stream,
            [&](std::uint32_t) {
                return static_cast<bool>(
                    cs::run_sequence_regulatory_relation_cached(
                        materialized, materialized_launch, {2000u}, &cache,
                        &cache_result));
            }, "cached materialized benchmark launch failed");

        cudaDeviceProp properties{};
        cuda_require(cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
        const ce::device_performance_class actual_device{
            0x10deu, static_cast<std::uint16_t>(properties.major),
            static_cast<std::uint16_t>(properties.minor), 0xce780002u};
        const cs::sequence_measurement_key actual_key{
            baseplane_plan.semantic_hash, request.persistent_coordinate_order,
            request.persistent_projection, actual_device, 0xce780002u,
            base_count, program.outputs[0].predicate_id,
            program.outputs[0].flags};
        const cs::sequence_strategy_evidence actual_evidence{
            actual_key, fused_timing.median_ns,
            first_materialized_timing.median_ns, cached_timing.median_ns,
            fused_timing.spread_percent,
            std::max(first_materialized_timing.spread_percent,
                cached_timing.spread_percent),
            9u, 0u};
        std::uint32_t first_materialized_state_count = 0u;
        for (std::uint32_t reuse = 1u; reuse <= 64u; ++reuse) {
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

        std::cout << "ce_arch_89_evidence"
                  << " device=" << properties.name
                  << " cc=" << properties.major << '.' << properties.minor
                  << " bases=" << base_count
                  << " samples=9 warmups=4 uses_per_sample=100"
                  << " fused_ns=" << fused_timing.median_ns
                  << " fused_spread_pct=" << fused_timing.spread_percent
                  << " first_materialized_ns="
                  << first_materialized_timing.median_ns
                  << " first_materialized_spread_pct="
                  << first_materialized_timing.spread_percent
                  << " cached_ns=" << cached_timing.median_ns
                  << " cached_spread_pct="
                  << cached_timing.spread_percent
                  << " first_materialized_state_count="
                  << first_materialized_state_count << '\n';

        cuda_require(cudaEventDestroy(cache_ready), "cudaEventDestroy cache_ready");
        cuda_require(cudaStreamDestroy(stream), "cudaStreamDestroy");
        std::cout << "celleratorBaseplaneSequenceIntegrationTest passed"
                  << " predicate_hash=" << baseplane_plan.semantic_hash
                  << " fused_launches=" << fused_bytes.launch_count
                  << " materialized_launches=" << materialized_bytes.launch_count
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "celleratorBaseplaneSequenceIntegrationTest: "
                  << error.what() << '\n';
        return 1;
    }
}
