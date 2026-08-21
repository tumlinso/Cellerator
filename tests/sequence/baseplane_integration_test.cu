#include <Cellerator/compute/sequence/baseplane_integration.cuh>

#include <Baseplane/seq/dna2_scan.cuh>
#include <Cellerator/execution/validation.hh>

#include <cuda_runtime.h>

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
    const ce::relation_structure *relation,
    const ce::biological_operand_view *input,
    ce::biological_operand_view *outputs,
    std::uint32_t output_count,
    const ce::value_binding *values,
    cudaStream_t stream,
    int device) {
    ce::launch_bindings launch{};
    launch.structure = relation;
    launch.inputs = input;
    launch.outputs = outputs;
    launch.values = values;
    launch.input_count = 1u;
    launch.output_count = output_count;
    launch.value_count = 1u;
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
        const ce::axis_identity predicate_mask_axis = axis(41u, 42u, 45u, 44u);
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
            device_mask(word_count), device_offsets(offsets.size()),
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
        cuda_require(cudaStreamSynchronize(stream), "finish input preparation");

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
        ce::relation_structure relation{
            {61u, 1u}, {1u}, coordinate_axis, gene_axis, {62u, 1u}, 3u};
        const cs::regulatory_projection_view device_projection{
            device_intervals.get(), device_offsets.get(), device_genes.get(),
            3u, 3u, 3u, 3u,
            {ce::residency_kind::device, {}, device, 0u}};
        const cs::sequence_prepare_request request{
            &program, &baseplane_plan, {71u, 72u}, {73u, 74u}, {63u, 1u},
            relation, source_domain, predicate_mask_axis, device_projection};
        ce::value_plane value_plane{
            relation.identity, relation.epoch, device_weights.get(),
            {ce::residency_kind::device, {}, device, 0u},
            {ce::numeric_type::f32, ce::numeric_type::f32,
             ce::numeric_type::f32, 0u},
            {ce::quantization_kind::none, ce::numeric_type::invalid,
             ce::numeric_type::invalid, 0u, nullptr, nullptr, 0u},
            ce::value_layout_kind::logical_edge_order, {}, {1u}, 3u,
            3u * sizeof(float)};
        const ce::value_binding value_binding{&value_plane, {1u}};

        const std::array<float, 3> expected = scalar_reference(
            packed, validity, chunk, program.exact_motifs[0],
            intervals, offsets, genes, weights);
        require(expected[0] == 2.0f && expected[1] == 0.5f
                && expected[2] == 1.5f,
            "scalar fixture does not cover expected regulatory mapping");

        cs::prepared_sequence_state fused_state{};
        co::prepared_operation fused{};
        require(static_cast<bool>(cs::prepare_sequence_regulatory_operation(
                request, {cs::sequence_strategy::automatic, 1u, true, true, {}},
                &fused_state, &fused)),
            "fused operation did not prepare");
        require(fused_state.strategy == cs::sequence_strategy::fuse_predicate,
            "one-shot policy did not select fusion");
        ce::biological_operand_view fused_output{};
        fused_output.kind = ce::operand_kind::dense_tensor;
        fused_output.storage.dense = gene_output(device_output.get(), gene_axis,
            {ce::residency_kind::device, {}, device, 0u});
        cuda_require(cudaMemsetAsync(device_output.get(), 0, 3u * sizeof(float), stream),
            "zero fused output");
        ce::launch_bindings fused_launch = launch_bindings(
            &relation, &input, &fused_output, 1u, &value_binding, stream, device);
        require(static_cast<bool>(co::run_prepared_operation(fused, fused_launch)),
            "fused operation launch rejected");
        std::array<float, 3> fused_result{};
        cuda_require(cudaMemcpyAsync(fused_result.data(), device_output.get(),
            sizeof(fused_result), cudaMemcpyDeviceToHost, stream),
            "copy fused output");
        cuda_require(cudaStreamSynchronize(stream), "finish fused path");
        compare_output(fused_result, expected, "fused scalar parity failed");

        cs::prepared_sequence_state materialized_state{};
        co::prepared_operation materialized{};
        require(static_cast<bool>(cs::prepare_sequence_regulatory_operation(
                request, {cs::sequence_strategy::automatic, 2u, true, true, {}},
                &materialized_state, &materialized)),
            "materialized operation did not prepare");
        require(materialized_state.strategy
                == cs::sequence_strategy::materialize_mask,
            "reuse policy did not select materialization");
        ce::biological_operand_view materialized_outputs[2]{};
        materialized_outputs[0].kind = ce::operand_kind::dense_tensor;
        materialized_outputs[0].storage.dense = gene_output(
            device_output.get(), gene_axis,
            {ce::residency_kind::device, {}, device, 0u});
        materialized_outputs[1].kind = ce::operand_kind::dense_tensor;
        materialized_outputs[1].storage.dense = ce::dense_tensor_view{};
        materialized_outputs[1].storage.dense.data = device_mask.get();
        materialized_outputs[1].storage.dense.location =
            {ce::residency_kind::device, {}, device, 0u};
        materialized_outputs[1].storage.dense.value_type = ce::numeric_type::u32;
        materialized_outputs[1].storage.dense.rank = 1u;
        materialized_outputs[1].storage.dense.axes[0] = predicate_mask_axis;
        materialized_outputs[1].storage.dense.shape[0] = word_count;
        materialized_outputs[1].storage.dense.stride[0] = 1;
        cuda_require(cudaMemsetAsync(device_output.get(), 0, 3u * sizeof(float), stream),
            "zero materialized output");
        ce::launch_bindings materialized_launch = launch_bindings(
            &relation, &input, materialized_outputs, 2u,
            &value_binding, stream, device);
        require(static_cast<bool>(
                co::run_prepared_operation(materialized, materialized_launch)),
            "materialized operation launch rejected");
        std::array<float, 3> materialized_result{};
        std::array<std::uint32_t, word_count> mask_result{};
        cuda_require(cudaMemcpyAsync(materialized_result.data(), device_output.get(),
            sizeof(materialized_result), cudaMemcpyDeviceToHost, stream),
            "copy materialized output");
        cuda_require(cudaMemcpyAsync(mask_result.data(), device_mask.get(),
            sizeof(mask_result), cudaMemcpyDeviceToHost, stream),
            "copy predicate mask");
        cuda_require(cudaStreamSynchronize(stream), "finish materialized path");
        compare_output(materialized_result, expected,
            "materialized scalar parity failed");
        require(mask_result[0] == ((1u << 5u) | (1u << 30u))
                && mask_result[1] == 0u && mask_result[2] == 0u,
            "validity, halo, boundary, or tail mask semantics failed");

        const cs::sequence_execution_accounting fused_bytes =
            cs::sequence_accounting(fused_state, base_count);
        const cs::sequence_execution_accounting materialized_bytes =
            cs::sequence_accounting(materialized_state, base_count);
        require(fused_bytes.launch_count == 1u
                && fused_bytes.materialized_mask_bytes == 0u
                && materialized_bytes.launch_count == 2u
                && materialized_bytes.materialized_mask_bytes
                    == word_count * sizeof(std::uint32_t),
            "strategy accounting is incomplete");

        ce::biological_operand_view stale_input = input;
        stale_input.storage.bits.coordinate_axis.order.slot += 1u;
        ce::launch_bindings stale_launch = launch_bindings(
            &relation, &stale_input, &fused_output, 1u,
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
        cs::prepared_sequence_state rejected_state{};
        co::prepared_operation rejected{};
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
