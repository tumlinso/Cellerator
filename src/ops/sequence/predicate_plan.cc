#include <Cellerator/ops/sequence/predicate_plan.hh>

namespace cellerator::ops::sequence {
namespace {

using Op = predicate_opcode;

int operand_count(Op opcode) noexcept {
    switch (opcode) {
    case Op::base_equal: case Op::allowed_motif: case Op::exact_motif:
    case Op::gc_mask: case Op::base_class_mask: case Op::cpg_start_mask:
        return 0;
    case Op::shift: case Op::bit_not: case Op::any: case Op::all:
    case Op::popcount_threshold: case Op::run: case Op::boundary:
    case Op::emit_event: case Op::segment_reduce_request:
        return 1;
    case Op::bit_and: case Op::bit_or: case Op::bit_xor: case Op::bit_and_not:
    case Op::spacing_exact: case Op::spacing_within:
        return 2;
    default:
        return -1;
    }
}

template<typename T>
void hash_unsigned(uint64_t &hash, T value) noexcept {
    for (unsigned shift = 0u; shift < sizeof(T) * 8u; shift += 8u)
        hash = (hash ^ uint8_t(value >> shift))*1099511628211ULL;
}

bool extents_fit(const sequence_predicate_program &p) noexcept {
    return p.node_count <= sequence_program_max_nodes
        && p.output_count <= sequence_program_max_outputs
        && p.exact_motif_count <= sequence_program_max_motifs
        && p.allowed_motif_count <= sequence_program_max_motifs;
}

} // namespace

predicate_plan_status verify_sequence_predicate_program(
    const sequence_predicate_program &program) noexcept {
    if (program.version != sequence_program_version)
        return predicate_plan_status::unsupported_version;
    if (!program.node_count || !program.output_count || !extents_fit(program))
        return predicate_plan_status::invalid_count;
    for (uint8_t index = 0u; index < program.exact_motif_count; ++index)
        if (!dna2_valid_motif32_exact(program.exact_motifs[index]))
            return predicate_plan_status::invalid_motif;
    for (uint8_t index = 0u; index < program.allowed_motif_count; ++index)
        if (!dna2_valid_motif32_allowed(program.allowed_motifs[index]))
            return predicate_plan_status::invalid_motif;

    for (uint16_t index = 0u; index < program.node_count; ++index) {
        const predicate_instruction &node = program.nodes[index];
        const int operands = operand_count(node.opcode);
        if (operands < 0 || node.result_kind < predicate_value_kind::mask
            || node.result_kind > predicate_value_kind::segment_request)
            return predicate_plan_status::invalid_opcode;
        const bool valid_a = operands ? node.input_a < index : node.input_a == sequence_program_no_input;
        const bool valid_b = operands == 2 ? node.input_b < index : node.input_b == sequence_program_no_input;
        if (!valid_a || !valid_b)
            return predicate_plan_status::invalid_input;
        if (node.opcode == Op::base_equal && node.immediate > 3u)
            return predicate_plan_status::invalid_immediate;
        if (node.opcode == Op::exact_motif && node.immediate >= program.exact_motif_count)
            return predicate_plan_status::invalid_motif;
        if (node.opcode == Op::allowed_motif && node.immediate >= program.allowed_motif_count)
            return predicate_plan_status::invalid_motif;
        if ((node.opcode == Op::shift || node.opcode == Op::any || node.opcode == Op::all
                || node.opcode == Op::spacing_exact || node.opcode == Op::spacing_within
                || node.opcode == Op::run)
            && (node.immediate == 0u || node.immediate > sequence_program_max_span))
            return predicate_plan_status::invalid_immediate;
        if (node.opcode == Op::popcount_threshold && node.immediate > sequence_program_max_span)
            return predicate_plan_status::invalid_immediate;
    }
    for (uint16_t index = 0u; index < program.output_count; ++index) {
        const predicate_output &output = program.outputs[index];
        if (output.node >= program.node_count
            || output.mode < sequence_output_mode::count
            || output.mode > sequence_output_mode::unordered_emit)
            return predicate_plan_status::invalid_output;
    }
    return predicate_plan_status::ok;
}

uint64_t sequence_predicate_program_hash(
    const sequence_predicate_program &program) noexcept {
    if (!extents_fit(program)) return 0;
    uint64_t hash = 1469598103934665603ULL;
    hash_unsigned(hash, program.version);
    hash_unsigned(hash, program.node_count);
    hash_unsigned(hash, program.output_count);
    hash_unsigned(hash, program.exact_motif_count);
    hash_unsigned(hash, program.allowed_motif_count);
    hash_unsigned(hash, program.semantic_flags);
    for (uint16_t index = 0u; index < program.node_count; ++index) {
        const predicate_instruction &node = program.nodes[index];
        hash_unsigned(hash, uint8_t(node.opcode));
        hash_unsigned(hash, uint8_t(node.result_kind));
        hash_unsigned(hash, node.flags);
        hash_unsigned(hash, node.input_a);
        hash_unsigned(hash, node.input_b);
        hash_unsigned(hash, node.immediate);
        hash_unsigned(hash, node.auxiliary);
    }
    for (uint16_t index = 0u; index < program.output_count; ++index) {
        const predicate_output &output = program.outputs[index];
        hash_unsigned(hash, output.node);
        hash_unsigned(hash, output.predicate_id);
        hash_unsigned(hash, uint8_t(output.mode));
        hash_unsigned(hash, output.payload);
        hash_unsigned(hash, output.flags);
    }
    for (uint8_t index = 0u; index < program.exact_motif_count; ++index) {
        const motif32_exact &motif = program.exact_motifs[index];
        hash_unsigned(hash, motif.packed);
        hash_unsigned(hash, motif.active_fields);
        hash_unsigned(hash, motif.length);
        hash_unsigned(hash, motif.max_mismatches);
        hash_unsigned(hash, motif.motif_id);
    }
    for (uint8_t index = 0u; index < program.allowed_motif_count; ++index) {
        const motif32_allowed &motif = program.allowed_motifs[index];
        hash_unsigned(hash, motif.allow_A); hash_unsigned(hash, motif.allow_C);
        hash_unsigned(hash, motif.allow_G); hash_unsigned(hash, motif.allow_T);
        hash_unsigned(hash, motif.active_mask); hash_unsigned(hash, motif.length);
        hash_unsigned(hash, motif.max_mismatches); hash_unsigned(hash, motif.motif_id);
        hash_unsigned(hash, uint8_t(motif.strand_mode));
        hash_unsigned(hash, uint8_t(motif.palindrome_policy));
    }
    return hash;
}

predicate_plan_status prepare_sequence_predicate_program(
    const sequence_predicate_program &program,
    prepared_predicate_plan *prepared) noexcept {
    if (prepared == nullptr) return predicate_plan_status::null_output;
    const predicate_plan_status status = verify_sequence_predicate_program(program);
    if (status != predicate_plan_status::ok) return status;

    bool live[sequence_program_max_nodes]{};
    for (uint16_t index = 0u; index < program.output_count; ++index)
        live[program.outputs[index].node] = true;
    // Validated inputs precede their consumers, so one reverse pass finds all ancestors.
    for (int i = int(program.node_count) - 1; i >= 0; --i) {
        if (!live[i]) continue;
        const auto &node = program.nodes[i];
        if (node.input_a != sequence_program_no_input) live[node.input_a] = true;
        if (node.input_b != sequence_program_no_input) live[node.input_b] = true;
    }
    *prepared = prepared_predicate_plan{};
    prepared->version = sequence_program_version;
    prepared->node_count = program.node_count;
    prepared->output_count = program.output_count;
    prepared->semantic_hash = sequence_predicate_program_hash(program);
    prepared->family = predicate_lowering_family::exact_scan;
    for (uint16_t index = 0u; index < sequence_program_max_nodes; ++index)
        prepared->original_to_prepared[index] = sequence_program_no_input;
    for (uint16_t index = 0u; index < program.node_count; ++index) {
        if (!live[index]) continue;
        prepared->original_to_prepared[index] = prepared->live_node_count++;
        const predicate_instruction &node = program.nodes[index];
        if (node.opcode == Op::segment_reduce_request)
            prepared->family = predicate_lowering_family::segment;
        else if (prepared->family != predicate_lowering_family::segment
            && node.opcode != Op::exact_motif)
            prepared->family = predicate_lowering_family::fused_plane_mask;
        uint32_t before = 0u, after = 0u;
        switch (node.opcode) {
        case Op::exact_motif: after = program.exact_motifs[node.immediate].length - 1u; break;
        case Op::allowed_motif: after = program.allowed_motifs[node.immediate].length - 1u; break;
        case Op::cpg_start_mask: after = 1u; break;
        case Op::shift:
            if (node.flags & predicate_shift_right) after = node.immediate;
            else before = node.immediate;
            break;
        case Op::spacing_exact: case Op::spacing_within:
        case Op::run: case Op::any: case Op::all:
            after = node.immediate - 1u;
            break;
        default: break;
        }
        if (before > prepared->lookbehind) prepared->lookbehind = before;
        if (after > prepared->lookahead) prepared->lookahead = after;
    }
    prepared->scratch_mask_words_per_window = prepared->live_node_count;
    return predicate_plan_status::ok;
}

} // namespace cellerator::ops::sequence
