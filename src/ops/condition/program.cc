#include <Cellerator/ops/condition/program.hh>

#include <algorithm>
#include <limits>

namespace cellerator::ops::condition {
namespace {

std::uint64_t words_per_batch(const ProgramViewV1 &program) noexcept {
    return program.edge_count / 64u + (program.edge_count % 64u != 0u ? 1u : 0u);
}

const EdgeBindingV1 *binding_for(
    const ProgramViewV1 &program, std::uint64_t ordinal) noexcept {
    std::uint64_t first = 0u;
    std::uint64_t last = program.binding_count;
    while (first < last) {
        const auto middle = first + (last - first) / 2u;
        if (program.edges[middle].logical_edge_ordinal < ordinal)
            first = middle + 1u;
        else
            last = middle;
    }
    if (first == program.binding_count
        || program.edges[first].logical_edge_ordinal != ordinal)
        return nullptr;
    return &program.edges[first];
}

bool evaluate_nodes(const ProgramViewV1 &program,
                    const std::uint8_t *predicates,
                    std::uint8_t *scratch) noexcept {
    for (std::uint64_t i = 0u; i < program.node_count; ++i) {
        const auto &node = program.nodes[i];
        switch (node.opcode) {
        case OpcodeV1::predicate:
            scratch[i] = predicates[node.predicate_index] == node.expected_value;
            break;
        case OpcodeV1::all_of: {
            bool value = true;
            for (std::uint32_t j = 0u; j < node.child_count; ++j)
                value = value && scratch[node.first_child + j] != 0u;
            scratch[i] = value;
            break;
        }
        case OpcodeV1::any_of: {
            bool value = false;
            for (std::uint32_t j = 0u; j < node.child_count; ++j)
                value = value || scratch[node.first_child + j] != 0u;
            scratch[i] = value;
            break;
        }
        case OpcodeV1::logical_not:
            scratch[i] = scratch[node.first_child] == 0u;
            break;
        default:
            return false;
        }
    }
    return true;
}

bool evaluate_dirty_roots(const ProgramViewV1 &program,
                          const std::uint8_t *predicates,
                          const std::uint64_t *dirty_ordinals,
                          std::uint64_t dirty_count,
                          std::uint8_t *scratch) noexcept {
    for (std::uint64_t i = 0u; i < program.node_count; ++i) scratch[i] = 0u;
    for (std::uint64_t i = 0u; i < dirty_count; ++i) {
        const auto *binding = binding_for(program, dirty_ordinals[i]);
        if (binding != nullptr) scratch[binding->root_node] = 1u;
    }
    // Children always precede parents, so a descending pass marks the full
    // dependency closure without recursion or temporary allocation.
    for (std::uint64_t i = program.node_count; i != 0u; --i) {
        const auto index = i - 1u;
        if (scratch[index] != 1u) continue;
        const auto &node = program.nodes[index];
        if (node.opcode == OpcodeV1::all_of
            || node.opcode == OpcodeV1::any_of
            || node.opcode == OpcodeV1::logical_not)
            for (std::uint32_t child = 0u; child < node.child_count; ++child)
                scratch[node.first_child + child] = 1u;
    }
    // Scratch encodes 0=unused, 1=marked/uncomputed, 2=false, 3=true.
    for (std::uint64_t i = 0u; i < program.node_count; ++i) {
        if (scratch[i] != 1u) continue;
        const auto &node = program.nodes[i];
        bool value = false;
        switch (node.opcode) {
        case OpcodeV1::predicate:
            value = predicates[node.predicate_index] == node.expected_value;
            break;
        case OpcodeV1::all_of:
            value = true;
            for (std::uint32_t child = 0u; child < node.child_count; ++child)
                value = value && scratch[node.first_child + child] == 3u;
            break;
        case OpcodeV1::any_of:
            for (std::uint32_t child = 0u; child < node.child_count; ++child)
                value = value || scratch[node.first_child + child] == 3u;
            break;
        case OpcodeV1::logical_not:
            value = scratch[node.first_child] == 2u;
            break;
        default:
            return false;
        }
        scratch[i] = value ? 3u : 2u;
    }
    return true;
}

} // namespace

StatusV1 validate(
    const ProgramViewV1 &program) noexcept {
    if (program.structure_epoch == 0u || program.reserved != 0u
        || (program.node_count != 0u && program.nodes == nullptr)
        || (program.binding_count != 0u && program.edges == nullptr))
        return StatusV1::invalid_argument;
    for (std::uint64_t i = 0u; i < program.node_count; ++i) {
        const auto &node = program.nodes[i];
        if (node.reserved != 0u || node.reserved2 != 0u
            || node.expected_value > 1u)
            return StatusV1::invalid_program;
        switch (node.opcode) {
        case OpcodeV1::predicate:
            if (node.predicate_index >= program.predicate_count
                || node.child_count != 0u)
                return StatusV1::invalid_program;
            break;
        case OpcodeV1::all_of:
        case OpcodeV1::any_of:
            if (node.first_child > i
                || node.child_count > i - node.first_child)
                return StatusV1::invalid_program;
            break;
        case OpcodeV1::logical_not:
            if (node.first_child >= i || node.child_count != 1u)
                return StatusV1::invalid_program;
            break;
        default:
            return StatusV1::invalid_program;
        }
    }
    std::uint64_t previous = 0u;
    for (std::uint64_t i = 0u; i < program.binding_count; ++i) {
        const auto &binding = program.edges[i];
        if (binding.reserved != 0u || binding.logical_edge_ordinal >= program.edge_count
            || binding.root_node >= program.node_count
            || (i != 0u && binding.logical_edge_ordinal <= previous))
            return StatusV1::invalid_program;
        previous = binding.logical_edge_ordinal;
    }
    return StatusV1::success;
}

StatusV1 prepare_root_map(
    const ProgramViewV1 &program,
    std::uint32_t *root_node_by_edge_ordinal,
    std::uint64_t root_map_capacity,
    std::uint64_t *required_capacity) noexcept {
    if (required_capacity == nullptr)
        return StatusV1::invalid_argument;
    const auto validation = validate(program);
    if (validation != StatusV1::success) return validation;
    *required_capacity = program.edge_count;
    if (root_map_capacity < program.edge_count)
        return StatusV1::insufficient_root_map_capacity;
    if (program.edge_count != 0u && root_node_by_edge_ordinal == nullptr)
        return StatusV1::invalid_argument;

    std::fill_n(root_node_by_edge_ordinal, program.edge_count,
                unbound_root_v1);
    for (std::uint64_t i = 0u; i < program.binding_count; ++i) {
        const auto &binding = program.edges[i];
        root_node_by_edge_ordinal[binding.logical_edge_ordinal] = binding.root_node;
    }
    return StatusV1::success;
}

StatusV1 evaluate_bitmaps(
    const ProgramViewV1 &program, std::uint64_t batch_count,
    const std::uint8_t *predicate_values,
    std::uint64_t predicate_value_count,
    std::uint64_t *bitmap_words, std::uint64_t bitmap_word_capacity,
    std::uint8_t *node_scratch, std::uint64_t node_scratch_capacity) noexcept {
    if (validate(program) != StatusV1::success)
        return StatusV1::invalid_program;
    if (batch_count != 0u && program.predicate_count != 0u
        && predicate_values == nullptr)
        return StatusV1::invalid_argument;
    if (program.predicate_count != 0u
        && batch_count > std::numeric_limits<std::uint64_t>::max()
            / program.predicate_count)
        return StatusV1::invalid_argument;
    const auto expected_predicates = batch_count * program.predicate_count;
    if (predicate_value_count != expected_predicates)
        return StatusV1::invalid_argument;
    const auto words = words_per_batch(program);
    if (words != 0u && batch_count > std::numeric_limits<std::uint64_t>::max() / words)
        return StatusV1::invalid_argument;
    const auto total_words = words * batch_count;
    if (bitmap_word_capacity < total_words
        || (total_words != 0u && bitmap_words == nullptr))
        return StatusV1::insufficient_bitmap_capacity;
    if (program.node_count > node_scratch_capacity
        || (program.node_count != 0u && node_scratch == nullptr))
        return StatusV1::invalid_argument;
    for (std::uint64_t i = 0u; i < predicate_value_count; ++i)
        if (predicate_values[i] > 1u) return StatusV1::invalid_argument;

    for (std::uint64_t batch = 0u; batch < batch_count; ++batch) {
        auto *bitmap = words == 0u ? nullptr : bitmap_words + batch * words;
        for (std::uint64_t w = 0u; w < words; ++w) bitmap[w] = 0u;
        auto *predicates = program.predicate_count == 0u
            ? nullptr : predicate_values + batch * program.predicate_count;
        if (!evaluate_nodes(program, predicates, node_scratch))
            return StatusV1::invalid_program;
        for (std::uint64_t i = 0u; i < program.edge_count; ++i) {
            const auto *binding = binding_for(program, i);
            const bool pass = binding == nullptr || node_scratch[binding->root_node] != 0u;
            if (pass) bitmap[i / 64u] |= std::uint64_t{1} << (i % 64u);
        }
    }
    return StatusV1::success;
}

StatusV1 update_bitmap(
    const ProgramViewV1 &program,
    const std::uint8_t *predicate_values, std::uint64_t predicate_value_count,
    const std::uint64_t *dirty_edge_ordinals, std::uint64_t dirty_edge_count,
    std::uint64_t *bitmap_words, std::uint64_t bitmap_word_capacity,
    std::uint8_t *node_scratch, std::uint64_t node_scratch_capacity) noexcept {
    if (validate(program) != StatusV1::success)
        return StatusV1::invalid_program;
    if (predicate_value_count != program.predicate_count
        || (predicate_value_count != 0u && predicate_values == nullptr)
        || (dirty_edge_count != 0u && dirty_edge_ordinals == nullptr))
        return StatusV1::invalid_argument;
    const auto words = words_per_batch(program);
    if (bitmap_word_capacity < words || (words != 0u && bitmap_words == nullptr))
        return StatusV1::insufficient_bitmap_capacity;
    if (program.node_count > node_scratch_capacity
        || (program.node_count != 0u && node_scratch == nullptr))
        return StatusV1::invalid_argument;
    for (std::uint64_t i = 0u; i < dirty_edge_count; ++i)
        if (dirty_edge_ordinals[i] >= program.edge_count)
            return StatusV1::invalid_argument;
    for (std::uint64_t i = 0u; i < predicate_value_count; ++i)
        if (predicate_values[i] > 1u) return StatusV1::invalid_argument;
    if (!evaluate_dirty_roots(program, predicate_values, dirty_edge_ordinals,
            dirty_edge_count, node_scratch))
        return StatusV1::invalid_program;
    for (std::uint64_t i = 0u; i < dirty_edge_count; ++i) {
        const auto ordinal = dirty_edge_ordinals[i];
        const auto *binding = binding_for(program, ordinal);
        const bool pass = binding == nullptr || node_scratch[binding->root_node] == 3u;
        const auto bit = std::uint64_t{1} << (ordinal % 64u);
        auto &word = bitmap_words[ordinal / 64u];
        if (pass) word |= bit;
        else word &= ~bit;
    }
    return StatusV1::success;
}


} // namespace cellerator::ops::condition
