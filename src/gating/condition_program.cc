#include <Cellerator/compute/gating/condition_program.hh>

#include <algorithm>
#include <limits>

namespace cellerator::compute::gating {
namespace {

std::uint64_t words_per_cell(const condition_program_view_v1 &program) noexcept {
    return program.edge_count / 64u + (program.edge_count % 64u != 0u ? 1u : 0u);
}

const edge_condition_binding_v1 *binding_for(
    const condition_program_view_v1 &program, std::uint64_t ordinal) noexcept {
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

bool evaluate_nodes(const condition_program_view_v1 &program,
                    const std::uint8_t *predicates,
                    std::uint8_t *scratch) noexcept {
    for (std::uint64_t i = 0u; i < program.node_count; ++i) {
        const auto &node = program.nodes[i];
        switch (node.opcode) {
        case condition_opcode_v1::predicate:
            scratch[i] = predicates[node.predicate_index] == node.expected_value;
            break;
        case condition_opcode_v1::all_of: {
            bool value = true;
            for (std::uint32_t j = 0u; j < node.child_count; ++j)
                value = value && scratch[node.first_child + j] != 0u;
            scratch[i] = value;
            break;
        }
        case condition_opcode_v1::any_of: {
            bool value = false;
            for (std::uint32_t j = 0u; j < node.child_count; ++j)
                value = value || scratch[node.first_child + j] != 0u;
            scratch[i] = value;
            break;
        }
        case condition_opcode_v1::logical_not:
            scratch[i] = scratch[node.first_child] == 0u;
            break;
        default:
            return false;
        }
    }
    return true;
}

bool evaluate_dirty_roots(const condition_program_view_v1 &program,
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
        if (node.opcode == condition_opcode_v1::all_of
            || node.opcode == condition_opcode_v1::any_of
            || node.opcode == condition_opcode_v1::logical_not)
            for (std::uint32_t child = 0u; child < node.child_count; ++child)
                scratch[node.first_child + child] = 1u;
    }
    // Scratch encodes 0=unused, 1=marked/uncomputed, 2=false, 3=true.
    for (std::uint64_t i = 0u; i < program.node_count; ++i) {
        if (scratch[i] != 1u) continue;
        const auto &node = program.nodes[i];
        bool value = false;
        switch (node.opcode) {
        case condition_opcode_v1::predicate:
            value = predicates[node.predicate_index] == node.expected_value;
            break;
        case condition_opcode_v1::all_of:
            value = true;
            for (std::uint32_t child = 0u; child < node.child_count; ++child)
                value = value && scratch[node.first_child + child] == 3u;
            break;
        case condition_opcode_v1::any_of:
            for (std::uint32_t child = 0u; child < node.child_count; ++child)
                value = value || scratch[node.first_child + child] == 3u;
            break;
        case condition_opcode_v1::logical_not:
            value = scratch[node.first_child] == 2u;
            break;
        default:
            return false;
        }
        scratch[i] = value ? 3u : 2u;
    }
    return true;
}

bool bitmap_test(const std::uint64_t *bitmap, std::uint64_t ordinal) noexcept {
    return ((bitmap[ordinal / 64u] >> (ordinal % 64u)) & 1u) != 0u;
}

condition_status_v1 validate_candidate_view(
    const per_cell_candidate_view_v1 &view, std::uint64_t cell_count,
    std::uint64_t edge_count) noexcept {
    if (view.offsets == nullptr || view.offsets[0] != 0u
        || view.offsets[cell_count] != view.item_count
        || (view.item_count != 0u && view.items == nullptr))
        return condition_status_v1::invalid_workset;
    for (std::uint64_t cell = 0u; cell < cell_count; ++cell) {
        if (view.offsets[cell] > view.offsets[cell + 1u]
            || view.offsets[cell + 1u] > view.item_count)
            return condition_status_v1::invalid_workset;
        std::uint64_t previous_id = 0u;
        for (std::uint64_t i = view.offsets[cell]; i < view.offsets[cell + 1u]; ++i) {
            if (view.items[i].logical_edge_id == 0u
                || view.items[i].logical_edge_ordinal >= edge_count
                || (i != view.offsets[cell]
                    && view.items[i].logical_edge_id <= previous_id))
                return condition_status_v1::invalid_workset;
            previous_id = view.items[i].logical_edge_id;
        }
    }
    return condition_status_v1::success;
}

bool eligible(const indexing::indexed_work_item_v1 &item,
              const std::uint64_t *bitmap) noexcept {
    return bitmap_test(bitmap, item.logical_edge_ordinal);
}

std::uint64_t count_eligible(const indexing::indexed_work_item_v1 *items,
                             std::uint64_t begin, std::uint64_t end,
                             const std::uint64_t *bitmap) noexcept {
    std::uint64_t count = 0u;
    for (std::uint64_t i = begin; i < end; ++i)
        if (eligible(items[i], bitmap)) ++count;
    return count;
}

bool count_eligible_union(
    const indexing::indexed_work_item_v1 *a, std::uint64_t a_begin,
    std::uint64_t a_end, const indexing::indexed_work_item_v1 *b,
    std::uint64_t b_begin, std::uint64_t b_end, const std::uint64_t *bitmap,
    std::uint64_t *count) noexcept {
    std::uint64_t i = a_begin, j = b_begin;
    *count = 0u;
    while (i < a_end && j < b_end) {
        if (a[i].logical_edge_id == b[j].logical_edge_id) {
            if (a[i].logical_edge_ordinal != b[j].logical_edge_ordinal) return false;
            if (eligible(a[i], bitmap)) ++*count;
            ++i;
            ++j;
        } else if (a[i].logical_edge_id < b[j].logical_edge_id) {
            if (eligible(a[i], bitmap)) ++*count;
            ++i;
        } else {
            if (eligible(b[j], bitmap)) ++*count;
            ++j;
        }
    }
    *count += count_eligible(a, i, a_end, bitmap);
    *count += count_eligible(b, j, b_end, bitmap);
    return true;
}

} // namespace

condition_status_v1 validate_condition_program_v1(
    const condition_program_view_v1 &program) noexcept {
    if (program.structure_epoch == 0u || program.reserved != 0u
        || (program.node_count != 0u && program.nodes == nullptr)
        || (program.binding_count != 0u && program.edges == nullptr))
        return condition_status_v1::invalid_argument;
    for (std::uint64_t i = 0u; i < program.node_count; ++i) {
        const auto &node = program.nodes[i];
        if (node.reserved != 0u || node.reserved2 != 0u
            || node.expected_value > 1u)
            return condition_status_v1::invalid_program;
        switch (node.opcode) {
        case condition_opcode_v1::predicate:
            if (node.predicate_index >= program.predicate_count
                || node.child_count != 0u)
                return condition_status_v1::invalid_program;
            break;
        case condition_opcode_v1::all_of:
        case condition_opcode_v1::any_of:
            if (node.first_child > i
                || node.child_count > i - node.first_child)
                return condition_status_v1::invalid_program;
            break;
        case condition_opcode_v1::logical_not:
            if (node.first_child >= i || node.child_count != 1u)
                return condition_status_v1::invalid_program;
            break;
        default:
            return condition_status_v1::invalid_program;
        }
    }
    std::uint64_t previous = 0u;
    for (std::uint64_t i = 0u; i < program.binding_count; ++i) {
        const auto &binding = program.edges[i];
        if (binding.reserved != 0u || binding.logical_edge_ordinal >= program.edge_count
            || binding.root_node >= program.node_count
            || (i != 0u && binding.logical_edge_ordinal <= previous))
            return condition_status_v1::invalid_program;
        previous = binding.logical_edge_ordinal;
    }
    return condition_status_v1::success;
}

condition_status_v1 prepare_condition_root_map_v1(
    const condition_program_view_v1 &program,
    std::uint32_t *root_node_by_edge_ordinal,
    std::uint64_t root_map_capacity,
    std::uint64_t *required_capacity) noexcept {
    if (required_capacity == nullptr)
        return condition_status_v1::invalid_argument;
    const auto validation = validate_condition_program_v1(program);
    if (validation != condition_status_v1::success) return validation;
    *required_capacity = program.edge_count;
    if (root_map_capacity < program.edge_count)
        return condition_status_v1::insufficient_root_map_capacity;
    if (program.edge_count != 0u && root_node_by_edge_ordinal == nullptr)
        return condition_status_v1::invalid_argument;

    std::fill_n(root_node_by_edge_ordinal, program.edge_count,
                unbound_condition_root_v1);
    for (std::uint64_t i = 0u; i < program.binding_count; ++i) {
        const auto &binding = program.edges[i];
        root_node_by_edge_ordinal[binding.logical_edge_ordinal] = binding.root_node;
    }
    return condition_status_v1::success;
}

condition_status_v1 evaluate_condition_bitmaps_v1(
    const condition_program_view_v1 &program, std::uint64_t cell_count,
    const std::uint8_t *predicate_values,
    std::uint64_t predicate_value_count,
    std::uint64_t *bitmap_words, std::uint64_t bitmap_word_capacity,
    std::uint8_t *node_scratch, std::uint64_t node_scratch_capacity) noexcept {
    if (validate_condition_program_v1(program) != condition_status_v1::success)
        return condition_status_v1::invalid_program;
    if (cell_count != 0u && program.predicate_count != 0u
        && predicate_values == nullptr)
        return condition_status_v1::invalid_argument;
    if (program.predicate_count != 0u
        && cell_count > std::numeric_limits<std::uint64_t>::max()
            / program.predicate_count)
        return condition_status_v1::invalid_argument;
    const auto expected_predicates = cell_count * program.predicate_count;
    if (predicate_value_count != expected_predicates)
        return condition_status_v1::invalid_argument;
    const auto words = words_per_cell(program);
    if (words != 0u && cell_count > std::numeric_limits<std::uint64_t>::max() / words)
        return condition_status_v1::invalid_argument;
    const auto total_words = words * cell_count;
    if (bitmap_word_capacity < total_words
        || (total_words != 0u && bitmap_words == nullptr))
        return condition_status_v1::insufficient_bitmap_capacity;
    if (program.node_count > node_scratch_capacity
        || (program.node_count != 0u && node_scratch == nullptr))
        return condition_status_v1::invalid_argument;
    for (std::uint64_t i = 0u; i < predicate_value_count; ++i)
        if (predicate_values[i] > 1u) return condition_status_v1::invalid_argument;

    for (std::uint64_t cell = 0u; cell < cell_count; ++cell) {
        auto *bitmap = words == 0u ? nullptr : bitmap_words + cell * words;
        for (std::uint64_t w = 0u; w < words; ++w) bitmap[w] = 0u;
        auto *predicates = program.predicate_count == 0u
            ? nullptr : predicate_values + cell * program.predicate_count;
        if (!evaluate_nodes(program, predicates, node_scratch))
            return condition_status_v1::invalid_program;
        for (std::uint64_t i = 0u; i < program.edge_count; ++i) {
            const auto *binding = binding_for(program, i);
            const bool pass = binding == nullptr || node_scratch[binding->root_node] != 0u;
            if (pass) bitmap[i / 64u] |= std::uint64_t{1} << (i % 64u);
        }
    }
    return condition_status_v1::success;
}

condition_status_v1 update_condition_bitmap_v1(
    const condition_program_view_v1 &program,
    const std::uint8_t *predicate_values, std::uint64_t predicate_value_count,
    const std::uint64_t *dirty_edge_ordinals, std::uint64_t dirty_edge_count,
    std::uint64_t *bitmap_words, std::uint64_t bitmap_word_capacity,
    std::uint8_t *node_scratch, std::uint64_t node_scratch_capacity) noexcept {
    if (validate_condition_program_v1(program) != condition_status_v1::success)
        return condition_status_v1::invalid_program;
    if (predicate_value_count != program.predicate_count
        || (predicate_value_count != 0u && predicate_values == nullptr)
        || (dirty_edge_count != 0u && dirty_edge_ordinals == nullptr))
        return condition_status_v1::invalid_argument;
    const auto words = words_per_cell(program);
    if (bitmap_word_capacity < words || (words != 0u && bitmap_words == nullptr))
        return condition_status_v1::insufficient_bitmap_capacity;
    if (program.node_count > node_scratch_capacity
        || (program.node_count != 0u && node_scratch == nullptr))
        return condition_status_v1::invalid_argument;
    for (std::uint64_t i = 0u; i < dirty_edge_count; ++i)
        if (dirty_edge_ordinals[i] >= program.edge_count)
            return condition_status_v1::invalid_argument;
    for (std::uint64_t i = 0u; i < predicate_value_count; ++i)
        if (predicate_values[i] > 1u) return condition_status_v1::invalid_argument;
    if (!evaluate_dirty_roots(program, predicate_values, dirty_edge_ordinals,
            dirty_edge_count, node_scratch))
        return condition_status_v1::invalid_program;
    for (std::uint64_t i = 0u; i < dirty_edge_count; ++i) {
        const auto ordinal = dirty_edge_ordinals[i];
        const auto *binding = binding_for(program, ordinal);
        const bool pass = binding == nullptr || node_scratch[binding->root_node] == 3u;
        const auto bit = std::uint64_t{1} << (ordinal % 64u);
        auto &word = bitmap_words[ordinal / 64u];
        if (pass) word |= bit;
        else word &= ~bit;
    }
    return condition_status_v1::success;
}

condition_status_v1 build_condition_worksets_v1(
    const condition_program_view_v1 &program, std::uint64_t cell_count,
    std::uint64_t state_generation,
    const std::uint64_t *bitmap_words, std::uint64_t bitmap_word_count,
    const per_cell_candidate_view_v1 &primal_candidates,
    const per_cell_candidate_view_v1 &derivative_candidates,
    gate_workset_pair_v1 *output) noexcept {
    if (output == nullptr
        || validate_condition_program_v1(program) != condition_status_v1::success)
        return condition_status_v1::invalid_program;
    const auto words = words_per_cell(program);
    if (words != 0u && cell_count > std::numeric_limits<std::uint64_t>::max() / words)
        return condition_status_v1::invalid_argument;
    if (bitmap_word_count != words * cell_count
        || (bitmap_word_count != 0u && bitmap_words == nullptr))
        return condition_status_v1::invalid_argument;
    auto status = validate_candidate_view(primal_candidates, cell_count,
                                          program.edge_count);
    if (status != condition_status_v1::success) return status;
    status = validate_candidate_view(derivative_candidates, cell_count,
                                     program.edge_count);
    if (status != condition_status_v1::success) return status;
    auto &primal = output->primal;
    auto &derivative = output->derivative;
    primal.written = primal.required = 0u;
    derivative.written = derivative.required = 0u;
    primal.structure_epoch = derivative.structure_epoch = program.structure_epoch;
    primal.state_generation = derivative.state_generation = state_generation;
    if (cell_count == std::numeric_limits<std::uint64_t>::max()
        || primal.offsets_capacity < cell_count + 1u
        || derivative.offsets_capacity < cell_count + 1u
        || primal.offsets == nullptr || derivative.offsets == nullptr)
        return condition_status_v1::insufficient_offsets_capacity;

    for (std::uint64_t cell = 0u; cell < cell_count; ++cell) {
        const auto *bitmap = words == 0u ? nullptr : bitmap_words + cell * words;
        const auto p_begin = primal_candidates.offsets[cell];
        const auto p_end = primal_candidates.offsets[cell + 1u];
        const auto d_begin = derivative_candidates.offsets[cell];
        const auto d_end = derivative_candidates.offsets[cell + 1u];
        primal.required += count_eligible(primal_candidates.items,
            p_begin, p_end, bitmap);
        std::uint64_t union_count = 0u;
        if (!count_eligible_union(primal_candidates.items, p_begin, p_end,
                derivative_candidates.items, d_begin, d_end, bitmap, &union_count))
            return condition_status_v1::invalid_workset;
        derivative.required += union_count;
    }
    if (primal.required > primal.item_capacity
        || (primal.required != 0u && primal.items == nullptr))
        return condition_status_v1::primal_workset_overflow;
    if (derivative.required > derivative.item_capacity
        || (derivative.required != 0u && derivative.items == nullptr))
        return condition_status_v1::derivative_workset_overflow;

    primal.offsets[0] = 0u;
    derivative.offsets[0] = 0u;
    for (std::uint64_t cell = 0u; cell < cell_count; ++cell) {
        const auto *bitmap = words == 0u ? nullptr : bitmap_words + cell * words;
        const auto p_begin = primal_candidates.offsets[cell];
        const auto p_end = primal_candidates.offsets[cell + 1u];
        const auto d_begin = derivative_candidates.offsets[cell];
        const auto d_end = derivative_candidates.offsets[cell + 1u];
        for (std::uint64_t i = p_begin; i < p_end; ++i)
            if (eligible(primal_candidates.items[i], bitmap))
                primal.items[primal.written++] = primal_candidates.items[i];
        std::uint64_t i = p_begin, j = d_begin;
        while (true) {
            while (i < p_end && !eligible(primal_candidates.items[i], bitmap)) ++i;
            while (j < d_end && !eligible(derivative_candidates.items[j], bitmap)) ++j;
            if (i == p_end || j == d_end) break;
            const auto &a = primal_candidates.items[i];
            const auto &b = derivative_candidates.items[j];
            if (a.logical_edge_id == b.logical_edge_id) {
                derivative.items[derivative.written++] = a;
                ++i;
                ++j;
            } else if (a.logical_edge_id < b.logical_edge_id) {
                derivative.items[derivative.written++] = a;
                ++i;
            } else {
                derivative.items[derivative.written++] = b;
                ++j;
            }
        }
        while (i < p_end) {
            if (eligible(primal_candidates.items[i], bitmap))
                derivative.items[derivative.written++] = primal_candidates.items[i];
            ++i;
        }
        while (j < d_end) {
            if (eligible(derivative_candidates.items[j], bitmap))
                derivative.items[derivative.written++] = derivative_candidates.items[j];
            ++j;
        }
        primal.offsets[cell + 1u] = primal.written;
        derivative.offsets[cell + 1u] = derivative.written;
    }
    return condition_status_v1::success;
}

} // namespace cellerator::compute::gating
