#include <Cellerator/execution/dependency/workset_view.hh>

#include <algorithm>
#include <limits>

namespace cellerator::execution::dependency {
namespace {
std::uint64_t words_per_batch(const ops::condition::ProgramViewV1& program) noexcept {
    return program.edge_count / 64u + (program.edge_count % 64u != 0u ? 1u : 0u);
}
bool bitmap_test(const std::uint64_t *bitmap, std::uint64_t ordinal) noexcept {
    return ((bitmap[ordinal / 64u] >> (ordinal % 64u)) & 1u) != 0u;
}

ops::condition::StatusV1 validate_candidate_view(
    const CandidateBatchView &view, std::uint64_t batch_count,
    std::uint64_t edge_count) noexcept {
    if (view.offsets == nullptr || view.offsets[0] != 0u
        || view.offsets[batch_count] != view.item_count
        || (view.item_count != 0u && view.items == nullptr))
        return ops::condition::StatusV1::invalid_workset;
    for (std::uint64_t batch = 0u; batch < batch_count; ++batch) {
        if (view.offsets[batch] > view.offsets[batch + 1u]
            || view.offsets[batch + 1u] > view.item_count)
            return ops::condition::StatusV1::invalid_workset;
        std::uint64_t previous_id = 0u;
        for (std::uint64_t i = view.offsets[batch]; i < view.offsets[batch + 1u]; ++i) {
            if (view.items[i].logical_edge_id == 0u
                || view.items[i].logical_edge_ordinal >= edge_count
                || (i != view.offsets[batch]
                    && view.items[i].logical_edge_id <= previous_id))
                return ops::condition::StatusV1::invalid_workset;
            previous_id = view.items[i].logical_edge_id;
        }
    }
    return ops::condition::StatusV1::success;
}

bool eligible(const containers::IndexedWorkItem &item,
              const std::uint64_t *bitmap) noexcept {
    return bitmap_test(bitmap, item.logical_edge_ordinal);
}

std::uint64_t count_eligible(const containers::IndexedWorkItem *items,
                             std::uint64_t begin, std::uint64_t end,
                             const std::uint64_t *bitmap) noexcept {
    std::uint64_t count = 0u;
    for (std::uint64_t i = begin; i < end; ++i)
        if (eligible(items[i], bitmap)) ++count;
    return count;
}

bool count_eligible_union(
    const containers::IndexedWorkItem *a, std::uint64_t a_begin,
    std::uint64_t a_end, const containers::IndexedWorkItem *b,
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

ops::condition::StatusV1 build_condition_worksets(
    const ops::condition::ProgramViewV1 &program, std::uint64_t batch_count,
    std::uint64_t state_generation,
    const std::uint64_t *bitmap_words, std::uint64_t bitmap_word_count,
    const CandidateBatchView &primal_candidates,
    const CandidateBatchView &derivative_candidates,
    ConditionWorksets *output) noexcept {
    if (output == nullptr
        || ops::condition::validate(program) != ops::condition::StatusV1::success)
        return ops::condition::StatusV1::invalid_program;
    const auto words = words_per_batch(program);
    if (words != 0u && batch_count > std::numeric_limits<std::uint64_t>::max() / words)
        return ops::condition::StatusV1::invalid_argument;
    if (bitmap_word_count != words * batch_count
        || (bitmap_word_count != 0u && bitmap_words == nullptr))
        return ops::condition::StatusV1::invalid_argument;
    auto status = validate_candidate_view(primal_candidates, batch_count,
                                          program.edge_count);
    if (status != ops::condition::StatusV1::success) return status;
    status = validate_candidate_view(derivative_candidates, batch_count,
                                     program.edge_count);
    if (status != ops::condition::StatusV1::success) return status;
    auto &primal = output->primal;
    auto &derivative = output->derivative;
    primal.written = primal.required = 0u;
    derivative.written = derivative.required = 0u;
    primal.structure_epoch = derivative.structure_epoch = program.structure_epoch;
    primal.state_generation = derivative.state_generation = state_generation;
    if (batch_count == std::numeric_limits<std::uint64_t>::max()
        || primal.offsets_capacity < batch_count + 1u
        || derivative.offsets_capacity < batch_count + 1u
        || primal.offsets == nullptr || derivative.offsets == nullptr)
        return ops::condition::StatusV1::insufficient_offsets_capacity;

    for (std::uint64_t batch = 0u; batch < batch_count; ++batch) {
        const auto *bitmap = words == 0u ? nullptr : bitmap_words + batch * words;
        const auto p_begin = primal_candidates.offsets[batch];
        const auto p_end = primal_candidates.offsets[batch + 1u];
        const auto d_begin = derivative_candidates.offsets[batch];
        const auto d_end = derivative_candidates.offsets[batch + 1u];
        primal.required += count_eligible(primal_candidates.items,
            p_begin, p_end, bitmap);
        std::uint64_t union_count = 0u;
        if (!count_eligible_union(primal_candidates.items, p_begin, p_end,
                derivative_candidates.items, d_begin, d_end, bitmap, &union_count))
            return ops::condition::StatusV1::invalid_workset;
        derivative.required += union_count;
    }
    if (primal.required > primal.item_capacity
        || (primal.required != 0u && primal.items == nullptr))
        return ops::condition::StatusV1::primal_workset_overflow;
    if (derivative.required > derivative.item_capacity
        || (derivative.required != 0u && derivative.items == nullptr))
        return ops::condition::StatusV1::derivative_workset_overflow;

    primal.offsets[0] = 0u;
    derivative.offsets[0] = 0u;
    for (std::uint64_t batch = 0u; batch < batch_count; ++batch) {
        const auto *bitmap = words == 0u ? nullptr : bitmap_words + batch * words;
        const auto p_begin = primal_candidates.offsets[batch];
        const auto p_end = primal_candidates.offsets[batch + 1u];
        const auto d_begin = derivative_candidates.offsets[batch];
        const auto d_end = derivative_candidates.offsets[batch + 1u];
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
        primal.offsets[batch + 1u] = primal.written;
        derivative.offsets[batch + 1u] = derivative.written;
    }
    return ops::condition::StatusV1::success;
}

} // namespace cellerator::execution::dependency
