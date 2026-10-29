#include <Cellerator/compute/gating/derivative_sampling.hh>

#include <limits>

namespace cellerator::compute::gating {
namespace {

bool add_u64(std::uint64_t left, std::uint64_t right,
             std::uint64_t* result) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    *result = left + right;
    return true;
}

std::uint64_t mix(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

bool validate_rows(const derivative_candidate_csr_view_v1& view,
                   std::uint64_t cell_count) noexcept {
    if (view.cell_count != cell_count || view.cell_offsets == nullptr
        || (view.item_count != 0u && view.items == nullptr)
        || view.cell_offsets[0] != 0u
        || view.cell_offsets[cell_count] != view.item_count)
        return false;
    for (std::uint64_t cell = 0u; cell < cell_count; ++cell) {
        const auto begin = view.cell_offsets[cell];
        const auto end = view.cell_offsets[cell + 1u];
        if (begin > end || end > view.item_count) return false;
        for (std::uint64_t i = begin + (begin < end ? 1u : 0u); i < end; ++i)
            if (view.items[i - 1u].logical_edge_id
                    >= view.items[i].logical_edge_id)
                return false;
    }
    return true;
}

derivative_sample_status_v1 cell_dormant_counts(
    const derivative_sample_request_v1& request, std::uint64_t cell,
    std::uint64_t* eligible, std::uint64_t* selected,
    std::uint64_t* required) noexcept {
    const auto admitted_begin = request.admitted.cell_offsets[cell];
    const auto admitted_end = request.admitted.cell_offsets[cell + 1u];
    const auto dormant_begin = request.dormant.cell_offsets[cell];
    const auto dormant_end = request.dormant.cell_offsets[cell + 1u];
    std::uint64_t admitted = admitted_begin;
    std::uint64_t dormant = dormant_begin;
    std::uint64_t count = 0u;
    while (admitted < admitted_end && dormant < dormant_end) {
        const auto admitted_id = request.admitted.items[admitted].logical_edge_id;
        const auto dormant_id = request.dormant.items[dormant].logical_edge_id;
        if (admitted_id < dormant_id) {
            ++admitted;
        } else if (dormant_id < admitted_id) {
            ++count;
            ++dormant;
        } else {
            if (request.admitted.items[admitted].logical_edge_ordinal
                != request.dormant.items[dormant].logical_edge_ordinal)
                return derivative_sample_status_v1::invalid_candidates;
            ++admitted;
            ++dormant;
        }
    }
    if (dormant < dormant_end) {
        const auto tail = dormant_end - dormant;
        if (!add_u64(count, tail, &count))
            return derivative_sample_status_v1::count_overflow;
    }
    const auto budget = request.dormant_budget_by_cell[cell];
    const auto selected_count = budget < count ? budget : count;
    const auto admitted_count = admitted_end - admitted_begin;
    if (!add_u64(admitted_count, selected_count, required))
        return derivative_sample_status_v1::count_overflow;
    *eligible = count;
    *selected = selected_count;
    return derivative_sample_status_v1::success;
}

std::uint64_t rotation_start(std::uint64_t seed, std::uint64_t window,
                             std::uint64_t cell,
                             std::uint64_t candidate_count) noexcept {
    if (candidate_count == 0u) return 0u;
    const auto phase = mix(seed ^ mix(cell)) % candidate_count;
    const auto offset = window % candidate_count;
    const auto distance_to_wrap = candidate_count - phase;
    return offset >= distance_to_wrap ? offset - distance_to_wrap : phase + offset;
}

bool selected_rank(std::uint64_t rank, std::uint64_t start,
                   std::uint64_t selected_count,
                   std::uint64_t candidate_count) noexcept {
    if (selected_count == 0u) return false;
    if (selected_count == candidate_count) return true;
    const auto to_end = candidate_count - start;
    if (selected_count <= to_end)
        return rank >= start && rank - start < selected_count;
    const auto wrapped_end = selected_count - to_end;
    return rank >= start || rank < wrapped_end;
}

} // namespace

derivative_sample_status_v1 build_derivative_sample_v1(
    const derivative_sample_request_v1& request,
    derivative_sample_output_v1* output) noexcept {
    if (output == nullptr || request.structure_epoch == 0u
        || request.state_generation == 0u || request.cell_count == 0u
        || request.cell_count == std::numeric_limits<std::uint64_t>::max()
        || request.dormant_budget_by_cell == nullptr
        || !validate_rows(request.admitted, request.cell_count)
        || !validate_rows(request.dormant, request.cell_count))
        return derivative_sample_status_v1::invalid_argument;

    std::uint64_t total_required = 0u;
    std::uint64_t total_eligible = 0u;
    std::uint64_t total_selected = 0u;
    for (std::uint64_t cell = 0u; cell < request.cell_count; ++cell) {
        std::uint64_t eligible = 0u, selected = 0u, required = 0u;
        const auto count_status = cell_dormant_counts(request, cell,
            &eligible, &selected, &required);
        if (count_status != derivative_sample_status_v1::success)
            return count_status;
        if (!add_u64(total_required, required, &total_required)
            || !add_u64(total_eligible, eligible, &total_eligible)
            || !add_u64(total_selected, selected, &total_selected))
            return derivative_sample_status_v1::count_overflow;
    }

    output->written = 0u;
    output->required = total_required;
    output->eligible_dormant_count = total_eligible;
    output->selected_dormant_count = total_selected;
    output->remaining_dormant_count = total_eligible - total_selected;
    output->structure_epoch = request.structure_epoch;
    output->state_generation = request.state_generation;
    output->policy = dormant_sampling_policy_v1::biased_cyclic_v1;
    if (output->cell_offsets_capacity < request.cell_count + 1u
        || output->cell_offsets == nullptr)
        return derivative_sample_status_v1::insufficient_offsets_capacity;
    if (output->item_capacity < total_required
        || (total_required != 0u && output->items == nullptr))
        return derivative_sample_status_v1::insufficient_item_capacity;
    if (output->cell_statistics_capacity < request.cell_count
        || output->eligible_dormant_by_cell == nullptr
        || output->selected_dormant_by_cell == nullptr
        || output->remaining_dormant_by_cell == nullptr)
        return derivative_sample_status_v1::insufficient_cell_statistics_capacity;

    std::uint64_t output_cursor = 0u;
    output->cell_offsets[0] = 0u;
    for (std::uint64_t cell = 0u; cell < request.cell_count; ++cell) {
        std::uint64_t eligible_count = 0u, selected_count = 0u, required = 0u;
        const auto count_status = cell_dormant_counts(request, cell,
            &eligible_count, &selected_count, &required);
        if (count_status != derivative_sample_status_v1::success)
            return count_status;
        output->eligible_dormant_by_cell[cell] = eligible_count;
        output->selected_dormant_by_cell[cell] = selected_count;
        output->remaining_dormant_by_cell[cell] = eligible_count - selected_count;
        const auto start = rotation_start(request.seed, request.window_counter,
            cell, eligible_count);
        const auto admitted_begin = request.admitted.cell_offsets[cell];
        const auto admitted_end = request.admitted.cell_offsets[cell + 1u];
        const auto dormant_begin = request.dormant.cell_offsets[cell];
        const auto dormant_end = request.dormant.cell_offsets[cell + 1u];
        std::uint64_t admitted = admitted_begin;
        std::uint64_t dormant = dormant_begin;
        std::uint64_t eligible_rank = 0u;
        while (admitted < admitted_end || dormant < dormant_end) {
            if (dormant < dormant_end
                && (admitted == admitted_end
                    || request.dormant.items[dormant].logical_edge_id
                        < request.admitted.items[admitted].logical_edge_id)) {
                if (selected_rank(eligible_rank, start, selected_count,
                                  eligible_count))
                    output->items[output_cursor++] = request.dormant.items[dormant];
                ++eligible_rank;
                ++dormant;
            } else if (admitted < admitted_end && dormant < dormant_end
                && request.admitted.items[admitted].logical_edge_id
                    == request.dormant.items[dormant].logical_edge_id) {
                output->items[output_cursor++] = request.admitted.items[admitted];
                ++admitted;
                ++dormant;
            } else {
                output->items[output_cursor++] = request.admitted.items[admitted++];
            }
        }
        output->cell_offsets[cell + 1u] = output_cursor;
    }
    output->written = output_cursor;
    return derivative_sample_status_v1::success;
}

} // namespace cellerator::compute::gating
