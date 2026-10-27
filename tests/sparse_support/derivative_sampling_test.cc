#include <Cellerator/compute/gating/derivative_sampling.hh>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace gating = cellerator::compute::gating;
namespace indexing = cellerator::compute::indexing;

namespace {

struct sample_storage {
    std::vector<std::uint64_t> offsets;
    std::vector<indexing::indexed_work_item_v1> items;
    std::vector<std::uint64_t> eligible;
    std::vector<std::uint64_t> selected;
    std::vector<std::uint64_t> remaining;
    gating::derivative_sample_output_v1 view() {
        return {offsets.data(), offsets.size(), items.data(), items.size(),
            eligible.data(), selected.data(), remaining.data(), eligible.size()};
    }
};

void preserves_admitted_work_and_reports_per_cell_coverage() {
    const std::uint64_t admitted_offsets[]{0u, 2u, 2u, 5u};
    const indexing::indexed_work_item_v1 admitted_items[]{
        {2u, 2u}, {8u, 8u}, {100u, 100u}, {200u, 200u}, {300u, 300u}};
    const std::uint64_t dormant_offsets[]{0u, 5u, 9u, 16u};
    const indexing::indexed_work_item_v1 dormant_items[]{
        {1u, 1u}, {2u, 2u}, {4u, 4u}, {6u, 6u}, {9u, 9u},
        {10u, 10u}, {11u, 11u}, {12u, 12u}, {13u, 13u},
        {100u, 100u}, {101u, 101u}, {200u, 200u}, {201u, 201u},
        {300u, 300u}, {301u, 301u}, {400u, 400u}};
    const std::uint64_t budgets[]{1u, 3u, 0u};
    const gating::derivative_sample_request_v1 request{7u, 12u, 4u, 99u, 3u,
        {admitted_offsets, admitted_items, 3u, 5u},
        {dormant_offsets, dormant_items, 3u, 16u}, budgets};
    sample_storage storage{{0u, 0u, 0u, 0u},
        std::vector<indexing::indexed_work_item_v1>(11u),
        std::vector<std::uint64_t>(3u), std::vector<std::uint64_t>(3u),
        std::vector<std::uint64_t>(3u)};
    auto output = storage.view();
    assert(gating::build_derivative_sample_v1(request, &output)
        == gating::derivative_sample_status_v1::success);
    assert(output.structure_epoch == 7u && output.state_generation == 12u);
    assert(output.policy == gating::dormant_sampling_policy_v1::biased_cyclic_v1);
    assert(output.eligible_dormant_count == 12u);
    assert(output.selected_dormant_count == 4u);
    assert(output.remaining_dormant_count == 8u);
    assert((storage.eligible == std::vector<std::uint64_t>{4u, 4u, 4u}));
    assert((storage.selected == std::vector<std::uint64_t>{1u, 3u, 0u}));
    assert((storage.remaining == std::vector<std::uint64_t>{3u, 1u, 4u}));
    assert((storage.offsets == std::vector<std::uint64_t>{0u, 3u, 6u, 9u}));
    for (std::uint64_t cell = 0u; cell < 3u; ++cell) {
        const auto begin = storage.offsets[cell];
        const auto end = storage.offsets[cell + 1u];
        for (auto i = begin + (begin < end ? 1u : 0u); i < end; ++i)
            assert(storage.items[i - 1u].logical_edge_id
                < storage.items[i].logical_edge_id);
    }
    // Every admitted interaction appears even when its ID overlaps a dormant
    // proposal, and overlap does not consume the dormant sampling budget.
    const std::vector<std::uint64_t> first_cell_ids{
        storage.items[0].logical_edge_id, storage.items[1].logical_edge_id,
        storage.items[2].logical_edge_id};
    assert(std::find(first_cell_ids.begin(), first_cell_ids.end(), 2u)
        != first_cell_ids.end());
    assert(std::find(first_cell_ids.begin(), first_cell_ids.end(), 8u)
        != first_cell_ids.end());
    assert(std::count_if(first_cell_ids.begin(), first_cell_ids.end(),
        [](std::uint64_t id) { return id != 2u && id != 8u; }) == 1);
    assert(storage.items[6].logical_edge_id == 100u
        && storage.items[7].logical_edge_id == 200u
        && storage.items[8].logical_edge_id == 300u);
}

void rotation_covers_frozen_dormant_candidates() {
    const std::uint64_t admitted_offsets[]{0u, 0u, 0u};
    const std::uint64_t dormant_offsets[]{0u, 0u, 5u};
    const indexing::indexed_work_item_v1 dormant[]{
        {10u, 0u}, {20u, 1u}, {30u, 2u}, {40u, 3u}, {50u, 4u}};
    const std::uint64_t budgets[]{0u, 2u};
    const gating::derivative_candidate_csr_view_v1 empty_admitted{
        admitted_offsets, nullptr, 2u, 0u};
    const gating::derivative_candidate_csr_view_v1 dormant_view{
        dormant_offsets, dormant, 2u, 5u};
    std::uint64_t seen = 0u;
    for (std::uint64_t window = 0u; window < 5u; ++window) {
        const gating::derivative_sample_request_v1 request{1u, 1u, window,
            123u, 2u, empty_admitted, dormant_view, budgets};
        std::uint64_t result_offsets[3]{}, eligible[2]{}, selected[2]{}, remaining[2]{};
        indexing::indexed_work_item_v1 result_items[2]{};
        gating::derivative_sample_output_v1 output{result_offsets, 3u,
            result_items, 2u, eligible, selected, remaining, 2u};
        assert(gating::build_derivative_sample_v1(request, &output)
            == gating::derivative_sample_status_v1::success);
        assert(output.written == 2u && selected[1] == 2u && remaining[1] == 3u);
        for (std::uint64_t i = result_offsets[1]; i < result_offsets[2]; ++i) {
            const auto ordinal = result_items[i].logical_edge_ordinal;
            assert(ordinal < 5u);
            seen |= std::uint64_t{1u} << ordinal;
        }
    }
    assert(seen == 0x1fu);
}

void preflights_capacities_and_rejects_duplicate_admission() {
    const std::uint64_t offsets[]{0u, 1u};
    const indexing::indexed_work_item_v1 admitted[]{ {5u, 5u} };
    const indexing::indexed_work_item_v1 dormant[]{ {8u, 8u} };
    const std::uint64_t budget[]{1u};
    const gating::derivative_sample_request_v1 request{1u, 1u, 0u, 3u, 1u,
        {offsets, admitted, 1u, 1u}, {offsets, dormant, 1u, 1u}, budget};
    std::uint64_t result_offsets[]{91u, 92u}, eligible[]{93u}, selected[]{94u}, remaining[]{95u};
    indexing::indexed_work_item_v1 result_items[]{ {96u, 96u} };
    gating::derivative_sample_output_v1 too_small{result_offsets, 2u,
        result_items, 1u, eligible, selected, remaining, 1u};
    assert(gating::build_derivative_sample_v1(request, &too_small)
        == gating::derivative_sample_status_v1::insufficient_item_capacity);
    assert(too_small.required == 2u && too_small.written == 0u);
    assert((result_offsets[0] == 91u && result_offsets[1] == 92u
        && result_items[0].logical_edge_id == 96u && eligible[0] == 93u));

    const indexing::indexed_work_item_v1 duplicate_admitted[]{ {5u, 5u}, {5u, 5u} };
    const std::uint64_t duplicate_offsets[]{0u, 2u};
    const gating::derivative_sample_request_v1 invalid{1u, 1u, 0u, 3u, 1u,
        {duplicate_offsets, duplicate_admitted, 1u, 2u},
        {offsets, dormant, 1u, 1u}, budget};
    assert(gating::build_derivative_sample_v1(invalid, &too_small)
        == gating::derivative_sample_status_v1::invalid_argument);
    assert(result_offsets[0] == 91u && result_items[0].logical_edge_id == 96u);
}

void all_active_and_zero_budget_are_exact_admitted_sets() {
    const std::uint64_t offsets[]{0u, 2u};
    const indexing::indexed_work_item_v1 admitted[]{ {3u, 3u}, {9u, 9u} };
    const indexing::indexed_work_item_v1 dormant[]{ {3u, 3u}, {9u, 9u} };
    const std::uint64_t budget[]{8u};
    const gating::derivative_sample_request_v1 request{4u, 5u, 0u, 0u, 1u,
        {offsets, admitted, 1u, 2u}, {offsets, dormant, 1u, 2u}, budget};
    std::uint64_t result_offsets[2]{}, eligible[1]{}, selected[1]{}, remaining[1]{};
    indexing::indexed_work_item_v1 result_items[2]{};
    gating::derivative_sample_output_v1 output{result_offsets, 2u, result_items,
        2u, eligible, selected, remaining, 1u};
    assert(gating::build_derivative_sample_v1(request, &output)
        == gating::derivative_sample_status_v1::success);
    assert(output.written == 2u && eligible[0] == 0u && selected[0] == 0u);
    assert(result_items[0].logical_edge_id == 3u && result_items[1].logical_edge_id == 9u);
}

} // namespace

int main() {
    preserves_admitted_work_and_reports_per_cell_coverage();
    rotation_covers_frozen_dormant_candidates();
    preflights_capacities_and_rejects_duplicate_admission();
    all_active_and_zero_budget_are_exact_admitted_sets();
}
