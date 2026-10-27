#include <Cellerator/compute/gating/condition_program.hh>
#include <Cellerator/compute/indexing/temporal_occurrences.hh>
#include <Cellerator/compute/support/adaptive_support.hh>
#include <Cellerator/compute/support/proposal_discovery.hh>

#include <cassert>
#include <cstdint>
#include <limits>

using namespace cellerator::compute;

int main() {
    const support::logical_support_edge_v1 base_edges[] = {{10u, 0u, 2u, 0u}, {20u, 2u, 2u, 0u}};
    const support::logical_support_participant_v1 base_parts[] = {
        {0u, 0u, 101u}, {1u, 0u, 201u}, {0u, 0u, 102u}, {1u, 0u, 201u}};
    const support::logical_support_edge_v1 delta_edges[] = {{30u, 0u, 2u, 0u}};
    const support::logical_support_participant_v1 delta_parts[] = {
        {0u, 0u, 101u}, {1u, 0u, 202u}};
    const support::logical_support_view_v1 base{1u, 0u, 7u, 99u, 3u,
        base_edges, 2u, base_parts, 4u};
    const support::logical_support_view_v1 delta{1u, 0u, 8u, 99u, 3u,
        delta_edges, 1u, delta_parts, 2u};
    const support::support_snapshot_pair_v1 snapshot{&base, &delta, 3u};
    assert(support::validate_support_snapshot_pair_v1(snapshot)
        == support::support_snapshot_status_v1::success);

    support::adaptive_support_manager_v1 manager;
    assert(manager.initialize(base, 8u) == support::adaptive_support_status_v1::success);
    const support::logical_support_participant_v1 promoted_parts[] = {
        {0u, 0u, 103u}, {1u, 0u, 203u}};
    assert(manager.stage_promotion({40u, promoted_parts, 2u, 0u, 11u, 4u, 0.75f})
        == support::adaptive_support_status_v1::success);
    assert(manager.stage_revocation(10u) == support::adaptive_support_status_v1::success);
    assert(manager.pin_forward_tape(3u) == support::adaptive_support_status_v1::success);
    assert(manager.publish_pending(4u)
        == support::adaptive_support_status_v1::live_forward_tapes);
    assert(manager.retire_forward_tape(3u) == support::adaptive_support_status_v1::success);
    assert(manager.publish_pending(4u) == support::adaptive_support_status_v1::success);
    support::support_snapshot_pair_v1 managed_snapshot{};
    assert(manager.snapshot(&managed_snapshot) == support::adaptive_support_status_v1::success);
    assert(support::validate_support_snapshot_pair_v1(managed_snapshot)
        == support::support_snapshot_status_v1::success);
    assert(managed_snapshot.base->edge_count == 1u
        && managed_snapshot.base->edges[0].logical_edge_id == 20u
        && managed_snapshot.delta->edge_count == 1u
        && managed_snapshot.delta->edges[0].logical_edge_id == 40u);
    assert(manager.stage_promotion({10u, promoted_parts, 2u})
        == support::adaptive_support_status_v1::duplicate_logical_edge);
    assert(manager.promotion_records().size() == 1u
        && manager.promotion_records()[0].discovery_round_id == 11u);
    assert(manager.stage_promotion({50u, promoted_parts, 2u, 0u, 12u, 5u, 0.9f})
        == support::adaptive_support_status_v1::success);
    assert(manager.stage_base_promotion(40u)
        == support::adaptive_support_status_v1::success);
    assert(manager.publish_pending(5u) == support::adaptive_support_status_v1::success);
    assert(manager.snapshot(&managed_snapshot) == support::adaptive_support_status_v1::success);
    assert(managed_snapshot.base->edge_count == 2u
        && managed_snapshot.base->edges[1].logical_edge_id == 40u
        && managed_snapshot.delta->edge_count == 1u
        && managed_snapshot.delta->edges[0].logical_edge_id == 50u);
    assert(manager.promotion_records().size() == 2u
        && manager.promotion_records()[1].scorer_generation == 5u);

    indexing::reverse_posting_record_v1 records[6]{}, entries[6]{};
    indexing::reverse_posting_view_v1 index{};
    assert(indexing::build_reverse_postings_v1(snapshot, records, 6u,
        entries, 6u, &index) == indexing::posting_status_v1::success);
    const indexing::posting_term_v1 terms[] = {{0u, 0u, 101u}, {1u, 0u, 202u}};
    indexing::indexed_work_item_v1 scratch[4]{}, output_items[2]{};
    indexing::candidate_output_v1 output{output_items, 2u, 0u, 0u};
    assert(indexing::query_reverse_postings_v1(index, terms, 2u,
        indexing::posting_match_v1::any, {scratch, 4u}, &output)
        == indexing::posting_status_v1::success);
    assert(output.written == 2u && output_items[0].logical_edge_id == 10u
        && output_items[1].logical_edge_id == 30u);
    const indexing::posting_term_v1 all_terms[] = {{0u, 0u, 101u}, {1u, 0u, 202u}};
    output.written = output.required = 0u;
    assert(indexing::query_reverse_postings_v1(index, all_terms, 2u,
        indexing::posting_match_v1::all, {scratch, 4u}, &output)
        == indexing::posting_status_v1::success);
    assert(output.written == 1u && output_items[0].logical_edge_id == 30u);
    indexing::candidate_output_v1 posting_too_small{output_items, 1u, 0u, 0u};
    assert(indexing::query_reverse_postings_v1(index, terms, 2u,
        indexing::posting_match_v1::any, {scratch, 4u}, &posting_too_small)
        == indexing::posting_status_v1::output_overflow);
    assert(posting_too_small.written == 0u && posting_too_small.required == 2u);

    const support::logical_support_edge_v1 repeated_edge[] = {{40u, 0u, 2u, 0u}};
    const support::logical_support_participant_v1 repeated_parts[] = {
        {0u, 0u, 77u}, {0u, 0u, 77u}};
    const support::logical_support_view_v1 repeated_support{1u, 0u, 9u, 99u, 4u,
        repeated_edge, 1u, repeated_parts, 2u};
    const support::support_snapshot_pair_v1 repeated_snapshot{
        &repeated_support, nullptr, 4u};
    indexing::reverse_posting_record_v1 repeated_scratch[2]{}, repeated_entries[2]{};
    indexing::reverse_posting_view_v1 repeated_index{};
    assert(indexing::build_reverse_postings_v1(repeated_snapshot,
        repeated_scratch, 2u, repeated_entries, 2u, &repeated_index)
        == indexing::posting_status_v1::success);
    assert(repeated_index.entry_count == 1u);

    support::snapshot_lifecycle_v1 lifecycle{3u, 0u};
    assert(lifecycle.pin_forward_tape(3u) == support::snapshot_lifecycle_status_v1::success);
    assert(lifecycle.publish_structure(4u) == support::snapshot_lifecycle_status_v1::live_forward_tapes);
    assert(lifecycle.retire_forward_tape(3u) == support::snapshot_lifecycle_status_v1::success);
    assert(lifecycle.publish_structure(4u) == support::snapshot_lifecycle_status_v1::success);

    indexing::temporal_occurrence_v1 storage[3]{}, occurrence_output[2]{};
    indexing::temporal_event_ring_v1 ring{};
    assert(indexing::initialize_temporal_event_ring_v1(&ring, 5u, 101u,
        storage, 3u) == indexing::occurrence_status_v1::success);
    std::uint64_t evicted = 0u;
    for (std::uint64_t i = 1u; i <= 4u; ++i) {
        const indexing::temporal_occurrence_v1 event{5u, 101u, 100u + i,
            i, static_cast<std::int64_t>(10u * i),
            indexing::occurrence_has_timestamp_v1, 0u};
        assert(indexing::append_temporal_occurrence_v1(&ring, event, i, i + 1u,
            &evicted) == indexing::occurrence_status_v1::success);
    }
    indexing::occurrence_window_v1 window{};
    window.use_ordinal_window = true;
    window.minimum_ordinal = 3u;
    window.maximum_ordinal = 4u;
    window.use_timestamp_window = true;
    window.minimum_timestamp = 30;
    window.maximum_timestamp = 40;
    indexing::occurrence_query_result_v1 occurrence_result{};
    assert(indexing::query_temporal_occurrences_v1(ring, window,
        occurrence_output, 2u, &occurrence_result)
        == indexing::occurrence_status_v1::success);
    assert(evicted == 101u && occurrence_result.written == 2u
        && occurrence_result.history_truncated
        && occurrence_output[0].occurrence_id == 103u);
    indexing::occurrence_query_result_v1 overflow_result{};
    assert(indexing::query_temporal_occurrences_v1(ring, window,
        occurrence_output, 1u, &overflow_result)
        == indexing::occurrence_status_v1::output_overflow);
    assert(overflow_result.written == 0u && overflow_result.required == 2u);

    indexing::temporal_occurrence_v1 mixed_storage[4]{}, mixed_output[3]{};
    indexing::temporal_event_ring_v1 mixed_ring{};
    assert(indexing::initialize_temporal_event_ring_v1(&mixed_ring, 6u, 101u,
        mixed_storage, 4u) == indexing::occurrence_status_v1::success);
    const indexing::temporal_occurrence_v1 untimed{6u, 101u, 201u, 0u, 0,
        0u, 0u};
    assert(indexing::append_temporal_occurrence_v1(&mixed_ring, untimed, 1u, 2u,
        &evicted) == indexing::occurrence_status_v1::success);
    for (std::uint64_t i = 2u; i <= 4u; ++i) {
        const indexing::temporal_occurrence_v1 event{6u, 101u, 200u + i, i,
            static_cast<std::int64_t>(10u * i),
            indexing::occurrence_has_timestamp_v1, 0u};
        assert(indexing::append_temporal_occurrence_v1(&mixed_ring, event, i, i + 1u,
            &evicted) == indexing::occurrence_status_v1::success);
    }
    indexing::occurrence_window_v1 recent{};
    recent.use_ordinal_window = true;
    recent.minimum_ordinal = 2u;
    recent.use_timestamp_window = true;
    recent.minimum_timestamp = 20;
    assert(indexing::query_temporal_occurrences_v1(mixed_ring, recent,
        mixed_output, 3u, &occurrence_result)
        == indexing::occurrence_status_v1::success);
    assert(occurrence_result.written == 3u && mixed_output[0].ordinal == 2u);

    using namespace gating;
    const condition_node_v1 nodes[] = {
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 0u, 0u},
        {condition_opcode_v1::predicate, 1u, 0u, 0u, 0u, 1u, 0u},
        {condition_opcode_v1::all_of, 1u, 0u, 0u, 2u, 0u, 0u}};
    const edge_condition_binding_v1 bindings[] = {{0u, 2u, 0u}, {2u, 0u, 0u}};
    const condition_program_view_v1 program{3u, nodes, 3u, bindings, 2u, 3u, 2u, 0u};
    const std::uint8_t predicates[] = {1u, 1u, 0u, 1u};
    std::uint64_t bitmaps[2]{};
    std::uint8_t node_scratch[3]{};
    assert(evaluate_condition_bitmaps_v1(program, 2u, predicates, 4u,
        bitmaps, 2u, node_scratch, 3u) == condition_status_v1::success);
    assert(bitmaps[0] == 7u && bitmaps[1] == 2u);
    std::uint32_t roots[3] = {99u, 99u, 99u};
    std::uint64_t required_roots = 0u;
    assert(prepare_condition_root_map_v1(program, roots, 3u, &required_roots)
        == condition_status_v1::success);
    assert(required_roots == 3u && roots[0] == 2u
        && roots[1] == unbound_condition_root_v1 && roots[2] == 0u);
    std::uint32_t short_roots[2] = {77u, 88u};
    assert(prepare_condition_root_map_v1(program, short_roots, 2u,
        &required_roots) == condition_status_v1::insufficient_root_map_capacity);
    assert(required_roots == 3u && short_roots[0] == 77u && short_roots[1] == 88u);
    const std::uint64_t dirty[] = {0u};
    const std::uint8_t active[] = {1u, 1u};
    assert(update_condition_bitmap_v1(program, active, 2u, dirty, 1u,
        &bitmaps[1], 1u, node_scratch, 3u) == condition_status_v1::success);
    assert(bitmaps[1] == 3u);
    const std::uint64_t invalid_dirty[] = {0u, 3u};
    const auto bitmap_before_invalid_update = bitmaps[1];
    assert(update_condition_bitmap_v1(program, active, 2u, invalid_dirty, 2u,
        &bitmaps[1], 1u, node_scratch, 3u) == condition_status_v1::invalid_argument);
    assert(bitmaps[1] == bitmap_before_invalid_update);

    const std::uint64_t offsets[] = {0u, 3u, 6u}, dormant_offsets[] = {0u, 1u, 2u};
    const indexing::indexed_work_item_v1 candidates[] = {
        {10u, 0u}, {20u, 1u}, {30u, 2u}, {10u, 0u}, {20u, 1u}, {30u, 2u}};
    const indexing::indexed_work_item_v1 dormant[] = {{30u, 2u}, {10u, 0u}};
    const per_cell_candidate_view_v1 primal_input{offsets, candidates, 6u};
    const per_cell_candidate_view_v1 derivative_input{dormant_offsets, dormant, 2u};
    std::uint64_t po[3]{}, doff[3]{};
    indexing::indexed_work_item_v1 pi[6]{}, di[6]{};
    gate_workset_pair_v1 worksets{{po, 3u, pi, 6u, 0u, 0u},
                                  {doff, 3u, di, 6u, 0u, 0u}};
    assert(build_condition_worksets_v1(program, 2u, 9u, bitmaps, 2u,
        primal_input, derivative_input, &worksets) == condition_status_v1::success);
    assert(worksets.primal.written == 5u && worksets.derivative.written == 5u);
    assert(worksets.primal.structure_epoch == 3u
        && worksets.derivative.state_generation == 9u);
    indexing::indexed_work_item_v1 tiny[1]{};
    worksets.primal.items = tiny;
    worksets.primal.item_capacity = 1u;
    assert(build_condition_worksets_v1(program, 2u, 9u, bitmaps, 2u,
        primal_input, derivative_input, &worksets)
        == condition_status_v1::primal_workset_overflow);
    assert(worksets.primal.written == 0u && worksets.primal.required == 5u);

    const std::uint64_t entities[] = {101u, 102u, 103u}, mechanisms[] = {501u, 502u};
    const support::coactivation_pair_v1 coactivation[] = {{101u, 102u}};
    support::proposal_discovery_input_v1 proposal_input{};
    proposal_input.entity_ids = entities;
    proposal_input.entity_count = 3u;
    proposal_input.mechanism_ids = mechanisms;
    proposal_input.mechanism_count = 2u;
    proposal_input.recent_coactivations = coactivation;
    proposal_input.coactivation_count = 1u;
    proposal_input.participant_count = 2u;
    const std::uint64_t posting_seeds[] = {102u, 103u};
    const support::support_proposal_v1 known_route{501u, {102u, 103u}, 2u, 0u};
    proposal_input.posting_seed_entity_ids = posting_seeds;
    proposal_input.posting_seed_entity_count = 2u;
    proposal_input.known_support = &known_route;
    proposal_input.known_support_count = 1u;
    support::support_proposal_v1 proposals[8]{};
    support::proposal_discovery_result_v1 proposal_result{};
    assert(support::propose_support_candidates_v1(proposal_input, 123u, 8u, 32u,
        proposals, 8u, &proposal_result) == support::proposal_status_v1::success);
    assert(proposal_result.attempts <= 32u
        && proposal_result.candidate_space_upper_bound == 18u
        && proposal_result.written == proposal_result.required
        && proposal_result.missed_candidate_count_exact
        && !proposal_result.candidate_space_upper_bound_saturated);
    std::uint64_t actual_unseen = 0u;
    std::uint64_t actual_proposed = 0u;
    for (const auto mechanism : mechanisms)
        for (const auto first : entities)
            for (const auto second : entities) {
                if (first == second) continue;
                bool known = mechanism == known_route.mechanism_id
                    && first == known_route.participant_ids[0]
                    && second == known_route.participant_ids[1];
                if (known) continue;
                ++actual_unseen;
                bool proposed = false;
                for (std::uint64_t i = 0u; i < proposal_result.written; ++i)
                    proposed = proposed || (proposals[i].mechanism_id == mechanism
                        && proposals[i].participant_ids[0] == first
                        && proposals[i].participant_ids[1] == second);
                if (proposed) ++actual_proposed;
            }
    const auto actual_missed = actual_unseen - actual_proposed;
    assert(proposal_result.missed_candidate_lower_bound <= actual_missed
        && proposal_result.missed_candidate_lower_bound == actual_missed);
    assert(proposal_result.sampled_coverage_lower_bound
        <= static_cast<double>(actual_proposed) / 12.0);
    for (std::uint64_t i = 0u; i < proposal_result.written; ++i)
        assert(proposals[i].mechanism_id != known_route.mechanism_id
            || proposals[i].participant_ids[0] != known_route.participant_ids[0]
            || proposals[i].participant_ids[1] != known_route.participant_ids[1]);

    const auto overflowing_bounds = support::calculate_proposal_space_bounds_v1(
        3'000'000u, 1u, 3u);
    assert(overflowing_bounds.valid && overflowing_bounds.upper_bound_saturated
        && overflowing_bounds.candidate_space_upper_bound
            == std::numeric_limits<std::uint64_t>::max()
        && !overflowing_bounds.exact_count_representable);
}
