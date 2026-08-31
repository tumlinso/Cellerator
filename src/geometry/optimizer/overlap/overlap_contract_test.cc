#include <Cellerator/geometry/optimizer/overlap/overlap_contract.hh>
#include <Cellerator/geometry/optimizer/overlap/bounded_overlap_solver.hh>
#include <Cellerator/geometry/optimizer/overlap/logical_mapping.hh>
#include <Cellerator/geometry/optimizer/overlap/work_window.hh>

#include <array>
#include <cstdlib>
#include <limits>

using namespace cellerator::geometry::optimizer::overlap;

namespace {

void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

void test_dictionary_and_replication_cost() {
    const std::array<std::uint64_t, 4> offsets{0, 2, 4, 5};
    const std::array<source_ordinal, 5> sources{0, 1, 1, 2, 3};
    const std::array<source_id, 4> identities{
        (std::uint64_t{1} << 32) + 7, (std::uint64_t{1} << 32) + 11,
        (std::uint64_t{1} << 40) + 3, (std::uint64_t{1} << 48) + 1};
    const source_group_dictionary_view dictionary{
        offsets.data(), sources.data(), identities.data(), 3, 5, 4};
    require(static_cast<bool>(validate_source_group_dictionary(dictionary)));

    std::array<std::uint64_t, 4> counts{};
    replication_cost cost;
    const replication_unit_cost unit{1, 2, 3, 4, 5, 6, 7};
    require(static_cast<bool>(evaluate_replication_cost(
        dictionary, unit, {counts.data(), counts.size()}, &cost)));
    require(cost.replicated_sources == 1);
    require(cost.replicated_memberships == 1);
    require(cost.total == 28);

    bool disjoint = true;
    require(static_cast<bool>(query_is_disjoint(
        dictionary, {counts.data(), counts.size()}, &disjoint)));
    require(!disjoint);
}

void test_unique_contribution_ownership() {
    const std::array<logical_contribution_owner, 3> owners{{{0, 0}, {1, 2}, {2, 1}}};
    require(static_cast<bool>(validate_logical_contribution_ownership(
        {owners.data(), owners.size(), owners.size()}, 3)));

    auto duplicate = owners;
    duplicate[2].contribution = 1;
    require(validate_logical_contribution_ownership(
        {duplicate.data(), duplicate.size(), duplicate.size()}, 3).error
        == contract_error::duplicate_contribution_owner);
}

void test_contract_rejections() {
    const std::array<std::uint64_t, 2> offsets{0, 2};
    const std::array<source_ordinal, 2> duplicate{1, 1};
    const std::array<source_id, 2> identities{100, 200};
    const source_group_dictionary_view dictionary{
        offsets.data(), duplicate.data(), identities.data(), 1, 2, 2};
    require(validate_source_group_dictionary(dictionary).error
        == contract_error::duplicate_source_in_group);

    const std::array<source_ordinal, 2> valid{0, 1};
    const source_group_dictionary_view valid_dictionary{
        offsets.data(), valid.data(), identities.data(), 1, 2, 2};
    std::array<std::uint64_t, 1> short_workspace{};
    replication_cost cost;
    require(evaluate_replication_cost(
        valid_dictionary, {}, {short_workspace.data(), short_workspace.size()}, &cost).error
        == contract_error::insufficient_workspace);

    std::array<std::uint64_t, 2> workspace{};
    replication_unit_cost overflow;
    overflow.source_state = std::numeric_limits<std::uint64_t>::max();
    overflow.construction = 1;
    require(evaluate_replication_cost(
        valid_dictionary, overflow, {workspace.data(), workspace.size()}, &cost).error
        == contract_error::integer_overflow);
}

void test_bounded_solver_complete_cost_and_determinism() {
    const std::array<std::uint64_t, 3> offsets{0, 2, 4};
    const std::array<source_ordinal, 4> sources{0, 1, 2, 3};
    const std::array<source_id, 4> identities{10, 20, 30, 40};
    const source_group_dictionary_view baseline{
        offsets.data(), sources.data(), identities.data(), 2, 4, 4};
    const replication_unit_cost cost{1, 1, 1, 1, 1, 1, 1};
    const std::array<overlap_proposal, 5> proposals{{
        {0, 1, 20, cost},
        {1, 1, 13, cost},
        {0, 1, 20, cost},
        {2, 0, 7, cost},
        {3, 1, 100, cost}
    }};
    std::array<std::uint64_t, 4> source_uses{};
    std::array<std::uint64_t, 2> group_sizes{};
    std::array<std::uint8_t, 5> proposal_state{};
    std::array<std::uint64_t, 2> selected{};
    bounded_overlap_result result;
    const contract_status status = solve_bounded_overlap(
        baseline, proposals.data(), proposals.size(), {2, 2, 4},
        {source_uses.data(), source_uses.size(), group_sizes.data(), group_sizes.size(),
         proposal_state.data(), proposal_state.size()},
        {selected.data(), selected.size()}, &result);
    require(static_cast<bool>(status));
    require(result.selected_count == 2);
    require(selected[0] == 0 && selected[1] == 1);
    require(result.total_predicted_benefit == 33);
    require(result.total_duplication_cost == 14);
    require(result.net_predicted_benefit == 19);
    require(result.charged_duplication.gradient_reconciliation == 2);
    require(result.rejected_duplicate_count == 2);
}

void test_zero_overlap_solver_equivalence() {
    const std::array<std::uint64_t, 3> offsets{0, 1, 2};
    const std::array<source_ordinal, 2> sources{0, 1};
    const std::array<source_id, 2> identities{10, 20};
    const source_group_dictionary_view baseline{
        offsets.data(), sources.data(), identities.data(), 2, 2, 2};
    const overlap_proposal proposal{0, 1, 100, {}};
    std::array<std::uint64_t, 2> source_uses{};
    std::array<std::uint64_t, 2> group_sizes{};
    std::array<std::uint8_t, 1> proposal_state{};
    bounded_overlap_result result;
    require(static_cast<bool>(solve_bounded_overlap(
        baseline, &proposal, 1, {0, 1, 1},
        {source_uses.data(), source_uses.size(), group_sizes.data(), group_sizes.size(),
         proposal_state.data(), proposal_state.size()},
        {}, &result)));
    require(result.selected_count == 0 && result.net_predicted_benefit == 0);
}

void test_exact_logical_value_and_gradient_maps() {
    const std::array<logical_value_location, 3> locations{{{0, 2}, {1, 0}, {2, 4}}};
    const logical_value_map_view map{locations.data(), locations.size(), 3, 5};
    std::array<std::uint8_t, 5> physical_seen{};
    std::array<std::uint8_t, 3> logical_seen{};
    require(static_cast<bool>(validate_logical_value_map(
        map, {physical_seen.data(), physical_seen.size(),
              logical_seen.data(), logical_seen.size()})));

    const std::array<double, 3> logical_values{2.0, 3.0, 5.0};
    std::array<double, 5> physical_values{-1.0, -1.0, -1.0, -1.0, -1.0};
    require(static_cast<bool>(pack_logical_values(
        map, logical_values.data(), logical_values.size(),
        physical_values.data(), physical_values.size())));
    require(physical_values[2] == 2.0 && physical_values[0] == 3.0
        && physical_values[4] == 5.0 && physical_values[1] == -1.0);

    const std::array<double, 5> physical_gradients{7.0, 0.0, 11.0, 0.0, 13.0};
    std::array<double, 3> logical_gradients{};
    require(static_cast<bool>(gather_logical_gradients(
        map, physical_gradients.data(), physical_gradients.size(),
        logical_gradients.data(), logical_gradients.size())));
    require(logical_gradients[0] == 11.0 && logical_gradients[1] == 7.0
        && logical_gradients[2] == 13.0);
}

void test_replica_gradient_reconciliation() {
    const std::array<source_replica_location, 4> replicas{{
        {0, 0, 0, true}, {0, 1, 2, false}, {1, 0, 1, true}, {2, 1, 3, true}}};
    const source_replica_map_view map{replicas.data(), replicas.size(), 3, 2, 4};
    std::array<std::uint8_t, 4> physical_seen{};
    std::array<std::uint8_t, 3> logical_seen{};
    require(static_cast<bool>(validate_source_replica_map(
        map, {physical_seen.data(), physical_seen.size(),
              logical_seen.data(), logical_seen.size()})));
    const std::array<double, 4> physical_gradients{2.0, 3.0, 5.0, 7.0};
    std::array<double, 3> logical_gradients{};
    require(static_cast<bool>(reconcile_source_gradients(
        map, physical_gradients.data(), physical_gradients.size(),
        logical_gradients.data(), logical_gradients.size())));
    require(logical_gradients[0] == 7.0 && logical_gradients[1] == 3.0
        && logical_gradients[2] == 7.0);

    auto duplicate_owner = replicas;
    duplicate_owner[1].canonical_owner = true;
    require(validate_source_replica_map(
        {duplicate_owner.data(), duplicate_owner.size(), 3, 2, 4},
        {physical_seen.data(), physical_seen.size(), logical_seen.data(), logical_seen.size()}).error
        == contract_error::duplicate_source_owner);
}

void test_work_windows_and_disjoint_fallback() {
    const std::array<std::uint64_t, 3> offsets{0, 1, 2};
    const std::array<source_ordinal, 2> sources{0, 1};
    const std::array<source_id, 2> identities{10, 20};
    const source_group_dictionary_view skeleton{
        offsets.data(), sources.data(), identities.data(), 2, 2, 2};
    const std::array<windowed_overlap_proposal, 3> proposals{{
        {{0, 1, 10, {}}, 0, 2},
        {{1, 0, 20, {}}, 1, 3},
        {{0, 1, 30, {}}, 3, 4}
    }};
    std::array<overlap_proposal, 3> filtered{};
    std::array<std::uint64_t, 3> filtered_to_original{};
    std::array<std::uint64_t, 2> filtered_selected{};
    std::array<std::uint64_t, 2> source_uses{};
    std::array<std::uint64_t, 2> group_sizes{};
    std::array<std::uint8_t, 3> proposal_state{};
    std::array<std::uint64_t, 2> selected{};
    work_window_result result;
    const work_window_workspace_view workspace{
        filtered.data(), filtered.size(), filtered_to_original.data(), filtered_to_original.size(),
        filtered_selected.data(), filtered_selected.size(),
        {source_uses.data(), source_uses.size(), group_sizes.data(), group_sizes.size(),
         proposal_state.data(), proposal_state.size()}};

    require(static_cast<bool>(solve_work_window_overlap(
        skeleton, proposals.data(), proposals.size(), 1, {2, 2, 2}, workspace,
        {selected.data(), selected.size()}, &result)));
    require(result.kind == overlap_solution_kind::bounded_overlap);
    require(result.eligible_proposal_count == 2 && result.overlap.selected_count == 2);
    require(selected[0] == 1 && selected[1] == 0);

    require(static_cast<bool>(solve_work_window_overlap(
        skeleton, proposals.data(), proposals.size(), 2, {2, 2, 2}, workspace,
        {selected.data(), selected.size()}, &result)));
    require(result.eligible_proposal_count == 1 && result.overlap.selected_count == 1);
    require(selected[0] == 1);

    require(static_cast<bool>(solve_work_window_overlap(
        skeleton, proposals.data(), proposals.size(), 0, {0, 1, 1}, workspace,
        {}, &result)));
    require(result.kind == overlap_solution_kind::disjoint_baseline);
    require(result.overlap.selected_count == 0 && result.overlap.net_predicted_benefit == 0);
}

}  // namespace

int main() {
    test_dictionary_and_replication_cost();
    test_unique_contribution_ownership();
    test_contract_rejections();
    test_bounded_solver_complete_cost_and_determinism();
    test_zero_overlap_solver_equivalence();
    test_exact_logical_value_and_gradient_maps();
    test_replica_gradient_reconciliation();
    test_work_windows_and_disjoint_fallback();
    return 0;
}
