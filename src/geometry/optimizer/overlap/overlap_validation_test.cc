#include <Cellerator/geometry/optimizer/overlap/bounded_overlap_solver.hh>
#include <Cellerator/geometry/optimizer/overlap/logical_mapping.hh>
#include <Cellerator/geometry/optimizer/overlap/work_window.hh>

#include <array>
#include <cstdlib>
#include <type_traits>

using namespace cellerator::geometry::optimizer::overlap;

static_assert(std::is_trivially_copyable_v<source_group_dictionary_view>);
static_assert(std::is_trivially_copyable_v<logical_value_map_view>);
static_assert(std::is_trivially_copyable_v<source_replica_map_view>);
static_assert(std::is_trivially_copyable_v<overlap_proposal>);

namespace {

void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

struct solver_fixture {
    std::array<std::uint64_t, 3> offsets{0, 2, 4};
    std::array<source_ordinal, 4> ordinals{0, 1, 2, 3};
    std::array<source_id, 4> identities{
        (std::uint64_t{1} << 32) + 1,
        (std::uint64_t{1} << 40) + 2,
        (std::uint64_t{1} << 48) + 3,
        (std::uint64_t{1} << 56) + 4};
    std::array<std::uint64_t, 4> source_uses{};
    std::array<std::uint64_t, 2> group_sizes{};
    std::array<std::uint8_t, 8> proposal_state{};
    std::array<std::uint64_t, 8> selected{};

    source_group_dictionary_view dictionary() const {
        return {offsets.data(), ordinals.data(), identities.data(), 2, 4, 4};
    }

    bounded_overlap_workspace_view workspace() {
        return {source_uses.data(), source_uses.size(), group_sizes.data(), group_sizes.size(),
                proposal_state.data(), proposal_state.size()};
    }
};

void test_every_duplication_component_is_charged() {
    const std::array<replication_unit_cost, 7> costs{{
        {1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 1}}};
    for (replication_unit_cost cost : costs) {
        solver_fixture fixture;
        const overlap_proposal proposal{0, 1, 1, cost};
        bounded_overlap_result result;
        require(static_cast<bool>(solve_bounded_overlap(
            fixture.dictionary(), &proposal, 1, {1, 2, 3}, fixture.workspace(),
            {fixture.selected.data(), fixture.selected.size()}, &result)));
        require(result.selected_count == 0);
    }
}

void test_deterministic_replay_and_linear_bounded_counters() {
    const replication_unit_cost cost{1, 1, 1, 1, 1, 1, 1};
    const std::array<overlap_proposal, 8> proposals{{
        {0, 1, 20, cost}, {1, 1, 22, cost}, {2, 0, 21, cost}, {3, 0, 19, cost},
        {0, 1, 20, cost}, {1, 1, 22, cost}, {2, 0, 21, cost}, {3, 0, 19, cost}}};
    solver_fixture first;
    solver_fixture second;
    bounded_overlap_result first_result;
    bounded_overlap_result second_result;
    require(static_cast<bool>(solve_bounded_overlap(
        first.dictionary(), proposals.data(), proposals.size(), {3, 2, 4}, first.workspace(),
        {first.selected.data(), first.selected.size()}, &first_result)));
    require(static_cast<bool>(solve_bounded_overlap(
        second.dictionary(), proposals.data(), proposals.size(), {3, 2, 4}, second.workspace(),
        {second.selected.data(), second.selected.size()}, &second_result)));
    require(first_result.selected_count == second_result.selected_count);
    require(first_result.net_predicted_benefit == second_result.net_predicted_benefit);
    for (std::uint64_t index = 0; index < first_result.selected_count; ++index) {
        require(first.selected[index] == second.selected[index]);
    }
    const std::uint64_t bound = proposals.size() * (first_result.selected_count + 1);
    require(first_result.proposal_evaluations <= bound);
    require(first_result.duplicate_checks <= proposals.size() * first_result.selected_count);
}

void test_global_identity_and_local_ordinal_separation() {
    solver_fixture fixture;
    require(static_cast<bool>(validate_source_group_dictionary(fixture.dictionary())));
    require(fixture.identities[0] > (std::uint64_t{1} << 32));

    auto invalid_identities = fixture.identities;
    invalid_identities[2] = invalid_identities[1];
    source_group_dictionary_view invalid = fixture.dictionary();
    invalid.source_identities = invalid_identities.data();
    require(validate_source_group_dictionary(invalid).error
        == contract_error::duplicate_source_identity);
}

}  // namespace

int main() {
    test_every_duplication_component_is_charged();
    test_deterministic_replay_and_linear_bounded_counters();
    test_global_identity_and_local_ordinal_separation();
    return 0;
}
