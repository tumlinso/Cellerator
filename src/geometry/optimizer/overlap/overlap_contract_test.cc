#include <Cellerator/geometry/optimizer/overlap/overlap_contract.hh>

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
    const std::array<source_id, 5> sources{0, 1, 1, 2, 3};
    const source_group_dictionary_view dictionary{
        offsets.data(), sources.data(), 3, 5, 4};
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
    const std::array<source_id, 2> duplicate{1, 1};
    const source_group_dictionary_view dictionary{
        offsets.data(), duplicate.data(), 1, 2, 2};
    require(validate_source_group_dictionary(dictionary).error
        == contract_error::duplicate_source_in_group);

    const std::array<source_id, 2> valid{0, 1};
    const source_group_dictionary_view valid_dictionary{
        offsets.data(), valid.data(), 1, 2, 2};
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

}  // namespace

int main() {
    test_dictionary_and_replication_cost();
    test_unique_contribution_ownership();
    test_contract_rejections();
    return 0;
}
