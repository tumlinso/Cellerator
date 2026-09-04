#include <Cellerator/compiler/ir/planning/implement_exact_logical_coverage_nodes_v1.hh>

#include <array>
#include <cassert>

int main() {
    using namespace cellerator::compiler::ir::planning::v1;
    std::array<logical_coverage_member_v1, 4> members{{
        {1u, 11u, logical_member_kind_v1::member, logical_ownership_role_v1::exact_read, 0u, 0u},
        {2u, 12u, logical_member_kind_v1::member, logical_ownership_role_v1::replica, 0u, 1u},
        {3u, 21u, logical_member_kind_v1::edge, logical_ownership_role_v1::exclusive_output, 0u, 0u},
        {4u, 22u, logical_member_kind_v1::edge, logical_ownership_role_v1::partial_contribution, 0u, 1u}}};
    exact_logical_coverage_node_v1 coverage{{1u, 2u}, {3u, 4u}, {5u, 6u},
                                             members.data(), members.size(), 0u,
                                             {2u, 2u, 2u, 2u}, {7u, 8u}};
    assert(validate_exact_logical_coverage_node_v1(coverage) ==
           exact_coverage_status_v1::ok);

    coverage.equation.universe_members = 3u;
    assert(validate_exact_logical_coverage_node_v1(coverage) ==
           exact_coverage_status_v1::omitted_member);
    coverage.equation.universe_members = 2u;
    members[1].logical_id = members[0].logical_id;
    assert(validate_exact_logical_coverage_node_v1(coverage) ==
           exact_coverage_status_v1::duplicate_logical_id);
    members[1].logical_id = 2u;
    members[2].role = logical_ownership_role_v1::read_only_halo;
    assert(validate_exact_logical_coverage_node_v1(coverage) ==
           exact_coverage_status_v1::wrong_role);
}
