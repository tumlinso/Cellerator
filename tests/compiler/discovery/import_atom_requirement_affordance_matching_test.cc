#include <Cellerator/compiler/discovery/import_atom_requirement_affordance_matching_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::discovery;

namespace {

persistent_atom_identity_v1 id(std::uint64_t value) {
    return {1, value};
}

}  // namespace

int main() {
    migrated_atom_requirement_v1 requirement{
        id(1), id(2), {id(10), id(11)}, {id(20), id(21)}, id(30), id(40),
        7, UINT64_C(0x5), 1, 2, atom_generation_policy_v1::at_least};
    migrated_atom_affordance_v1 affordance{
        id(3), id(4), id(10), id(2), id(40),
        {{id(20), id(30), 8}, {id(21), id(30), 9}}, UINT64_C(0x7), 2};
    assert(match_migrated_atom_v1(requirement, affordance).matched());

    auto wrong_order = affordance;
    wrong_order.planes[1].order_identity = id(31);
    auto result = match_migrated_atom_v1(requirement, wrong_order);
    assert(result.status == atom_match_status_v1::order_mismatch);
    assert(result.requirement_plane_index == 1);

    auto wrong_coverage = affordance;
    wrong_coverage.exact_coverage_identity = id(99);
    assert(match_migrated_atom_v1(requirement, wrong_coverage).status ==
           atom_match_status_v1::coverage_mismatch);

    auto insufficient_target = affordance;
    insufficient_target.target_capabilities = UINT64_C(0x1);
    assert(match_migrated_atom_v1(requirement, insufficient_target).status ==
           atom_match_status_v1::target_capability_mismatch);

    auto stale = affordance;
    stale.planes[0].generation = 6;
    assert(match_migrated_atom_v1(requirement, stale).status ==
           atom_match_status_v1::generation_mismatch);

    auto multi_extent = affordance;
    multi_extent.extent_count = 3;
    assert(match_migrated_atom_v1(requirement, multi_extent).status ==
           atom_match_status_v1::extent_mismatch);
}
