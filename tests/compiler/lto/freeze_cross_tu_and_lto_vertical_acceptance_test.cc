#include <Cellerator/compiler/lto/freeze_cross_tu_and_lto_vertical_acceptance_v1.hh>

#include <cassert>

using namespace cellerator::compiler::lto::v1;

int main() {
    cross_tu_lto_acceptance_v1 receipt{2, true, true, true, true, true, true, true};
    assert(validate_cross_tu_lto_acceptance_v1(receipt));
    receipt.plain_cpp_coexists = false;
    assert(!validate_cross_tu_lto_acceptance_v1(receipt));

    const thin_lto_identity_v1 stable{{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    const std::vector<thin_lto_object_v1> cache{
        {{10, 1}, stable, {}, true}, {{10, 2}, stable, {}, true}};
    auto edited = cache;
    edited.front().identity.semantic = {9, 9};
    const auto incremental = plan_incremental_thin_lto_v1(edited, cache);
    assert(incremental.replan_fields.size() == 1);
    assert(incremental.reused_full_ceir.size() == 1);

    const connected_relation_chain_v1 chain{
        "gene.packed", "gene.packed", "modules", "modules",
        10, 20, 30, 40, true, true, true};
    connected_planning_result_v1 first;
    connected_planning_result_v1 second;
    assert(plan_connected_cross_tu_chain_v1(chain, &first) ==
           connected_planning_status_v1::valid);
    assert(plan_connected_cross_tu_chain_v1(chain, &second) ==
           connected_planning_status_v1::valid);
    assert(first.selected_total_ns == second.selected_total_ns);
    assert(first.persistent_order && first.shared_decomposition);
}
