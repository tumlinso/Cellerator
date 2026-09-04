#include <Cellerator/compiler/sema/implement_support_and_logical_edge_identity_semantics_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    const exact_support_member support[]{{7, 0, 1}, {9, 1, 0}};
    assert(valid_exact_support(support, 2));
    const projection_slot_binding reordered[]{{0, 9, false}, {1, no_logical_edge, true}, {2, 7, false}};
    assert(valid_projection_slots(reordered, 3));
    auto invalid_hole = reordered[1];
    invalid_hole.edge = 8;
    assert(!valid_projection_slots(&invalid_hole, 1));

    const active_support_member active{7, 3, true};
    assert(active_support_applies(active, 7, 3));
    assert(!active_support_applies(active, 7, 4));
}
