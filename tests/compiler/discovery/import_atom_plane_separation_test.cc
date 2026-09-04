#include <Cellerator/compiler/discovery/import_atom_plane_separation_v1.hh>

#include <cassert>
#include <vector>

using namespace Cellerator::compiler::discovery;

namespace {

persistent_atom_identity_v1 id(std::uint64_t value) {
    return {1, value};
}

std::vector<separated_atom_plane_v1> fixture() {
    const auto structure = atom_plane_bit_v1(atom_plane_kind_v1::structure);
    const auto values = atom_plane_bit_v1(atom_plane_kind_v1::mutable_values);
    return {
        {atom_plane_kind_v1::structure, id(1), 2, 0},
        {atom_plane_kind_v1::mutable_values, id(2), 7, structure},
        {atom_plane_kind_v1::active_support, id(3), 3, structure | values},
        {atom_plane_kind_v1::gradients, id(4), 5, structure | values},
        {atom_plane_kind_v1::partials, id(5), 4, structure | values},
        {atom_plane_kind_v1::physical_views, id(6), 8, structure},
        {atom_plane_kind_v1::evidence, id(7), 6, structure},
        {atom_plane_kind_v1::lineage, id(8), 9, structure},
    };
}

}  // namespace

int main() {
    const auto planes = fixture();
    std::vector<atom_plane_reuse_v1> reuse;
    assert(evaluate_atom_plane_reuse_v1(
               planes, {{atom_plane_kind_v1::mutable_values, 7, 8}}, &reuse) ==
           atom_plane_separation_status_v1::success);
    assert(reuse[0].reusable);
    assert(!reuse[1].reusable);
    assert(!reuse[2].reusable);
    assert(!reuse[3].reusable);
    assert(!reuse[4].reusable);
    assert(reuse[5].reusable);
    assert(reuse[6].reusable);
    assert(reuse[7].reusable);

    assert(evaluate_atom_plane_reuse_v1(
               planes, {{atom_plane_kind_v1::structure, 2, 3}}, &reuse) ==
           atom_plane_separation_status_v1::success);
    for (const auto& decision : reuse) {
        assert(!decision.reusable);
    }

    auto missing = planes;
    missing.pop_back();
    assert(evaluate_atom_plane_reuse_v1(missing, {}, &reuse) ==
           atom_plane_separation_status_v1::missing_plane_kind);
    assert(evaluate_atom_plane_reuse_v1(
               planes, {{atom_plane_kind_v1::mutable_values, 6, 8}}, &reuse) ==
           atom_plane_separation_status_v1::invalid_mutation);
}
