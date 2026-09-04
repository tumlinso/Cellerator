#include <Cellerator/compiler/discovery/create_temporary_cellshard_compiler_compatibility_adapte_v1.hh>

#include <cassert>

cellshard::compiler::compatibility_v1::atom_persistent_identity_v1
old_consumer_identity();
Cellerator::compiler::discovery::persistent_atom_identity_v1
new_consumer_identity();

int main() {
    using namespace Cellerator::compiler::discovery;
    assert(old_consumer_identity() == new_consumer_identity());
    const auto& manifest = cellshard_compiler_compatibility_manifest_v1();
    assert(manifest.retirement_todo == "CE-CCP1-E02-018");
    assert(!cellshard_compatibility_retirement_ready_v1({1, 1, true}));
    assert(!cellshard_compatibility_retirement_ready_v1({0, 1, false}));
    assert(cellshard_compatibility_retirement_ready_v1({0, 1, true}));
}
