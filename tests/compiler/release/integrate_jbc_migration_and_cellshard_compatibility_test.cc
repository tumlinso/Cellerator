#include "cmake/compiler/part_one_acceptance_v1.hh"

#include <cassert>

int main() {
    using namespace cellerator::compiler::acceptance::v1;
    assert(jbc_migration.compiler_owner == "Cellerator");
    assert(jbc_migration.storage_runtime_owner == "CellShard");
    assert(jbc_migration.source_repository == "tumlinso/CellShard");
    assert(jbc_migration.source_revision.size() == 40);
    assert(jbc_migration.compatibility_adapter);
}
