#include <Cellerator/compiler/program/freeze_cellerator_owned_global_compiler_migration_v1.hh>

#include <cassert>
#include <string>

using namespace Cellerator::compiler::composition;

int main() {
    static_assert(grammar_contract_version_v1 == 1);
    static_assert(basis_contract_version_v1 == 1);
    static_assert(Cellerator::compiler::program::ruleset_contract_version_v1 == 1);

    global_compiler_migration_receipt_v1 receipt;
    receipt.planning_ir_version = 1;
    receipt.discovery_atom_version = 1;
    receipt.cellerator_owns_compilation = true;
    receipt.cellshard_is_application_only = true;
    receipt.sources = {
        {"planning/jbc-preledger-v1/02_INTERFACES_OWNERSHIP_AND_SOURCE_MAP.md",
         "include/Cellerator/compiler/composition/grammar_v1.hh",
         "preledger source map plus E03 port inventory",
         migration_source_disposition_v1::migrated,
         true},
        {"components/CellShard/include/CellShard/compiler/composition",
         "",
         "component absent from the M30-prepared E03 source tree",
         migration_source_disposition_v1::source_absent,
         false},
    };

    std::string error;
    assert(validate_global_compiler_migration_receipt_v1(receipt, &error));
    assert(error.empty());

    auto invalid = receipt;
    invalid.discovery_atom_version = 0;
    assert(!validate_global_compiler_migration_receipt_v1(invalid, &error));

    invalid = receipt;
    invalid.sources.push_back(invalid.sources.front());
    assert(!validate_global_compiler_migration_receipt_v1(invalid, &error));

    invalid = receipt;
    invalid.sources.front().destination_path.clear();
    assert(!validate_global_compiler_migration_receipt_v1(invalid, &error));

    invalid = receipt;
    invalid.part_two_deferred = false;
    assert(!validate_global_compiler_migration_receipt_v1(invalid, &error));
}
