#include <Cellerator/compiler/composition/freeze_cellerator_owned_global_compiler_migration_v1.hh>

#include <set>

namespace Cellerator::compiler::composition {
namespace {

bool fail(std::string* error, const char* message) {
    if (error != nullptr) {
        *error = message;
    }
    return false;
}

} // namespace

bool validate_global_compiler_migration_receipt_v1(
    const global_compiler_migration_receipt_v1& receipt,
    std::string* error) {
    if (receipt.contract_version != global_compiler_migration_contract_version_v1) {
        return fail(error, "unsupported migration receipt version");
    }
    if (receipt.planning_ir_version != 1 || receipt.discovery_atom_version != 1) {
        return fail(error, "frozen Planning IR and discovery/atom v1 contracts are required");
    }
    if (!receipt.cellerator_owns_compilation || !receipt.cellshard_is_application_only) {
        return fail(error, "compiler and application ownership must be explicit");
    }
    if (!receipt.part_two_deferred) {
        return fail(error, "Part Two runtime work is outside this migration");
    }
    if (receipt.sources.empty()) {
        return fail(error, "at least one source disposition is required");
    }

    std::set<std::string> source_paths;
    bool has_migrated_contract = false;
    bool has_behavior_receipt = false;
    for (const auto& source : receipt.sources) {
        if (source.source_path.empty() || source.provenance.empty()) {
            return fail(error, "source path and provenance are required");
        }
        if (!source_paths.insert(source.source_path).second) {
            return fail(error, "source dispositions must be unique");
        }
        const bool has_destination = !source.destination_path.empty();
        if ((source.disposition == migration_source_disposition_v1::migrated ||
             source.disposition == migration_source_disposition_v1::compatibility_alias) &&
            !has_destination) {
            return fail(error, "migrated sources and aliases require a destination");
        }
        if ((source.disposition == migration_source_disposition_v1::source_absent ||
             source.disposition == migration_source_disposition_v1::intentionally_retired) &&
            has_destination) {
            return fail(error, "absent or retired sources cannot claim a destination");
        }
        has_migrated_contract = has_migrated_contract ||
            source.disposition == migration_source_disposition_v1::migrated;
        has_behavior_receipt = has_behavior_receipt || source.behavior_checked;
    }
    if (!has_migrated_contract || !has_behavior_receipt) {
        return fail(error, "migration and behavior evidence are both required");
    }
    if (error != nullptr) {
        error->clear();
    }
    return true;
}

} // namespace Cellerator::compiler::composition
