#pragma once

#include <Cellerator/compiler/composition/basis_v1.hh>
#include <Cellerator/compiler/composition/grammar_v1.hh>
#include <Cellerator/compiler/program/ruleset_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::composition {

inline constexpr std::uint32_t global_compiler_migration_contract_version_v1 = 1;
inline constexpr const char* planning_ir_interface_id_v1 = "CE-CCP1-I19-PLANNING-IR";
inline constexpr const char* discovery_atom_interface_id_v1 = "CE-CCP1-I20-DISCOVERY-ATOM";
inline constexpr const char* composition_basis_interface_id_v1 = "CE-CCP1-I21-COMPOSITION-BASIS";
inline constexpr const char* program_ruleset_interface_id_v1 = "CE-CCP1-I22-PROGRAM-RULESET";

enum class migration_source_disposition_v1 : std::uint8_t {
    migrated = 0,
    compatibility_alias,
    source_absent,
    intentionally_retired,
};

struct migration_source_receipt_v1 {
    std::string source_path;
    std::string destination_path;
    std::string provenance;
    migration_source_disposition_v1 disposition = migration_source_disposition_v1::migrated;
    bool behavior_checked = false;
};

struct global_compiler_migration_receipt_v1 {
    std::uint32_t contract_version = global_compiler_migration_contract_version_v1;
    std::uint32_t planning_ir_version = 0;
    std::uint32_t discovery_atom_version = 0;
    std::vector<migration_source_receipt_v1> sources;
    bool cellerator_owns_compilation = false;
    bool cellshard_is_application_only = false;
    bool part_two_deferred = true;
};

[[nodiscard]] bool validate_global_compiler_migration_receipt_v1(
    const global_compiler_migration_receipt_v1& receipt,
    std::string* error = nullptr);

} // namespace Cellerator::compiler::composition
