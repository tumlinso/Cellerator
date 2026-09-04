#pragma once

#include <cstddef>
#include <string_view>

namespace Cellerator::compiler::discovery {

struct jbc_migration_provenance_v1 {
    std::string_view destination_stem;
    std::string_view migrated_from_repository;
    std::string_view migrated_from_commit;
    std::string_view migrated_from_path;
    std::string_view migrated_from_todos;
    std::string_view cellerator_todo;
};

[[nodiscard]] const jbc_migration_provenance_v1* jbc_discovery_migration_manifest_v1(
    std::size_t* count) noexcept;

[[nodiscard]] const jbc_migration_provenance_v1* find_jbc_migration_provenance_v1(
    std::string_view destination_path) noexcept;

[[nodiscard]] bool valid_jbc_migration_provenance_v1(
    const jbc_migration_provenance_v1& record) noexcept;

}  // namespace Cellerator::compiler::discovery
