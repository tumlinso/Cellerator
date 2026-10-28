#pragma once

#include <Cellerator/compiler/discovery/atom_v1.hh>
#include <Cellerator/compiler/discovery/discovery_v1.hh>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace Cellerator::compiler::discovery {

struct discovery_atom_slice_receipt_v1 {
    std::uint32_t contract_version = 0;
    std::string_view migration_manifest_interface;
    std::string_view profile_environment_interface;
    std::string_view planning_ir_interface;
    std::string_view published_interface;
    std::string_view migrated_from_repository;
    std::string_view migrated_from_commit;
    std::size_t migrated_source_record_count = 0;
    std::size_t migrated_fixture_source_file_count = 0;
    std::size_t provider_family_count = 0;
    bool exact_certification_required = false;
    bool execution_authorization_separate = false;
    bool compatibility_retirement_ready = false;
};

[[nodiscard]] const discovery_atom_slice_receipt_v1&
get_discovery_atom_slice_receipt_v1() noexcept;

[[nodiscard]] bool valid_discovery_atom_slice_receipt_v1() noexcept;

}  // namespace Cellerator::compiler::discovery
