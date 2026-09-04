#pragma once

#include <cstddef>
#include <string_view>

namespace Cellerator::compiler::discovery {

struct cellshard_compiler_path_classification_v1 {
    std::string_view source_prefix;
    std::string_view source_tree_sha256;
    std::string_view cellerator_owner;
    std::size_t source_file_count;
};

struct cellshard_compiler_authority_audit_v1 {
    std::string_view repository;
    std::string_view audited_commit;
    std::size_t audited_branch_count;
    std::size_t jbc_branch_count;
    std::size_t classified_compiler_path_count;
    std::size_t unclassified_compiler_path_count;
    std::size_t production_authority_consumer_count;
    std::size_t retained_authoritative_api_count;
};

[[nodiscard]] const cellshard_compiler_path_classification_v1*
cellshard_compiler_path_classifications_v1(std::size_t* count) noexcept;

[[nodiscard]] const cellshard_compiler_authority_audit_v1&
cellshard_compiler_authority_audit_receipt_v1() noexcept;

[[nodiscard]] bool valid_cellshard_compiler_authority_audit_v1() noexcept;

}  // namespace Cellerator::compiler::discovery
