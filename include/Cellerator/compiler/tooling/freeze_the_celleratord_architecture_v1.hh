#pragma once

#include <Cellerator/compiler/ast/freeze_ast_node_ownership_and_lifetime_v1.hh>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace Cellerator::compiler::tooling {

inline constexpr std::uint32_t celleratord_architecture_schema_v1 = 1;

enum class lsp_feature_owner_v1 : std::uint8_t {
    lib_cellerator_snapshot,
    upstream_clangd_worker
};

enum class clangd_worker_mode_v1 : std::uint8_t {
    supervised_process,
    reusable_upstream_components
};

enum class celleratord_architecture_status_v1 : std::uint8_t {
    valid,
    schema_mismatch,
    missing_clangd_command,
    missing_clangd_version,
    missing_snapshot_version,
    missing_source_map_version,
    permanent_fork_forbidden
};

struct celleratord_architecture_v1 {
    std::uint32_t schema_version = celleratord_architecture_schema_v1;
    std::uint32_t compiler_snapshot_version = 1;
    std::uint32_t source_map_version = 1;
    std::string clangd_command = "clangd";
    std::string clangd_version_requirement = ">=18,<19";
    clangd_worker_mode_v1 clangd_mode = clangd_worker_mode_v1::supervised_process;
    bool permanent_clang_fork = false;
    bool request_scoped_cancellation = true;
    bool restart_worker_after_protocol_failure = true;
};

struct celleratord_snapshot_ticket_v1 {
    std::uint64_t document_revision = 0;
    std::int64_t clangd_document_version = 0;
    std::shared_ptr<const ast::ast_snapshot_v1> compiler_snapshot;
    std::string source_map_identity;
};

struct celleratord_source_mapping_v1 {
    std::string generated_uri;
    std::uint64_t generated_begin = 0;
    std::uint64_t generated_end = 0;
    std::string original_uri;
    std::uint64_t original_begin = 0;
    std::uint64_t original_end = 0;
};

[[nodiscard]] celleratord_architecture_status_v1
validate_celleratord_architecture_v1(const celleratord_architecture_v1 &architecture) noexcept;
[[nodiscard]] lsp_feature_owner_v1 route_lsp_feature_v1(std::string_view method) noexcept;
[[nodiscard]] bool valid_snapshot_ticket_v1(
    const celleratord_snapshot_ticket_v1 &ticket) noexcept;
[[nodiscard]] bool valid_source_mapping_v1(
    const celleratord_source_mapping_v1 &mapping) noexcept;

} // namespace Cellerator::compiler::tooling
