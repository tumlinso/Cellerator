#pragma once

#include <Cellerator/compiler/ast/ast_v1.hh>
#include <Cellerator/compiler/ast/diagnostic_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ast {

inline constexpr std::uint32_t ast_source_diagnostics_interface_version_v1 = 1;

struct resolved_source_capture_v1 {
    compilation_source_identity_v1 original_source{};
    ast_node_handle_v1 parse_node{};
    cxx_ast_reference_v1 resolved_cxx{};
};

struct ast_source_diagnostics_interface_v1 {
    std::uint32_t interface_version = ast_source_diagnostics_interface_version_v1;
    ast_arena_id_v1 arena = 0;
    std::uint64_t original_source_hash = 0;
    std::size_t node_count = 0;
    std::size_t resolved_capture_count = 0;
    std::size_t provenance_record_count = 0;
    std::size_t diagnostic_count = 0;
    bool incremental_identity_reuse = true;
};

[[nodiscard]] std::optional<ast_source_diagnostics_interface_v1>
freeze_ast_source_diagnostics_interface_v1(
    std::uint64_t original_source_hash,
    const ast_snapshot_v1& syntax,
    const ast_semantic_table_v1& semantics,
    const cxx_ast_reference_table_v1& cxx_references,
    const ast_query_index_v1& queries,
    const token_provenance_sidecar_v1& provenance,
    const std::vector<resolved_source_capture_v1>& captures,
    const std::vector<frontend_diagnostic_v1>& diagnostics,
    std::string* error = nullptr);

} // namespace Cellerator::compiler::ast
