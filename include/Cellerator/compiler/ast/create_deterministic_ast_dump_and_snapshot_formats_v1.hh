#pragma once

#include <Cellerator/compiler/ast/assign_deterministic_source_identities_v1.hh>
#include <Cellerator/compiler/ast/define_source_level_ast_node_families_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ast {

inline constexpr std::uint32_t ast_dump_schema_version_v1 = 1;

struct ast_dump_node_v1 {
    std::uint64_t semantic_identity = 0;
    std::uint64_t parent_semantic_identity = 0;
    compilation_source_identity_v1 source_identity{};
    ast_semantic_family_v1 family = ast_semantic_family_v1::invalid;
    std::uint16_t form = 0;
    std::string name;
};

struct ast_dump_document_v1 {
    std::uint32_t schema_version = ast_dump_schema_version_v1;
    std::uint32_t language_revision = 1;
    std::vector<ast_dump_node_v1> nodes;
};

[[nodiscard]] std::optional<ast_dump_document_v1>
canonicalize_ast_dump_v1(ast_dump_document_v1 document, std::string* error = nullptr);
[[nodiscard]] std::string render_ast_text_v1(const ast_dump_document_v1& document);
[[nodiscard]] std::string render_ast_json_v1(const ast_dump_document_v1& document);

} // namespace Cellerator::compiler::ast
