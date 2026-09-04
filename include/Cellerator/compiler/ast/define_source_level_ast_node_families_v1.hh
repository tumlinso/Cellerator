#pragma once

#include <Cellerator/compiler/ast/freeze_ast_node_ownership_and_lifetime_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::ast {

enum class ast_semantic_family_v1 : std::uint16_t {
    invalid = 0,
    declaration,
    execution_field,
    operation,
    policy_directive,
    effect_contract,
    profile_binding,
    inline_ir,
    reflection,
    compiler_pass,
    native_fragment,
};

struct ast_family_contract_v1 {
    ast_semantic_family_v1 family = ast_semantic_family_v1::invalid;
    std::string_view stable_name;
    ast_node_class_v1 storage_class = ast_node_class_v1::unknown;
    bool planning_visible = false;
    bool may_form_boundary = false;
};

// Family and form are semantic identities. They intentionally do not encode a
// parser production, token spelling, Clang class, or raw address.
struct ast_semantic_node_v1 {
    ast_node_handle_v1 node{};
    ast_semantic_family_v1 family = ast_semantic_family_v1::invalid;
    std::uint16_t form = 0;
    std::uint32_t name_identity = 0;
    std::uint64_t semantic_identity = 0;
};

class ast_semantic_table_v1 {
public:
    [[nodiscard]] ast_arena_id_v1 arena_id() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const ast_semantic_node_v1* find(ast_node_handle_v1 node) const noexcept;
    [[nodiscard]] const std::vector<ast_semantic_node_v1>& records() const noexcept;

private:
    ast_arena_id_v1 arena_id_ = 0;
    std::vector<ast_semantic_node_v1> records_;
    friend std::optional<ast_semantic_table_v1>
    freeze_semantic_nodes_v1(const ast_snapshot_v1&, std::vector<ast_semantic_node_v1>,
                             std::string*);
};

[[nodiscard]] const std::vector<ast_family_contract_v1>& ast_family_contracts_v1();
[[nodiscard]] std::optional<ast_semantic_family_v1>
classify_semantic_concept_v1(std::string_view concept_name) noexcept;
[[nodiscard]] bool validate_ast_family_contracts_v1(std::string* error = nullptr);
[[nodiscard]] std::optional<ast_semantic_table_v1>
freeze_semantic_nodes_v1(const ast_snapshot_v1& syntax,
                         std::vector<ast_semantic_node_v1> records,
                         std::string* error = nullptr);

} // namespace Cellerator::compiler::ast
