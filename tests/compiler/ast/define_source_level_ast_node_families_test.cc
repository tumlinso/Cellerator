#include <Cellerator/compiler/ast/define_source_level_ast_node_families_v1.hh>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace Cellerator::compiler::ast;

int main() {
    try {
        static_assert(std::is_trivially_copyable_v<ast_semantic_node_v1>);
        std::string error;
        if (!validate_ast_family_contracts_v1(&error)) {
            throw std::runtime_error(error);
        }

        constexpr std::array concepts{
            "declaration", "field", "operation", "given", "effects", "profile",
            "inline_ir", "reflection", "pass", "native",
        };
        constexpr std::array expected{
            ast_semantic_family_v1::declaration,
            ast_semantic_family_v1::execution_field,
            ast_semantic_family_v1::operation,
            ast_semantic_family_v1::policy_directive,
            ast_semantic_family_v1::effect_contract,
            ast_semantic_family_v1::profile_binding,
            ast_semantic_family_v1::inline_ir,
            ast_semantic_family_v1::reflection,
            ast_semantic_family_v1::compiler_pass,
            ast_semantic_family_v1::native_fragment,
        };

        ast_arena_v1 arena{0xC020002U};
        const auto region = arena.append_region();
        if (!region) {
            throw std::runtime_error("semantic AST region was not created");
        }
        std::vector<ast_semantic_node_v1> semantics;
        for (std::size_t index = 0; index < concepts.size(); ++index) {
            const auto classified = classify_semantic_concept_v1(concepts[index]);
            if (!classified || *classified != expected[index]) {
                throw std::runtime_error("language concept lacks one canonical family");
            }
            const auto node = arena.append_node(
                ast_family_contracts_v1()[static_cast<std::size_t>(*classified) - 1].storage_class,
                {}, *region, index + 1);
            if (!node) {
                throw std::runtime_error("semantic AST node was not created");
            }
            semantics.push_back({*node, *classified, static_cast<std::uint16_t>(index + 1),
                                 static_cast<std::uint32_t>(100 + index), 1000 + index});
        }
        auto syntax = std::move(arena).freeze();
        auto table = freeze_semantic_nodes_v1(syntax, semantics, &error);
        if (!table || table->size() != concepts.size()) {
            throw std::runtime_error(error.empty() ? "semantic table freeze failed" : error);
        }
        for (const auto& record : semantics) {
            const auto* found = table->find(record.node);
            if (found == nullptr || found->family != record.family ||
                found->semantic_identity != record.semantic_identity) {
                throw std::runtime_error("canonical semantic record lookup failed");
            }
        }

        auto duplicate = semantics;
        duplicate.back().node = duplicate.front().node;
        if (freeze_semantic_nodes_v1(syntax, std::move(duplicate), &error)) {
            throw std::runtime_error("duplicate semantic representation was accepted");
        }
        auto foreign = semantics;
        foreign.front().node.arena += 1;
        if (freeze_semantic_nodes_v1(syntax, std::move(foreign), &error)) {
            throw std::runtime_error("foreign semantic representation was accepted");
        }
        if (classify_semantic_concept_v1("parser_expression_production")) {
            throw std::runtime_error("parser production leaked into semantic families");
        }

        std::cout << "canonical_families=" << ast_family_contracts_v1().size()
                  << " semantic_nodes=" << table->size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
