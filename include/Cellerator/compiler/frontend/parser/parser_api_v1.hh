#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

struct parse_tree_node_v1 {
    std::string kind;
    std::string name;
    std::string spelling;
    parser_source_range_v1 range{};
};

struct parse_tree_v1 {
    std::string language_revision;
    std::vector<parse_tree_node_v1> nodes;
};

struct parser_library_result_v1 {
    std::shared_ptr<const parse_tree_v1> tree;
    std::vector<declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

class parse_tree_visitor_v1 {
public:
    virtual ~parse_tree_visitor_v1() = default;
    virtual void visit(const parse_tree_node_v1 &node) = 0;
};

[[nodiscard]] parser_library_result_v1 parse_cellerator_source_v1(
    std::string_view activated_source);
void visit_parse_tree_v1(const parse_tree_v1 &tree, parse_tree_visitor_v1 &visitor);
[[nodiscard]] std::string dump_parse_tree_text_v1(const parse_tree_v1 &tree);
[[nodiscard]] std::string dump_parse_tree_json_v1(const parse_tree_v1 &tree);

} // namespace Cellerator::compiler::frontend::parser
