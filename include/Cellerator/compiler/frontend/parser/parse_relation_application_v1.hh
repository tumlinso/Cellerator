#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class relation_orientation_v1 { forward, transpose };
enum class relation_update_v1 { expression, overwrite, accumulate };

struct relation_selector_syntax_v1 {
    std::string relation_expression;
    std::string source_axis_expression;
    std::string support_expression;
    relation_orientation_v1 orientation = relation_orientation_v1::forward;
};

struct relation_application_v1 {
    std::string result_expression;
    std::string source_expression;
    relation_selector_syntax_v1 selector;
    std::string destination_axis_expression;
    relation_update_v1 update = relation_update_v1::expression;
    parser_source_range_v1 range{};
};

struct relation_parse_v1 {
    std::vector<relation_application_v1> applications;
    std::vector<declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] relation_parse_v1 parse_relation_applications_v1(
    std::string_view expression);

} // namespace Cellerator::compiler::frontend::parser
