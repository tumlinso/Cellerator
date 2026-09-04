#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

struct anonymous_execution_field_v1 {
    parser_source_range_v1 range{};
    parser_source_range_v1 content_range{};
    std::vector<std::string> planning_attributes;
    std::vector<std::string> captured_cxx_statements;
    std::vector<anonymous_execution_field_v1> nested_fields;
};

struct anonymous_field_parse_v1 {
    std::vector<anonymous_execution_field_v1> fields;
    std::vector<declaration_diagnostic_v1> diagnostics;

    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] anonymous_field_parse_v1 parse_anonymous_execution_fields_v1(
    std::string_view activated_source);

} // namespace Cellerator::compiler::frontend::parser
