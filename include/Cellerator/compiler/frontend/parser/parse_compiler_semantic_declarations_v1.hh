#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

struct parser_source_range_v1 {
    std::size_t begin = 0;
    std::size_t end = 0;
};

enum class semantic_declaration_kind_v1 {
    domain,
    axis,
    state,
    relation,
    support,
    order,
    profile,
    field,
    candidate,
    pass,
    ir_binding
};

struct semantic_declaration_v1 {
    semantic_declaration_kind_v1 kind = semantic_declaration_kind_v1::domain;
    std::string name;
    std::string type_spelling;
    parser_source_range_v1 range{};
};

struct declaration_diagnostic_v1 {
    std::string message;
    parser_source_range_v1 range{};
};

struct declaration_parse_result_v1 {
    std::vector<semantic_declaration_v1> declarations;
    std::vector<declaration_diagnostic_v1> diagnostics;

    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] declaration_parse_result_v1 parse_semantic_declarations_v1(
    std::string_view source);

} // namespace Cellerator::compiler::frontend::parser
