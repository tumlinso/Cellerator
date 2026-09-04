#pragma once

#include <Cellerator/compiler/frontend/parser/parse_anonymous_execution_fields_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class field_linkage_intent_v1 { local, export_field, import_field };

struct named_execution_field_v1 {
    std::string name;
    std::string signature;
    field_linkage_intent_v1 linkage = field_linkage_intent_v1::local;
    bool forward_declaration = false;
    parser_source_range_v1 range{};
    anonymous_execution_field_v1 body;
    std::vector<std::string> references;
};

struct named_field_parse_v1 {
    std::vector<named_execution_field_v1> fields;
    std::vector<declaration_diagnostic_v1> diagnostics;

    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] named_field_parse_v1 parse_named_execution_fields_v1(
    std::string_view activated_source);

} // namespace Cellerator::compiler::frontend::parser
