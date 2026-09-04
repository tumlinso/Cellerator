#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

struct biological_type_syntax_v1 {
    std::string constructor;
    std::vector<biological_type_syntax_v1> arguments;
    std::vector<std::string> qualifiers;
    std::string spelling;
};

struct biological_type_parse_v1 {
    biological_type_syntax_v1 type;
    std::string diagnostic;

    [[nodiscard]] bool accepted() const noexcept { return diagnostic.empty(); }
};

[[nodiscard]] biological_type_parse_v1 parse_biological_type_v1(
    std::string_view spelling);
[[nodiscard]] std::string render_biological_type_v1(
    const biological_type_syntax_v1 &type);

} // namespace Cellerator::compiler::frontend::parser
