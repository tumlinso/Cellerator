#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <algorithm>
#include <array>
#include <cctype>

namespace Cellerator::compiler::frontend::parser {
namespace {

struct declaration_prefix {
    std::string_view spelling;
    semantic_declaration_kind_v1 kind;
    bool template_form;
};

constexpr std::array<declaration_prefix, 14> prefixes{{
    {"domain", semantic_declaration_kind_v1::domain, false},
    {"axis", semantic_declaration_kind_v1::axis, true},
    {"state", semantic_declaration_kind_v1::state, true},
    {"relation_structure", semantic_declaration_kind_v1::relation, true},
    {"relation_values", semantic_declaration_kind_v1::relation, true},
    {"relation", semantic_declaration_kind_v1::relation, true},
    {"active_support", semantic_declaration_kind_v1::support, true},
    {"support", semantic_declaration_kind_v1::support, true},
    {"order", semantic_declaration_kind_v1::order, true},
    {"profile", semantic_declaration_kind_v1::profile, false},
    {"field", semantic_declaration_kind_v1::field, false},
    {"candidate", semantic_declaration_kind_v1::candidate, false},
    {"pass", semantic_declaration_kind_v1::pass, false},
    {"ir", semantic_declaration_kind_v1::ir_binding, true},
}};

bool is_space(char value) noexcept {
    return std::isspace(static_cast<unsigned char>(value)) != 0;
}

bool is_identifier(std::string_view value) noexcept {
    if (value.empty() || !(std::isalpha(static_cast<unsigned char>(value.front()))
                           || value.front() == '_'))
        return false;
    return std::all_of(value.begin() + 1, value.end(), [](char character) {
        return std::isalnum(static_cast<unsigned char>(character)) != 0
            || character == '_';
    });
}

std::string_view trim(std::string_view value, std::size_t &leading) {
    leading = 0;
    while (leading < value.size() && is_space(value[leading]))
        ++leading;
    auto end = value.size();
    while (end > leading && is_space(value[end - 1]))
        --end;
    return value.substr(leading, end - leading);
}

} // namespace

declaration_parse_result_v1 parse_semantic_declarations_v1(std::string_view source) {
    declaration_parse_result_v1 result;
    std::size_t statement_begin = 0;
    while (statement_begin < source.size()) {
        const auto semicolon = source.find(';', statement_begin);
        if (semicolon == std::string_view::npos) {
            std::size_t leading = 0;
            if (!trim(source.substr(statement_begin), leading).empty())
                result.diagnostics.push_back({"declaration requires ';'",
                    {statement_begin + leading, source.size()}});
            break;
        }
        const auto raw = source.substr(statement_begin, semicolon - statement_begin);
        std::size_t leading = 0;
        const auto statement = trim(raw, leading);
        const auto absolute_begin = statement_begin + leading;
        statement_begin = semicolon + 1;
        if (statement.empty())
            continue;

        const declaration_prefix *matched = nullptr;
        for (const auto &prefix : prefixes) {
            if (statement.compare(0, prefix.spelling.size(), prefix.spelling) == 0
                && (statement.size() == prefix.spelling.size()
                    || is_space(statement[prefix.spelling.size()])
                    || statement[prefix.spelling.size()] == '<')) {
                matched = &prefix;
                break;
            }
        }
        if (matched == nullptr) {
            result.diagnostics.push_back({"not a compiler-semantic declaration",
                {absolute_begin, semicolon + 1}});
            continue;
        }

        std::size_t name_begin = matched->spelling.size();
        if (matched->template_form) {
            if (name_begin >= statement.size() || statement[name_begin] != '<') {
                result.diagnostics.push_back({"semantic type requires template arguments",
                    {absolute_begin + name_begin, absolute_begin + name_begin + 1}});
                continue;
            }
            int depth = 0;
            for (; name_begin < statement.size(); ++name_begin) {
                depth += statement[name_begin] == '<' ? 1 : 0;
                depth -= statement[name_begin] == '>' ? 1 : 0;
                if (depth == 0) {
                    ++name_begin;
                    break;
                }
            }
            if (depth != 0) {
                result.diagnostics.push_back({"unclosed semantic type arguments",
                    {absolute_begin + matched->spelling.size(), semicolon}});
                continue;
            }
        }
        while (name_begin < statement.size() && is_space(statement[name_begin]))
            ++name_begin;
        const auto name_end = statement.find_first_of(" \t\r\n=(", name_begin);
        const auto actual_end = name_end == std::string_view::npos ? statement.size() : name_end;
        const auto name = statement.substr(name_begin, actual_end - name_begin);
        if (!is_identifier(name)) {
            result.diagnostics.push_back({"declaration requires an identifier",
                {absolute_begin + name_begin, absolute_begin + actual_end}});
            continue;
        }
        result.declarations.push_back({matched->kind, std::string(name),
            std::string(statement.substr(0, name_begin)), {absolute_begin, semicolon + 1}});
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
