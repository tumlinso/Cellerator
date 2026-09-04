#include <Cellerator/compiler/frontend/parser/parse_named_execution_fields_and_references_v1.hh>

#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace Cellerator::compiler::frontend::parser {
namespace {

bool identifier_character(char value) noexcept {
    return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

std::size_t matching_parenthesis(std::string_view source, std::size_t open) {
    unsigned depth = 0;
    for (auto offset = open; offset < source.size(); ++offset) {
        depth += source[offset] == '(' ? 1u : 0u;
        if (source[offset] == ')' && --depth == 0)
            return offset;
    }
    return std::string_view::npos;
}

std::string function_name(std::string_view prefix) {
    while (!prefix.empty() && std::isspace(static_cast<unsigned char>(prefix.back())))
        prefix.remove_suffix(1);
    auto begin = prefix.size();
    while (begin != 0 && (identifier_character(prefix[begin - 1])
                          || prefix[begin - 1] == ':'))
        --begin;
    return std::string(prefix.substr(begin));
}

field_linkage_intent_v1 linkage_before(std::string_view source, std::size_t field) {
    const auto begin = source.rfind('\n', field);
    const auto prefix = source.substr(begin == std::string_view::npos ? 0 : begin + 1,
                                      field - (begin == std::string_view::npos ? 0 : begin + 1));
    if (prefix.find("export") != std::string_view::npos)
        return field_linkage_intent_v1::export_field;
    if (prefix.find("import") != std::string_view::npos)
        return field_linkage_intent_v1::import_field;
    return field_linkage_intent_v1::local;
}

} // namespace

named_field_parse_v1 parse_named_execution_fields_v1(std::string_view source) {
    named_field_parse_v1 result;
    std::size_t search = 0;
    while ((search = source.find("field ", search)) != std::string_view::npos) {
        const auto open_paren = source.find('(', search + 6);
        if (open_paren == std::string_view::npos) {
            result.diagnostics.push_back({"field requires a function declarator",
                                          {search, source.size()}});
            break;
        }
        auto name = function_name(source.substr(search + 6, open_paren - search - 6));
        const auto close_paren = matching_parenthesis(source, open_paren);
        if (name.empty() || close_paren == std::string_view::npos) {
            result.diagnostics.push_back({"malformed named field declarator",
                                          {search, open_paren + 1}});
            search = open_paren + 1;
            continue;
        }
        named_execution_field_v1 field;
        field.name = std::move(name);
        field.signature = std::string(source.substr(search + 6, close_paren - search - 5));
        field.linkage = linkage_before(source, search);
        auto following = source.find_first_not_of(" \t\r\n", close_paren + 1);
        if (following != std::string_view::npos && source[following] == ';') {
            field.forward_declaration = true;
            field.range = {search, following + 1};
            search = following + 1;
        } else if (following != std::string_view::npos
                   && source.substr(following, 2) == "<[") {
            const auto parsed_body = parse_anonymous_execution_fields_v1(source.substr(following));
            if (!parsed_body.accepted() || parsed_body.fields.empty()) {
                result.diagnostics.push_back({"malformed named field body",
                                              {following, source.size()}});
                break;
            }
            field.body = parsed_body.fields.front();
            field.body.range.begin += following;
            field.body.range.end += following;
            field.body.content_range.begin += following;
            field.body.content_range.end += following;
            field.range = {search, field.body.range.end};
            search = field.range.end;
        } else {
            result.diagnostics.push_back({"field requires ';' or an execution body",
                                          {close_paren + 1, following + 1}});
            search = close_paren + 1;
            continue;
        }
        result.fields.push_back(std::move(field));
    }

    std::unordered_map<std::string, std::size_t> definitions;
    for (std::size_t index = 0; index < result.fields.size(); ++index) {
        auto &field = result.fields[index];
        if (!field.forward_declaration && !definitions.emplace(field.name, index).second)
            result.diagnostics.push_back({"duplicate named field definition", field.range});
    }
    for (auto &field : result.fields) {
        if (field.forward_declaration)
            continue;
        const auto body = source.substr(field.body.content_range.begin,
                                        field.body.content_range.end
                                            - field.body.content_range.begin);
        for (const auto &[name, index] : definitions) {
            (void)index;
            if (body.find(name + "(") != std::string_view::npos)
                field.references.push_back(name);
        }
        std::sort(field.references.begin(), field.references.end());
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
