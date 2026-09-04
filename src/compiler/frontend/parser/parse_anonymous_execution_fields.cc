#include <Cellerator/compiler/frontend/parser/parse_anonymous_execution_fields_v1.hh>

#include <cctype>

namespace Cellerator::compiler::frontend::parser {
namespace {

std::string trim_copy(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    return std::string(value);
}

struct scan_state {
    bool quoted = false;
    bool character = false;
    bool line_comment = false;
    bool block_comment = false;
    bool escaped = false;
};

bool syntax_active(scan_state &state, std::string_view source, std::size_t offset) {
    const char value = source[offset];
    const char next = offset + 1 < source.size() ? source[offset + 1] : '\0';
    if (state.line_comment) {
        state.line_comment = value != '\n';
        return false;
    }
    if (state.block_comment) {
        if (value == '*' && next == '/')
            state.block_comment = false;
        return false;
    }
    if (state.quoted || state.character) {
        if (state.escaped)
            state.escaped = false;
        else if (value == '\\')
            state.escaped = true;
        else if ((state.quoted && value == '"') || (state.character && value == '\'')) {
            state.quoted = false;
            state.character = false;
        }
        return false;
    }
    if (value == '/' && next == '/') {
        state.line_comment = true;
        return false;
    }
    if (value == '/' && next == '*') {
        state.block_comment = true;
        return false;
    }
    state.quoted = value == '"';
    state.character = value == '\'';
    return !state.quoted && !state.character;
}

std::size_t find_field_close(std::string_view source, std::size_t open) {
    scan_state state;
    unsigned depth = 1;
    for (std::size_t offset = open + 2; offset + 1 < source.size(); ++offset) {
        if (!syntax_active(state, source, offset))
            continue;
        const auto pair = source.substr(offset, 2);
        if (pair == "<[") {
            ++depth;
            ++offset;
        } else if (pair == "]>" && --depth == 0) {
            return offset;
        }
    }
    return std::string_view::npos;
}

std::size_t planning_separator(std::string_view content) {
    const auto first = content.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return first;
    constexpr std::string_view directives[] = {
        "given", "prefer", "require", "offer", "force", "inspect"};
    bool planning = false;
    for (const auto directive : directives)
        planning = planning || content.compare(first, directive.size(), directive) == 0;
    if (!planning)
        return std::string_view::npos;
    for (auto offset = content.find("::", first); offset != std::string_view::npos;
         offset = content.find("::", offset + 2)) {
        const bool separated_before = offset == 0
            || std::isspace(static_cast<unsigned char>(content[offset - 1]));
        const bool separated_after = offset + 2 == content.size()
            || std::isspace(static_cast<unsigned char>(content[offset + 2]));
        if (separated_before && separated_after)
            return offset;
    }
    return std::string_view::npos;
}

std::vector<std::string> split_statements(std::string_view source) {
    std::vector<std::string> result;
    scan_state state;
    unsigned parentheses = 0;
    unsigned braces = 0;
    std::size_t begin = 0;
    for (std::size_t offset = 0; offset < source.size(); ++offset) {
        if (!syntax_active(state, source, offset))
            continue;
        parentheses += source[offset] == '(' ? 1u : 0u;
        parentheses -= source[offset] == ')' && parentheses ? 1u : 0u;
        braces += source[offset] == '{' ? 1u : 0u;
        braces -= source[offset] == '}' && braces ? 1u : 0u;
        if (source[offset] == ';' && parentheses == 0 && braces == 0) {
            auto statement = trim_copy(source.substr(begin, offset - begin + 1));
            if (!statement.empty())
                result.push_back(std::move(statement));
            begin = offset + 1;
        }
    }
    auto tail = trim_copy(source.substr(begin));
    if (!tail.empty())
        result.push_back(std::move(tail));
    return result;
}

anonymous_execution_field_v1 make_field(std::string_view source,
                                        std::size_t open, std::size_t close) {
    anonymous_execution_field_v1 field;
    field.range = {open, close + 2};
    field.content_range = {open + 2, close};
    const auto content = source.substr(open + 2, close - open - 2);
    const auto separator = planning_separator(content);
    if (separator != std::string_view::npos) {
        field.planning_attributes = split_statements(content.substr(0, separator));
        field.captured_cxx_statements = split_statements(content.substr(separator + 2));
    } else {
        field.captured_cxx_statements = split_statements(content);
    }
    const auto nested = parse_anonymous_execution_fields_v1(content);
    field.nested_fields = nested.fields;
    for (auto &child : field.nested_fields) {
        child.range.begin += open + 2;
        child.range.end += open + 2;
        child.content_range.begin += open + 2;
        child.content_range.end += open + 2;
    }
    return field;
}

} // namespace

anonymous_field_parse_v1 parse_anonymous_execution_fields_v1(
    std::string_view source) {
    anonymous_field_parse_v1 result;
    scan_state state;
    for (std::size_t offset = 0; offset + 1 < source.size(); ++offset) {
        if (!syntax_active(state, source, offset) || source.substr(offset, 2) != "<[")
            continue;
        const auto close = find_field_close(source, offset);
        if (close == std::string_view::npos) {
            result.diagnostics.push_back({"unterminated execution field", {offset, source.size()}});
            break;
        }
        result.fields.push_back(make_field(source, offset, close));
        offset = close + 1;
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
