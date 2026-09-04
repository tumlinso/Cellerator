#include <Cellerator/compiler/ir/common/implement_the_ceir_text_lexer_and_parser_framework_v1.hh>

#include <cctype>

namespace cellerator::compiler::ir::text {

std::vector<token> lex(std::string_view source) {
    std::vector<token> result;
    std::size_t cursor = 0u;
    while (cursor < source.size()) {
        if (std::isspace(static_cast<unsigned char>(source[cursor]))) {
            ++cursor;
            continue;
        }
        const auto begin = cursor;
        if (source[cursor] == '"') {
            ++cursor;
            bool escaped = false;
            while (cursor < source.size()) {
                const char character = source[cursor++];
                if (character == '"' && !escaped)
                    break;
                escaped = character == '\\' && !escaped;
                if (character != '\\')
                    escaped = false;
            }
            const bool closed = cursor <= source.size() && cursor > begin + 1u
                && source[cursor - 1u] == '"';
            result.push_back({closed ? token_kind::string_literal : token_kind::invalid,
                source.substr(begin, cursor - begin), {begin, cursor}});
        } else if (std::isalnum(static_cast<unsigned char>(source[cursor]))
            || source[cursor] == '_' || source[cursor] == '.') {
            while (cursor < source.size()) {
                const char character = source[cursor];
                if (!(std::isalnum(static_cast<unsigned char>(character))
                        || character == '_' || character == '.'))
                    break;
                ++cursor;
            }
            result.push_back({token_kind::word, source.substr(begin, cursor - begin),
                {begin, cursor}});
        } else {
            ++cursor;
            result.push_back({token_kind::punctuation, source.substr(begin, 1u),
                {begin, cursor}});
        }
    }
    result.push_back({token_kind::end, {}, {source.size(), source.size()}});
    return result;
}

void parser::register_dialect(std::string name, dialect_callback callback) {
    dialects_.insert_or_assign(std::move(name), std::move(callback));
}

parsed_unit parser::parse(std::string_view source) const {
    parsed_unit unit;
    const auto tokens = lex(source);
    for (std::size_t index = 0u; tokens[index].kind != token_kind::end;) {
        const auto &current = tokens[index];
        if (current.kind == token_kind::invalid) {
            unit.diagnostics.push_back({current.range, "unterminated string literal"});
            ++index;
            continue;
        }
        if (current.text == "include" || current.text == "import") {
            if (tokens[index + 1u].kind != token_kind::string_literal
                && tokens[index + 1u].kind != token_kind::word) {
                unit.diagnostics.push_back({current.range, "expected include/import target"});
                ++index;
                continue;
            }
            auto value = std::string(tokens[index + 1u].text);
            if (tokens[index + 1u].kind == token_kind::string_literal)
                value = value.substr(1u, value.size() - 2u);
            (current.text == "include" ? unit.includes : unit.imports).push_back(std::move(value));
            index += 2u;
            continue;
        }
        if (current.kind == token_kind::word) {
            const auto separator = current.text.find('.');
            if (separator == std::string_view::npos) {
                unit.diagnostics.push_back({current.range, "expected dialect.operation"});
                ++index;
                continue;
            }
            const auto dialect = std::string(current.text.substr(0u, separator));
            const auto found = dialects_.find(dialect);
            if (found == dialects_.end() || !found->second(current.text)) {
                unit.diagnostics.push_back({current.range, "unknown or rejected dialect operation"});
                ++index;
                continue;
            }
            bool inline_block = false;
            if (tokens[index + 1u].text == "{") {
                inline_block = true;
                unsigned depth = 0u;
                do {
                    if (tokens[index].text == "{") ++depth;
                    if (tokens[index].text == "}") --depth;
                    ++index;
                } while (tokens[index].kind != token_kind::end && depth != 0u);
                if (depth != 0u)
                    unit.diagnostics.push_back({current.range, "unterminated inline block"});
            } else {
                ++index;
            }
            unit.operations.push_back({std::string(current.text), current.range, inline_block});
            continue;
        }
        ++index; // recover at the next token
    }
    return unit;
}

} // namespace cellerator::compiler::ir::text
