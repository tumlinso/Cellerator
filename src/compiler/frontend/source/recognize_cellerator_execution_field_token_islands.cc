#include <Cellerator/compiler/frontend/source/recognize_cellerator_execution_field_token_islands_v1.hh>

namespace Cellerator::compiler::frontend::source {

field_island_scan_v1 recognize_execution_field_islands_v1(source_space_id_v1 source,
                                                            std::string_view bytes,
                                                            std::uint64_t activation_offset) {
    field_island_scan_v1 result;
    enum class lexical { code, string, character, line_comment, block_comment } state = lexical::code;
    std::vector<std::uint64_t> opens;
    for (std::uint64_t i = activation_offset; i < bytes.size(); ++i) {
        const char c = bytes[i];
        const char n = i + 1 < bytes.size() ? bytes[i + 1] : '\0';
        if (state == lexical::line_comment) { if (c == '\n') state = lexical::code; continue; }
        if (state == lexical::block_comment) { if (c == '*' && n == '/') { state = lexical::code; ++i; } continue; }
        if (state == lexical::string || state == lexical::character) {
            if (c == '\\') { ++i; continue; }
            if ((state == lexical::string && c == '"') || (state == lexical::character && c == '\'')) state = lexical::code;
            continue;
        }
        if (c == '/' && n == '/') { state = lexical::line_comment; ++i; continue; }
        if (c == '/' && n == '*') { state = lexical::block_comment; ++i; continue; }
        if (c == '"') { state = lexical::string; continue; }
        if (c == '\'') { state = lexical::character; continue; }
        if (c == '<' && n == '[') { opens.push_back(i); ++i; continue; }
        if (c == ']' && n == '>') {
            if (opens.empty()) { result.balanced = false; continue; }
            const auto begin = opens.back(); opens.pop_back();
            if (opens.empty()) result.islands.push_back({{source, begin}, {source, i + 2}});
            ++i;
        }
    }
    result.balanced = result.balanced && opens.empty() && state != lexical::block_comment;
    return result;
}

} // namespace Cellerator::compiler::frontend::source
