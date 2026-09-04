#include <Cellerator/compiler/frontend/source/recognize_relation_and_operation_token_forms_v1.hh>

#include <cctype>

namespace Cellerator::compiler::frontend::source {
namespace {
void skip_trivia(std::string_view s, std::size_t& i) {
    for (;;) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '*') {
            const auto end = s.find("*/", i + 2); i = end == std::string_view::npos ? s.size() : end + 2; continue;
        }
        break;
    }
}
}

raw_operation_scan_v1 recognize_operation_forms_v1(source_space_id_v1 source, std::string_view bytes) {
    raw_operation_scan_v1 result;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        if (bytes[i] != '-') continue;
        const auto begin = i;
        auto cursor = i + 1; skip_trivia(bytes, cursor);
        if (cursor >= bytes.size() || bytes[cursor] != '[') continue;
        const auto payload_begin = ++cursor;
        std::uint64_t brackets = 1;
        while (cursor < bytes.size() && brackets) {
            if (bytes[cursor] == '[') ++brackets;
            if (bytes[cursor] == ']') --brackets;
            ++cursor;
        }
        if (brackets) { result.recovered = false; break; }
        const auto payload_end = cursor - 1; skip_trivia(bytes, cursor);
        if (cursor >= bytes.size() || bytes[cursor] != '-') { result.recovered = false; i = payload_end; continue; }
        ++cursor; skip_trivia(bytes, cursor);
        if (cursor >= bytes.size() || bytes[cursor] != '>') { result.recovered = false; i = payload_end; continue; }
        ++cursor;
        result.forms.push_back({raw_operation_form_kind_v1::relation_transfer,
                                {{source, begin}, {source, cursor}},
                                std::string(bytes.substr(payload_begin, payload_end - payload_begin))});
        i = cursor - 1;
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::source
