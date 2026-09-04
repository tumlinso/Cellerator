#include <Cellerator/compiler/frontend/source/classify_source_inputs_independently_of_extension_v1.hh>

namespace Cellerator::compiler::frontend::source {
namespace {

constexpr bool horizontal_space(char value) noexcept {
    return value == ' ' || value == '\t' || value == '\r';
}

bool ends_with(std::string_view value, std::string_view suffix) noexcept {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

} // namespace

source_input_classification_v1 classify_source_input_v1(std::string_view path,
                                                         std::string_view bytes) noexcept {
    if (ends_with(path, ".ceir")) {
        return {source_input_mode_v1::standalone_ceir, 0, {}};
    }

    constexpr std::string_view marker = "#pragma cellerator";
    std::uint64_t line_start = 0;
    while (line_start <= bytes.size()) {
        auto cursor = line_start;
        while (cursor < bytes.size() && horizontal_space(bytes[cursor])) {
            ++cursor;
        }
        if (bytes.substr(cursor, marker.size()) == marker) {
            const auto after = cursor + marker.size();
            if (after == bytes.size() || horizontal_space(bytes[after]) || bytes[after] == '\n') {
                auto revision_begin = after;
                while (revision_begin < bytes.size() && horizontal_space(bytes[revision_begin])) {
                    ++revision_begin;
                }
                auto revision_end = bytes.find('\n', revision_begin);
                if (revision_end == std::string_view::npos) {
                    revision_end = bytes.size();
                }
                while (revision_end > revision_begin && horizontal_space(bytes[revision_end - 1])) {
                    --revision_end;
                }
                return {source_input_mode_v1::activated_cellerator,
                        static_cast<std::uint64_t>(after),
                        bytes.substr(revision_begin, revision_end - revision_begin)};
            }
        }
        const auto newline = bytes.find('\n', line_start);
        if (newline == std::string_view::npos) {
            break;
        }
        line_start = newline + 1;
    }
    return {};
}

} // namespace Cellerator::compiler::frontend::source
