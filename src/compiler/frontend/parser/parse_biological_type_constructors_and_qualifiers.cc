#include <Cellerator/compiler/frontend/parser/parse_biological_type_constructors_and_qualifiers_v1.hh>

#include <cctype>

namespace Cellerator::compiler::frontend::parser {
namespace {

std::string_view trim(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    return value;
}

std::size_t matching_angle(std::string_view source, std::size_t open) {
    unsigned depth = 0;
    for (auto offset = open; offset < source.size(); ++offset) {
        depth += source[offset] == '<' ? 1u : 0u;
        if (source[offset] == '>' && --depth == 0)
            return offset;
    }
    return std::string_view::npos;
}

std::vector<std::string_view> split_arguments(std::string_view source) {
    std::vector<std::string_view> arguments;
    unsigned angle_depth = 0;
    unsigned paren_depth = 0;
    std::size_t begin = 0;
    for (std::size_t offset = 0; offset < source.size(); ++offset) {
        angle_depth += source[offset] == '<' ? 1u : 0u;
        angle_depth -= source[offset] == '>' ? 1u : 0u;
        paren_depth += source[offset] == '(' ? 1u : 0u;
        paren_depth -= source[offset] == ')' ? 1u : 0u;
        if (source[offset] == ',' && angle_depth == 0 && paren_depth == 0) {
            arguments.push_back(trim(source.substr(begin, offset - begin)));
            begin = offset + 1;
        }
    }
    arguments.push_back(trim(source.substr(begin)));
    return arguments;
}

std::vector<std::string> parse_qualifiers(std::string_view suffix) {
    std::vector<std::string> qualifiers;
    while (!(suffix = trim(suffix)).empty()) {
        std::size_t end = 0;
        unsigned depth = 0;
        do {
            depth += suffix[end] == '(' ? 1u : 0u;
            depth -= suffix[end] == ')' ? 1u : 0u;
            ++end;
        } while (end < suffix.size()
                 && (depth != 0 || !std::isspace(static_cast<unsigned char>(suffix[end]))));
        qualifiers.emplace_back(suffix.substr(0, end));
        suffix.remove_prefix(end);
    }
    return qualifiers;
}

biological_type_parse_v1 parse_impl(std::string_view source) {
    biological_type_parse_v1 result;
    source = trim(source);
    result.type.spelling = std::string(source);
    if (source.empty()) {
        result.diagnostic = "empty biological type";
        return result;
    }
    const auto open = source.find('<');
    if (open == std::string_view::npos) {
        const auto qualifier = source.find_first_of(" \t\r\n");
        result.type.constructor = std::string(trim(source.substr(0, qualifier)));
        if (qualifier != std::string_view::npos)
            result.type.qualifiers = parse_qualifiers(source.substr(qualifier));
        return result;
    }
    result.type.constructor = std::string(trim(source.substr(0, open)));
    const auto close = matching_angle(source, open);
    if (result.type.constructor.empty() || close == std::string_view::npos) {
        result.diagnostic = "unbalanced biological type constructor";
        return result;
    }
    for (const auto argument : split_arguments(source.substr(open + 1, close - open - 1))) {
        auto parsed = parse_impl(argument);
        if (!parsed.accepted())
            return parsed;
        result.type.arguments.push_back(std::move(parsed.type));
    }
    result.type.qualifiers = parse_qualifiers(source.substr(close + 1));
    return result;
}

} // namespace

biological_type_parse_v1 parse_biological_type_v1(std::string_view spelling) {
    return parse_impl(spelling);
}

std::string render_biological_type_v1(const biological_type_syntax_v1 &type) {
    return type.spelling;
}

} // namespace Cellerator::compiler::frontend::parser
