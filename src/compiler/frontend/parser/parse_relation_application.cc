#include <Cellerator/compiler/frontend/parser/parse_relation_application_v1.hh>

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

std::string remove_comments(std::string_view source) {
    std::string result;
    for (std::size_t offset = 0; offset < source.size();) {
        if (offset + 1 < source.size() && source.substr(offset, 2) == "/*") {
            const auto close = source.find("*/", offset + 2);
            offset = close == std::string_view::npos ? source.size() : close + 2;
        } else if (offset + 1 < source.size() && source.substr(offset, 2) == "//") {
            const auto close = source.find('\n', offset + 2);
            offset = close == std::string_view::npos ? source.size() : close + 1;
            result.push_back(' ');
        } else {
            result.push_back(source[offset++]);
        }
    }
    return result;
}

relation_selector_syntax_v1 parse_selector(std::string_view spelling) {
    relation_selector_syntax_v1 selector;
    const auto normalized = remove_comments(spelling);
    const auto where = normalized.find(" where ");
    const auto on = normalized.find(" on ");
    auto relation_end = normalized.size();
    if (on != std::string::npos)
        relation_end = on;
    if (where != std::string::npos && where < relation_end)
        relation_end = where;
    selector.relation_expression = trim_copy(
        std::string_view(normalized).substr(0, relation_end));
    if (on != std::string::npos) {
        const auto end = where != std::string::npos && where > on ? where : normalized.size();
        selector.source_axis_expression = trim_copy(
            std::string_view(normalized).substr(on + 4, end - on - 4));
    }
    if (where != std::string::npos)
        selector.support_expression = trim_copy(
            std::string_view(normalized).substr(where + 7));
    selector.orientation = selector.relation_expression.compare(0, 10, "transpose(") == 0
        ? relation_orientation_v1::transpose : relation_orientation_v1::forward;
    return selector;
}

std::size_t expression_end(std::string_view source, std::size_t begin) {
    unsigned depth = 0;
    for (auto offset = begin; offset < source.size(); ++offset) {
        depth += source[offset] == '(' || source[offset] == '[' ? 1u : 0u;
        depth -= (source[offset] == ')' || source[offset] == ']') && depth ? 1u : 0u;
        if (depth == 0 && (source[offset] == ';' || source.substr(offset, 2) == "-["))
            return offset;
    }
    return source.size();
}

} // namespace

relation_parse_v1 parse_relation_applications_v1(std::string_view source) {
    relation_parse_v1 result;
    const auto first_open = source.find("-[");
    if (first_open == std::string_view::npos)
        return result;
    const auto assignment = source.rfind('=', first_open);
    std::string result_expression;
    std::string chain_source;
    relation_update_v1 update = relation_update_v1::expression;
    if (assignment != std::string_view::npos) {
        auto lhs_end = assignment;
        if (lhs_end > 0 && source[lhs_end - 1] == '+') {
            --lhs_end;
            update = relation_update_v1::accumulate;
        } else {
            update = relation_update_v1::overwrite;
        }
        result_expression = trim_copy(source.substr(0, lhs_end));
        chain_source = trim_copy(source.substr(assignment + 1, first_open - assignment - 1));
    } else {
        chain_source = trim_copy(source.substr(0, first_open));
    }

    auto open = first_open;
    while (open != std::string_view::npos) {
        const auto close = source.find("]->", open + 2);
        if (close == std::string_view::npos) {
            result.diagnostics.push_back({"unterminated relation selector", {open, source.size()}});
            break;
        }
        const auto destination_begin = close + 3;
        const auto destination_end = expression_end(source, destination_begin);
        relation_application_v1 application;
        application.result_expression = result_expression;
        application.source_expression = chain_source;
        application.selector = parse_selector(source.substr(open + 2, close - open - 2));
        application.destination_axis_expression = trim_copy(
            source.substr(destination_begin, destination_end - destination_begin));
        application.update = result.applications.empty() ? update : relation_update_v1::expression;
        application.range = {result.applications.empty() ? 0 : open, destination_end};
        if (application.source_expression.empty()
            || application.selector.relation_expression.empty()
            || application.destination_axis_expression.empty()) {
            result.diagnostics.push_back({"relation application requires source, relation, and destination",
                                          application.range});
            break;
        }
        chain_source = application.destination_axis_expression;
        result.applications.push_back(std::move(application));
        open = destination_end < source.size() && source.substr(destination_end, 2) == "-["
            ? destination_end : std::string_view::npos;
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
