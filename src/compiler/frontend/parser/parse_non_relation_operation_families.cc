#include <Cellerator/compiler/frontend/parser/parse_non_relation_operation_families_v1.hh>

#include <cctype>

namespace Cellerator::compiler::frontend::parser {
namespace {

constexpr std::array<operation_family_definition_v1, 14> families{{
    {operation_family_v1::transpose, "transpose", operation_parse_form_v1::relation_selector},
    {operation_family_v1::support_contraction, "contract_on", operation_parse_form_v1::library_lowering},
    {operation_family_v1::segment_reduce, "segment_reduce", operation_parse_form_v1::library_lowering},
    {operation_family_v1::segment_normalize, "segment_normalize", operation_parse_form_v1::library_lowering},
    {operation_family_v1::edge_map, "edge_map", operation_parse_form_v1::library_lowering},
    {operation_family_v1::edge_gate, "edge_gate", operation_parse_form_v1::library_lowering},
    {operation_family_v1::active_support_update, "update_active_support", operation_parse_form_v1::library_lowering},
    {operation_family_v1::sparse_axis_update, "sparse_update", operation_parse_form_v1::library_lowering},
    {operation_family_v1::relation_bundle, "bundle", operation_parse_form_v1::library_lowering},
    {operation_family_v1::relation_chain, "chain", operation_parse_form_v1::library_lowering},
    {operation_family_v1::relation_moments, "relation_moments", operation_parse_form_v1::library_lowering},
    {operation_family_v1::hierarchy_pool, "pool", operation_parse_form_v1::library_lowering},
    {operation_family_v1::hierarchy_broadcast, "broadcast", operation_parse_form_v1::library_lowering},
    {operation_family_v1::relation_exchange, "exchange", operation_parse_form_v1::library_lowering},
}};

std::string trim_copy(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.remove_suffix(1);
    return std::string(value);
}

std::size_t matching_parenthesis(std::string_view source, std::size_t open) {
    unsigned depth = 0;
    bool quoted = false;
    for (auto offset = open; offset < source.size(); ++offset) {
        if (source[offset] == '"' && (offset == 0 || source[offset - 1] != '\\'))
            quoted = !quoted;
        if (quoted)
            continue;
        depth += source[offset] == '(' ? 1u : 0u;
        if (source[offset] == ')' && --depth == 0)
            return offset;
    }
    return std::string_view::npos;
}

std::vector<std::string> split_arguments(std::string_view source) {
    std::vector<std::string> arguments;
    unsigned depth = 0;
    std::size_t begin = 0;
    for (std::size_t offset = 0; offset < source.size(); ++offset) {
        depth += source[offset] == '(' || source[offset] == '<' ? 1u : 0u;
        depth -= (source[offset] == ')' || source[offset] == '>') && depth ? 1u : 0u;
        if (source[offset] == ',' && depth == 0) {
            arguments.push_back(trim_copy(source.substr(begin, offset - begin)));
            begin = offset + 1;
        }
    }
    const auto tail = trim_copy(source.substr(begin));
    if (!tail.empty())
        arguments.push_back(tail);
    return arguments;
}

} // namespace

const std::array<operation_family_definition_v1, 14> &
operation_family_table_v1() noexcept {
    return families;
}

operation_family_parse_v1 parse_operation_families_v1(std::string_view source) {
    operation_family_parse_v1 result;
    for (const auto &definition : families) {
        const std::string needle = definition.family == operation_family_v1::transpose
            ? std::string(definition.spelling) + "("
            : "ce::" + std::string(definition.spelling) + "(";
        std::size_t search = 0;
        while ((search = source.find(needle, search)) != std::string_view::npos) {
            const auto open = search + needle.size() - 1;
            const auto close = matching_parenthesis(source, open);
            if (close == std::string_view::npos) {
                result.diagnostics.push_back({"unterminated semantic operation",
                                              {search, source.size()}});
                break;
            }
            result.operations.push_back({definition.family,
                std::string(definition.spelling),
                split_arguments(source.substr(open + 1, close - open - 1)),
                {search, close + 1}});
            search = close + 1;
        }
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
