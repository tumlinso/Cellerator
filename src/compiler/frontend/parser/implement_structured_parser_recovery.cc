#include <Cellerator/compiler/frontend/parser/implement_structured_parser_recovery_v1.hh>

#include <algorithm>
#include <array>
#include <utility>

namespace Cellerator::compiler::frontend::parser {
namespace {

std::string boundary_name(parser_recovery_boundary_v1 boundary) {
    switch (boundary) {
    case parser_recovery_boundary_v1::field: return "field";
    case parser_recovery_boundary_v1::declaration: return "declaration";
    case parser_recovery_boundary_v1::operation: return "operation";
    case parser_recovery_boundary_v1::qualifier: return "qualifier";
    case parser_recovery_boundary_v1::inline_ir: return "inline IR";
    }
    return "parser";
}

std::size_t earliest(std::string_view source, std::size_t begin,
                     const std::initializer_list<std::string_view> markers) {
    auto result = std::string_view::npos;
    for (const auto marker : markers) {
        const auto found = source.find(marker, begin);
        if (found != std::string_view::npos)
            result = result == std::string_view::npos ? found : std::min(result, found);
    }
    return result;
}

std::pair<std::size_t, std::size_t> synchronization(
    std::string_view source, std::size_t begin, parser_recovery_boundary_v1 boundary) {
    std::size_t marker = std::string_view::npos;
    std::size_t width = 0;
    switch (boundary) {
    case parser_recovery_boundary_v1::field:
        marker = earliest(source, begin, {"]>", "field ", "named field ", "export field ", "import field "});
        width = marker != std::string_view::npos && source.substr(marker, 2) == "]>" ? 2u : 0u;
        break;
    case parser_recovery_boundary_v1::declaration:
        marker = source.find(';', begin);
        width = marker == std::string_view::npos ? 0u : 1u;
        break;
    case parser_recovery_boundary_v1::operation:
        marker = earliest(source, begin, {";", "\n", "]>"});
        width = marker == std::string_view::npos ? 0u
            : source.substr(marker, 2) == "]>" ? 2u : 1u;
        break;
    case parser_recovery_boundary_v1::qualifier:
        marker = earliest(source, begin, {",", ">", ")", ";"});
        width = marker == std::string_view::npos ? 0u : 1u;
        break;
    case parser_recovery_boundary_v1::inline_ir:
        marker = earliest(source, std::min(source.size(), begin + 1),
                          {"}", "ceir<", "]>"});
        width = marker == std::string_view::npos ? 0u
            : source.substr(marker, 5) == "ceir<" ? 0u
            : source.substr(marker, 2) == "]>" ? 2u : 1u;
        break;
    }
    if (marker == std::string_view::npos)
        return {source.size(), 0u};
    return {marker, width};
}

} // namespace

parser_recovery_result_v1 recover_parser_v1(
    std::string_view source,
    std::size_t error_offset,
    parser_recovery_boundary_v1 boundary,
    std::size_t max_notes) {
    error_offset = std::min(error_offset, source.size());
    const auto sync = synchronization(source, error_offset, boundary);
    parser_recovery_result_v1 result;
    result.boundary = boundary;
    result.resume_offset = std::min(source.size(), sync.first + sync.second);
    result.primary = {"malformed " + boundary_name(boundary) + " construct",
                      {error_offset, result.resume_offset}};

    if (max_notes != 0 && sync.first > error_offset)
        result.notes.push_back({"discarded malformed input up to the next "
                                    + boundary_name(boundary) + " boundary",
                                {error_offset, sync.first}});
    if (result.notes.size() < max_notes && sync.first == source.size())
        result.notes.push_back({"no later synchronization boundary was found",
                                {source.size(), source.size()}});
    if (result.notes.size() < max_notes && sync.first < source.size())
        result.notes.push_back({"parsing resumes at a stable boundary",
                                {result.resume_offset, result.resume_offset}});
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
