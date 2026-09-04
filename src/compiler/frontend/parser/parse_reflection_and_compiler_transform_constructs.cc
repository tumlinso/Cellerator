#include <Cellerator/compiler/frontend/parser/parse_reflection_and_compiler_transform_constructs_v1.hh>

#include <algorithm>
#include <cctype>
#include <string_view>
#include <unordered_set>

namespace Cellerator::compiler::frontend::parser {
namespace {

bool identifier_character(char value) {
    return std::isalnum(static_cast<unsigned char>(value)) || value == '_';
}

std::size_t skip_space(std::string_view source, std::size_t offset) {
    while (offset < source.size()
           && std::isspace(static_cast<unsigned char>(source[offset])))
        ++offset;
    return offset;
}

std::string identifier_at(std::string_view source, std::size_t offset) {
    offset = skip_space(source, offset);
    const auto begin = offset;
    while (offset < source.size() && identifier_character(source[offset]))
        ++offset;
    return std::string(source.substr(begin, offset - begin));
}

std::size_t matching(std::string_view source, std::size_t open, char left, char right) {
    unsigned depth = 0;
    bool quoted = false;
    for (auto offset = open; offset < source.size(); ++offset) {
        if (source[offset] == '"' && (offset == 0 || source[offset - 1] != '\\'))
            quoted = !quoted;
        if (quoted)
            continue;
        if (source[offset] == left)
            ++depth;
        else if (source[offset] == right && --depth == 0)
            return offset;
    }
    return std::string_view::npos;
}

std::string level_after(std::string_view source, std::size_t marker) {
    const auto open = source.find('<', marker);
    const auto close = open == std::string_view::npos
        ? std::string_view::npos : source.find('>', open + 1);
    return open == std::string_view::npos || close == std::string_view::npos
        ? std::string{} : std::string(source.substr(open + 1, close - open - 1));
}

bool valid_level(std::string_view level) {
    static constexpr std::string_view levels[] = {
        "semantic", "geometry", "decomposition", "cover", "projection",
        "packed", "executable", "native", "evidence", "atom", "composition",
        "basis", "global_schedule", "topology"};
    return std::find(std::begin(levels), std::end(levels), level) != std::end(levels);
}

void add_level_diagnostic(compiler_transform_parse_v1 &result,
                          std::string_view level, parser_source_range_v1 range) {
    if (!valid_level(level))
        result.diagnostics.push_back({"unknown compiler IR level", range});
}

} // namespace

compiler_transform_parse_v1
parse_reflection_and_compiler_transform_constructs_v1(std::string_view source) {
    compiler_transform_parse_v1 result;
    std::unordered_set<std::string> available_transforms;
    std::size_t offset = 0;
    while (offset < source.size()) {
        offset = skip_space(source, offset);
        if (offset >= source.size())
            break;

        compiler_transform_construct_v1 construct;
        construct.range.begin = offset;
        std::size_t end = std::string_view::npos;

        if (source.compare(offset, 10, "inspect ir") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::inspect_query;
            construct.ir_level = level_after(source, offset + 8);
            end = source.find(';', offset);
        } else if (source.compare(offset, 5, "ir_of") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::ir_query;
            construct.ir_level = level_after(source, offset);
            const auto open = source.find('(', offset);
            const auto close = open == std::string_view::npos
                ? std::string_view::npos : matching(source, open, '(', ')');
            if (open != std::string_view::npos)
                construct.target = identifier_at(source, open + 1);
            end = close == std::string_view::npos ? close : source.find(';', close);
        } else if (source.compare(offset, 13, "transform ir<") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::transform_definition;
            construct.ir_level = level_after(source, offset + 10);
            const auto close = source.find('>', offset);
            construct.name = identifier_at(source, close + 1);
            const auto open = source.find('{', close);
            const auto body_close = open == std::string_view::npos
                ? std::string_view::npos : matching(source, open, '{', '}');
            if (open != std::string_view::npos && body_close != std::string_view::npos)
                construct.body = std::string(source.substr(open + 1, body_close - open - 1));
            end = body_close;
            if (!construct.name.empty())
                available_transforms.insert(construct.name);
        } else if (source.compare(offset, 5, "pass ") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::pass_definition;
            construct.name = identifier_at(source, offset + 5);
            const auto open = source.find('{', offset);
            const auto body_close = open == std::string_view::npos
                ? std::string_view::npos : matching(source, open, '{', '}');
            if (open != std::string_view::npos && body_close != std::string_view::npos)
                construct.body = std::string(source.substr(open + 1, body_close - open - 1));
            end = body_close;
            if (!construct.name.empty())
                available_transforms.insert(construct.name);
        } else if (source.compare(offset, 16, "compiler_prelude") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::compiler_prelude;
            const auto open = source.find('{', offset);
            const auto body_close = open == std::string_view::npos
                ? std::string_view::npos : matching(source, open, '{', '}');
            if (open != std::string_view::npos && body_close != std::string_view::npos)
                construct.body = std::string(source.substr(open + 1, body_close - open - 1));
            end = body_close;
            if (!construct.body.empty()) {
                const auto nested = parse_reflection_and_compiler_transform_constructs_v1(
                    construct.body);
                for (const auto &item : nested.constructs)
                    if ((item.kind == compiler_transform_construct_kind_v1::transform_definition
                         || item.kind == compiler_transform_construct_kind_v1::pass_definition)
                        && !item.name.empty())
                        available_transforms.insert(item.name);
                result.diagnostics.insert(result.diagnostics.end(), nested.diagnostics.begin(),
                                          nested.diagnostics.end());
            }
        } else if (source.compare(offset, 15, "pipeline insert") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::pipeline_insert;
            construct.name = identifier_at(source, offset + 15);
            auto relation = source.find(" before ", offset);
            if (relation != std::string_view::npos) {
                construct.insertion = pipeline_insertion_v1::before;
                construct.target = identifier_at(source, relation + 8);
            } else if ((relation = source.find(" after ", offset)) != std::string_view::npos) {
                construct.insertion = pipeline_insertion_v1::after;
                construct.target = identifier_at(source, relation + 7);
            }
            end = source.find(';', offset);
        } else if (source.compare(offset, 16, "pipeline replace") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::pipeline_replace;
            construct.target = identifier_at(source, offset + 16);
            const auto with = source.find(" with ", offset);
            if (with != std::string_view::npos)
                construct.name = identifier_at(source, with + 6);
            end = source.find(';', offset);
        } else if (source.compare(offset, 15, "apply transform") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::transform_application;
            construct.name = identifier_at(source, offset + 15);
            const auto to = source.find(" to ", offset);
            if (to != std::string_view::npos)
                construct.target = identifier_at(source, to + 4);
            end = source.find(';', offset);
            if (!available_transforms.count(construct.name))
                result.diagnostics.push_back({"transform used before it is available",
                                              {offset, end == std::string_view::npos
                                                  ? source.size() : end + 1}});
        } else if (source.compare(offset, 8, "ce::ir::") == 0) {
            construct.kind = compiler_transform_construct_kind_v1::ir_builder;
            construct.name = identifier_at(source, offset + 8);
            const auto open = source.find('(', offset);
            const auto close = open == std::string_view::npos
                ? std::string_view::npos : matching(source, open, '(', ')');
            end = close == std::string_view::npos ? close : source.find(';', close);
        } else {
            const auto next = source.find_first_of(";\n", offset);
            offset = next == std::string_view::npos ? source.size() : next + 1;
            continue;
        }

        if (end == std::string_view::npos) {
            result.diagnostics.push_back({"unterminated compiler transform construct",
                                          {offset, source.size()}});
            break;
        }
        if (!construct.ir_level.empty())
            add_level_diagnostic(result, construct.ir_level, {offset, end + 1});
        if ((construct.kind == compiler_transform_construct_kind_v1::pipeline_insert
             || construct.kind == compiler_transform_construct_kind_v1::pipeline_replace)
            && (construct.name.empty() || construct.target.empty()))
            result.diagnostics.push_back({"incomplete compiler pipeline edit", {offset, end + 1}});
        construct.range.end = end + 1;
        result.constructs.push_back(std::move(construct));
        offset = end + 1;
    }
    return result;
}

} // namespace Cellerator::compiler::frontend::parser
