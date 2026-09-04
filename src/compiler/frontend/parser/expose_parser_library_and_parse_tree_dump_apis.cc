#include <Cellerator/compiler/frontend/parser/expose_parser_library_and_parse_tree_dump_apis_v1.hh>

#include <Cellerator/compiler/frontend/parser/parse_anonymous_execution_fields_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_inline_ceir_blocks_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_named_execution_fields_and_references_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_native_backend_fragments_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_non_relation_operation_families_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_planning_facts_preferences_and_hard_constraints_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_reflection_and_compiler_transform_constructs_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_relation_application_v1.hh>

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <utility>

namespace Cellerator::compiler::frontend::parser {
namespace {

std::string spelling(std::string_view source, parser_source_range_v1 range) {
    if (range.begin > source.size() || range.end < range.begin)
        return {};
    return std::string(source.substr(range.begin,
        std::min(range.end, source.size()) - range.begin));
}

void add(std::vector<parse_tree_node_v1> &nodes, std::string kind, std::string name,
         parser_source_range_v1 range, std::string_view source) {
    nodes.push_back({std::move(kind), std::move(name), spelling(source, range), range});
}

template <class Parse>
void append_diagnostics(std::vector<declaration_diagnostic_v1> &destination,
                        const Parse &parsed) {
    destination.insert(destination.end(), parsed.diagnostics.begin(), parsed.diagnostics.end());
}

std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (ch < 0x20)
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned>(ch) << std::dec;
            else
                output << static_cast<char>(ch);
        }
    }
    return output.str();
}

void add_declaration_lines(std::vector<parse_tree_node_v1> &nodes,
                           std::string_view source) {
    static constexpr std::string_view prefixes[] = {
        "domain ", "axis ", "support ", "order ", "profile ", "candidate ",
        "state<", "relation<"};
    std::size_t begin = 0;
    while (begin < source.size()) {
        const auto line_end = source.find('\n', begin);
        const auto end = line_end == std::string_view::npos ? source.size() : line_end;
        auto first = begin;
        while (first < end && (source[first] == ' ' || source[first] == '\t'))
            ++first;
        const auto line = source.substr(first, end - first);
        for (const auto prefix : prefixes) {
            if (line.compare(0, prefix.size(), prefix) != 0)
                continue;
            auto name_begin = first + prefix.size();
            if (prefix.back() == '<') {
                const auto close = source.find('>', name_begin);
                name_begin = close == std::string_view::npos ? end : close + 1;
            }
            while (name_begin < end && (source[name_begin] == ' '
                                        || source[name_begin] == '\t'))
                ++name_begin;
            auto name_end = name_begin;
            while (name_end < end && (std::isalnum(static_cast<unsigned char>(source[name_end]))
                                      || source[name_end] == '_'))
                ++name_end;
            add(nodes, "declaration", std::string(source.substr(name_begin,
                name_end - name_begin)), {first, end}, source);
            break;
        }
        begin = line_end == std::string_view::npos ? source.size() : line_end + 1;
    }
}

} // namespace

parser_library_result_v1 parse_cellerator_source_v1(std::string_view source) {
    auto mutable_tree = std::make_shared<parse_tree_v1>();
    mutable_tree->language_revision = "0.1";
    parser_library_result_v1 result;

    add_declaration_lines(mutable_tree->nodes, source);

    if (source.find("<[") != std::string_view::npos) {
        const auto fields = parse_anonymous_execution_fields_v1(source);
        for (const auto &item : fields.fields)
            add(mutable_tree->nodes, "anonymous_field", {}, item.range, source);
    }
    if (source.find("field ") != std::string_view::npos) {
        const auto fields = parse_named_execution_fields_v1(source);
        for (const auto &item : fields.fields)
            add(mutable_tree->nodes, "named_field", item.name, item.range, source);
    }
    if (source.find("-[") != std::string_view::npos) {
        const auto relations = parse_relation_applications_v1(source);
        for (const auto &item : relations.applications)
            add(mutable_tree->nodes, "relation_application", item.selector.relation_expression,
                item.range, source);
    }
    const auto operations = parse_operation_families_v1(source);
    for (const auto &item : operations.operations)
        add(mutable_tree->nodes, "operation", item.callee, item.range, source);

    const auto planning = parse_planning_directives_v1(source);
    for (const auto &item : planning.directives)
        add(mutable_tree->nodes, "planning_directive", item.expression, item.range, source);

    if (source.find("ceir<") != std::string_view::npos) {
        const auto blocks = parse_inline_ceir_blocks_v1(source);
        append_diagnostics(result.diagnostics, blocks);
        for (const auto &item : blocks.blocks)
            add(mutable_tree->nodes, "inline_ceir", {}, item.range, source);
    }
    if (source.find("native<") != std::string_view::npos) {
        const auto fragments = parse_native_backend_fragments_v1(source);
        append_diagnostics(result.diagnostics, fragments);
        for (const auto &item : fragments.fragments)
            add(mutable_tree->nodes, "native_fragment", item.target, item.range, source);
    }
    const auto transforms = parse_reflection_and_compiler_transform_constructs_v1(source);
    append_diagnostics(result.diagnostics, transforms);
    for (const auto &item : transforms.constructs)
        add(mutable_tree->nodes, "compiler_transform", item.name, item.range, source);

    std::sort(mutable_tree->nodes.begin(), mutable_tree->nodes.end(),
              [](const auto &left, const auto &right) {
                  if (left.range.begin != right.range.begin)
                      return left.range.begin < right.range.begin;
                  if (left.range.end != right.range.end)
                      return left.range.end < right.range.end;
                  if (left.kind != right.kind)
                      return left.kind < right.kind;
                  return left.name < right.name;
              });
    mutable_tree->nodes.erase(std::unique(mutable_tree->nodes.begin(), mutable_tree->nodes.end(),
        [](const auto &left, const auto &right) {
            return left.kind == right.kind && left.name == right.name
                && left.range.begin == right.range.begin && left.range.end == right.range.end;
        }), mutable_tree->nodes.end());
    result.tree = std::move(mutable_tree);
    return result;
}

void visit_parse_tree_v1(const parse_tree_v1 &tree, parse_tree_visitor_v1 &visitor) {
    for (const auto &node : tree.nodes)
        visitor.visit(node);
}

std::string dump_parse_tree_text_v1(const parse_tree_v1 &tree) {
    std::ostringstream output;
    output << "cellerator-parse-tree " << tree.language_revision << '\n';
    for (const auto &node : tree.nodes)
        output << node.range.begin << ':' << node.range.end << ' ' << node.kind
               << " name=" << node.name << " spelling=" << std::quoted(node.spelling) << '\n';
    return output.str();
}

std::string dump_parse_tree_json_v1(const parse_tree_v1 &tree) {
    std::ostringstream output;
    output << "{\"language_revision\":\"" << json_escape(tree.language_revision)
           << "\",\"nodes\":[";
    for (std::size_t index = 0; index < tree.nodes.size(); ++index) {
        if (index)
            output << ',';
        const auto &node = tree.nodes[index];
        output << "{\"begin\":" << node.range.begin << ",\"end\":" << node.range.end
               << ",\"kind\":\"" << json_escape(node.kind) << "\",\"name\":\""
               << json_escape(node.name) << "\",\"spelling\":\""
               << json_escape(node.spelling) << "\"}";
    }
    output << "]}";
    return output.str();
}

} // namespace Cellerator::compiler::frontend::parser
