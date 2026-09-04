#include <Cellerator/compiler/frontend/parser/expose_parser_library_and_parse_tree_dump_apis_v1.hh>

#include <cassert>
#include <string>
#include <vector>

using namespace Cellerator::compiler::frontend::parser;

namespace {
class collecting_visitor final : public parse_tree_visitor_v1 {
public:
    void visit(const parse_tree_node_v1 &node) override { kinds.push_back(node.kind); }
    std::vector<std::string> kinds;
};
} // namespace

int main() {
    const std::string source =
        "#pragma cellerator 0.1\n"
        "domain gene;\n"
        "axis genes : gene;\n"
        "inspect ir<semantic>;\n"
        "ceir<semantic> captures(input) results(output) validation(checked) { %0 = input }\n"
        "native<cuda> target(sm_70) inputs(input) outputs(output) effects(writes(output)) fallback(exact) {\n"
        "  kernel();\n"
        "}\n";
    const auto parsed = parse_cellerator_source_v1(source);
    assert(parsed.accepted());
    assert(parsed.tree);
    assert(parsed.tree->language_revision == "0.1");
    assert(parsed.tree->nodes.size() >= 5);

    collecting_visitor visitor;
    visit_parse_tree_v1(*parsed.tree, visitor);
    assert(visitor.kinds.size() == parsed.tree->nodes.size());

    const auto text = dump_parse_tree_text_v1(*parsed.tree);
    const auto json = dump_parse_tree_json_v1(*parsed.tree);
    assert(text == dump_parse_tree_text_v1(*parsed.tree));
    assert(json == dump_parse_tree_json_v1(*parsed.tree));
    assert(text.find("cellerator-parse-tree 0.1") == 0);
    assert(text.find("native_fragment") != std::string::npos);
    assert(json.find("\"language_revision\":\"0.1\"") != std::string::npos);
    assert(json.find("\\n") != std::string::npos);

    std::shared_ptr<const parse_tree_v1> immutable_tree = parsed.tree;
    assert(immutable_tree->nodes.front().range.begin < immutable_tree->nodes.back().range.end);
}
