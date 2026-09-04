#include <Cellerator/compiler/ir/common/implement_deterministic_canonical_printing_v1.hh>
#include <Cellerator/compiler/ir/common/implement_the_ceir_text_lexer_and_parser_framework_v1.hh>

#include <algorithm>
#include <cassert>

using namespace cellerator::compiler::ir;

int main() {
    common_operation operation;
    operation.namespace_name = "semantic";
    operation.operation_name = "apply";
    operation.attributes = {{"zeta", "2"}, {"alpha", "1"}};
    operation.unknown_extensions = {{"x.z", {0u, 0xffu}}, {"x.a", {1u, 2u}}};
    text::print_document document{1u, 3u, {operation}};
    const auto canonical = text::canonical_print(document);
    assert(canonical == "ceir 1.3\nsemantic.apply #alpha=1 #zeta=2 `x.a:0102` `x.z:00ff`\n");

    std::reverse(operation.attributes.begin(), operation.attributes.end());
    std::reverse(operation.unknown_extensions.begin(), operation.unknown_extensions.end());
    assert(text::canonical_print({1u, 3u, {operation}}) == canonical);
    text::parser parser;
    parser.register_dialect("semantic", [](std::string_view) { return true; });
    const auto first = parser.parse(canonical);
    const auto second = parser.parse(text::canonical_print(document));
    assert(first.operations.size() == second.operations.size());
    assert(first.operations.front().qualified_name == second.operations.front().qualified_name);
    assert(text::pretty_print(document).find("  semantic.apply") != std::string::npos);
}
