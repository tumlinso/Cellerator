#include <Cellerator/compiler/ast/create_deterministic_ast_dump_and_snapshot_formats_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ast;

int main() {
    const ast_dump_node_v1 field{10, 0, {100, 101},
                                 ast_semantic_family_v1::execution_field, 1, "propagate"};
    const ast_dump_node_v1 operation{20, 10, {200, 201},
                                     ast_semantic_family_v1::operation, 7, "gene transfer"};
    std::string error;
    auto first = canonicalize_ast_dump_v1({1, 1, {operation, field}}, &error);
    auto second = canonicalize_ast_dump_v1({1, 1, {field, operation}}, &error);
    assert(first && second && error.empty());
    const std::string expected_text =
        "cellerator-ast-dump-v1 language-revision=1\n"
        "node 10 parent=0 family=execution_field form=1 source=100:101 name=\"propagate\"\n"
        "node 20 parent=10 family=operation form=7 source=200:201 name=\"gene transfer\"\n";
    const std::string expected_json =
        "{\"schema\":\"cellerator.ast.snapshot\",\"version\":1,\"languageRevision\":1,\"nodes\":["
        "{\"id\":10,\"parent\":0,\"family\":\"execution_field\",\"form\":1,\"source\":{\"high\":100,\"low\":101},\"name\":\"propagate\"},"
        "{\"id\":20,\"parent\":10,\"family\":\"operation\",\"form\":7,\"source\":{\"high\":200,\"low\":201},\"name\":\"gene transfer\"}]}";
    assert(render_ast_text_v1(*first) == expected_text);
    assert(render_ast_text_v1(*second) == expected_text);
    assert(render_ast_json_v1(*first) == expected_json);
    assert(render_ast_json_v1(*second) == expected_json);
    assert(expected_text.find("0x") == std::string::npos);
    assert(expected_json.find("address") == std::string::npos);

    auto invalid = *first;
    invalid.nodes[1].parent_semantic_identity = 999;
    assert(!canonicalize_ast_dump_v1(std::move(invalid), &error));
    std::cout << "text_bytes=" << expected_text.size() << " json_bytes=" << expected_json.size() << '\n';
}
