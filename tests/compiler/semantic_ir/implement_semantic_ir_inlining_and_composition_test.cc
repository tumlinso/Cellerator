#include <Cellerator/compiler/ir/semantic/implement_semantic_ir_inlining_and_composition_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

int main() {
    semantic_inline_graph_v1 callee;
    callee.identity = 10;
    callee.captures = {1, 2};
    callee.nodes = {
        {20, "relation.apply", {1, 2}, {3}, {4, 5}, 2, {100}},
        {21, "canonicalize", {3}, {4}, {4, 5}, 3, {101}},
    };
    semantic_inline_request_v1 request;
    request.caller_identity = 30;
    request.captures = {{1, 11}, {2, 12}};
    request.profile_replacement = {40, 41};
    request.minimum_generation = 5;
    request.identity_seed = 1000;
    semantic_inline_graph_v1 inlined;
    assert(inline_semantic_graph_v1(callee, request, &inlined) ==
           semantic_inline_status_v1::success);
    assert(inlined.nodes[0].identity == 1000 && inlined.nodes[1].identity == 1001);
    assert(inlined.nodes[0].operands[0] == 11 && inlined.nodes[0].operands[1] == 12);
    assert(inlined.nodes[0].generation == 5 && inlined.nodes[1].generation == 5);
    assert(inlined.nodes[0].provenance.front() == callee.identity);

    semantic_inline_graph_v1 expected;
    expected.identity = 999;
    expected.nodes = {
        {7, "relation.apply", {11, 12}, {3}, {40, 41}, 5, {10, 100}},
        {8, "canonicalize", {3}, {4}, {40, 41}, 5, {10, 101}},
    };
    const auto canonical_inlined = canonicalize_semantic_inline_graph_v1(inlined);
    const auto canonical_expected = canonicalize_semantic_inline_graph_v1(expected);
    assert(canonical_inlined && canonical_expected);
    assert(*canonical_inlined == *canonical_expected);

    request.captures.pop_back();
    assert(inline_semantic_graph_v1(callee, request, &inlined) ==
           semantic_inline_status_v1::missing_capture);

    std::cout << "nodes=2 captures_substituted=2 canonical_equivalence=true\n";
}
