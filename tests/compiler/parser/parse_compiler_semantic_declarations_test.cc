#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const auto accepted = parse_semantic_declarations_v1(
        "domain gene; axis<gene> genes; state<float, gene> expression; "
        "relation<float, gene, gene> regulation; support<gene, gene> active; "
        "order<gene> packed; profile pbmc; candidate tiled; pass fuse; ir<semantic> graph;");
    assert(accepted.accepted());
    assert(accepted.declarations.size() == 10);
    assert(accepted.declarations.front().name == "gene");
    assert(accepted.declarations.front().range.begin == 0);
    assert(accepted.declarations.front().range.end == 12);
    assert(accepted.declarations.back().kind == semantic_declaration_kind_v1::ir_binding);

    const auto rejected = parse_semantic_declarations_v1(
        "domain 7gene; axis gene_axis; state<float, gene broken");
    assert(!rejected.accepted());
    assert(rejected.diagnostics.size() == 3);
    assert(rejected.diagnostics[0].range.begin == 7);
    assert(rejected.diagnostics[0].range.end == 12);
    assert(rejected.diagnostics[1].range.begin == 18);
    assert(rejected.diagnostics[2].range.end == 54);
}
