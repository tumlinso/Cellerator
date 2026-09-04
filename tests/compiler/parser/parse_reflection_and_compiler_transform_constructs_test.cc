#include <Cellerator/compiler/frontend/parser/parse_reflection_and_compiler_transform_constructs_v1.hh>

#include <cassert>
#include <string>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const std::string staged = R"(
inspect ir<semantic>;
ir_of<decomposition>(propagate, activated_fibroblast);
compiler_prelude {
  transform ir<semantic> fuse_regulatory_moments(ir<semantic> graph) {
    auto edit = ce::ir::rewrite(graph);
    return edit.commit();
  }
  pass verify_exact_cover { ce::ir::builder(graph); }
}
apply transform fuse_regulatory_moments to propagate;
pipeline insert verify_exact_cover after semantic_analysis;
pipeline replace projection_selection with verify_exact_cover;
ce::ir::builder<semantic>(propagate);
)";
    const auto parsed = parse_reflection_and_compiler_transform_constructs_v1(staged);
    assert(parsed.accepted());
    assert(parsed.constructs.size() == 7);
    assert(parsed.constructs[0].kind
           == compiler_transform_construct_kind_v1::inspect_query);
    assert(parsed.constructs[0].ir_level == "semantic");
    assert(parsed.constructs[1].target == "propagate");
    assert(parsed.constructs[2].kind
           == compiler_transform_construct_kind_v1::compiler_prelude);
    assert(parsed.constructs[3].name == "fuse_regulatory_moments");
    assert(parsed.constructs[4].insertion == pipeline_insertion_v1::after);
    assert(parsed.constructs[5].target == "projection_selection");
    assert(parsed.constructs[6].kind
           == compiler_transform_construct_kind_v1::ir_builder);

    const auto early = parse_reflection_and_compiler_transform_constructs_v1(R"(
apply transform later to propagate;
transform ir<semantic> later(ir<semantic> graph) { return graph; }
)"
    );
    assert(!early.accepted());
    assert(early.diagnostics.front().message == "transform used before it is available");

    assert(!parse_reflection_and_compiler_transform_constructs_v1(
        "inspect ir<unknown>;").accepted());
    assert(!parse_reflection_and_compiler_transform_constructs_v1(
        "pipeline insert pass_without_anchor;").accepted());
}
