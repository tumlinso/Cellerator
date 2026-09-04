#include <Cellerator/compiler/frontend/parser/parse_inline_ceir_blocks_v1.hh>

#include <cassert>
#include <string>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const std::string source = R"(
ceir<semantic> captures(%input: state<gene>, %r: relation<gene,gene>)
    results(%out: state<gene>) validation(verified) transition(semantic, planning) {
  %out = cellerator.relation_apply %input, %r {
    ceir<planning> captures(%out) results(%candidate) validation(checked) {
      offer candidate @tiled
    }
  }
})";
    const auto parsed = parse_inline_ceir_blocks_v1(source);
    assert(parsed.accepted());
    assert(parsed.blocks.size() == 1);
    const auto &block = parsed.blocks.front();
    assert(block.level == inline_ceir_level_v1::semantic);
    assert(block.validation == inline_ceir_validation_v1::verified);
    assert(block.captures.size() == 2);
    assert(block.results.size() == 1);
    assert(block.transition_from == "semantic");
    assert(block.transition_to == "planning");
    assert(block.nested.size() == 1);
    assert(render_inline_ceir_block_v1(block) == source.substr(1));

    assert(!parse_inline_ceir_blocks_v1("ceir<physical> {}").accepted());
    assert(!parse_inline_ceir_blocks_v1("ceir<realization> validation(magic) {}").accepted());
    assert(!parse_inline_ceir_blocks_v1("ceir<planning> {").accepted());
}
