#include <Cellerator/compiler/frontend/parser/parse_named_execution_fields_and_references_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const auto parsed = parse_named_execution_fields_v1(R"cpp(
export field void propagate(state<float, gene> input);
field void helper(state<float, gene> input) <[ propagate(input); ]>
field void propagate(state<float, gene> input) <[ propagate(input); helper(input); ]>
import field void remote(axis<gene> genes);
)cpp");
    assert(parsed.accepted());
    assert(parsed.fields.size() == 4);
    assert(parsed.fields[0].forward_declaration);
    assert(parsed.fields[0].linkage == field_linkage_intent_v1::export_field);
    assert(parsed.fields[2].references.size() == 2);
    assert(parsed.fields[2].references[0] == "helper");
    assert(parsed.fields[2].references[1] == "propagate");
    assert(parsed.fields[3].linkage == field_linkage_intent_v1::import_field);

    const auto duplicate = parse_named_execution_fields_v1(
        "field void f() <[ ]> field void f() <[ ]>");
    assert(!duplicate.accepted());
    assert(duplicate.diagnostics.front().message == "duplicate named field definition");

    const auto forwards = parse_named_execution_fields_v1(
        "field void f(); field void f(); field void f() <[ f(); ]>");
    assert(forwards.accepted());
    assert(forwards.fields.back().references.front() == "f");
}
