#include <Cellerator/compiler/frontend/parser/parse_biological_type_constructors_and_qualifiers_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const std::string source =
        "relation_values<storage<f16>, endpoint<gene, order<module_major>>> "
        "forward mutable generation(42) persistent tag(regulatory)";
    const auto parsed = parse_biological_type_v1(source);
    assert(parsed.accepted());
    assert(parsed.type.constructor == "relation_values");
    assert(parsed.type.arguments.size() == 2);
    assert(parsed.type.arguments[0].constructor == "storage");
    assert(parsed.type.arguments[1].arguments[1].constructor == "order");
    assert(parsed.type.qualifiers.size() == 5);
    assert(render_biological_type_v1(parsed.type) == source);

    const std::string dependent =
        "typename_T::template_state<compute<T>, accumulation<typename_T::value>> ordered(axis)";
    const auto dependent_parse = parse_biological_type_v1(dependent);
    assert(dependent_parse.accepted());
    assert(render_biological_type_v1(dependent_parse.type) == dependent);

    const auto rejected = parse_biological_type_v1("state<float, axis<gene>");
    assert(!rejected.accepted());
}
