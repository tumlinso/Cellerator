#include <Cellerator/compiler/frontend/parser/parse_relation_application_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    const auto parsed = parse_relation_applications_v1(
        "response += overloaded(source, 3) -[ /*typed*/ regulation on T::genes where active ]-> "
        "modules -[transpose(module_to_gene)]-> target_genes;");
    assert(parsed.accepted());
    assert(parsed.applications.size() == 2);
    assert(parsed.applications[0].result_expression == "response");
    assert(parsed.applications[0].source_expression == "overloaded(source, 3)");
    assert(parsed.applications[0].selector.relation_expression == "regulation");
    assert(parsed.applications[0].selector.source_axis_expression == "T::genes");
    assert(parsed.applications[0].selector.support_expression == "active");
    assert(parsed.applications[0].update == relation_update_v1::accumulate);
    assert(parsed.applications[1].selector.orientation == relation_orientation_v1::transpose);
    assert(parsed.applications[1].source_expression == "modules");
    assert(parsed.applications[1].destination_axis_expression == "target_genes");

    const auto whitespace = parse_relation_applications_v1(
        "f(x -[ relation<T> /* c */ ]-> dependent_axis) + 1");
    assert(whitespace.accepted());
    assert(whitespace.applications.size() == 1);
    assert(whitespace.applications[0].update == relation_update_v1::expression);
    assert(!parse_relation_applications_v1("source -[relation destination").accepted());
}
