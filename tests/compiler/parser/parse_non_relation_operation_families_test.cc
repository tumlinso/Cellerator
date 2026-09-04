#include <Cellerator/compiler/frontend/parser/parse_non_relation_operation_families_v1.hh>

#include <algorithm>
#include <cassert>
#include <string>

using namespace Cellerator::compiler::frontend::parser;

int main() {
    assert(operation_family_table_v1().size() == 14);
    for (const auto &definition : operation_family_table_v1()) {
        const std::string source = definition.family == operation_family_v1::transpose
            ? "transpose(regulation)"
            : "ce::" + std::string(definition.spelling) + "(input, nested<T, U>(x), policy)";
        const auto parsed = parse_operation_families_v1(source);
        assert(parsed.accepted());
        const auto found = std::find_if(parsed.operations.begin(), parsed.operations.end(),
            [&](const operation_syntax_v1 &operation) {
                return operation.family == definition.family;
            });
        assert(found != parsed.operations.end());
        assert(!found->arguments.empty());
        if (definition.form == operation_parse_form_v1::library_lowering)
            assert(found->arguments.size() == 3);
    }

    const auto current_core = parse_operation_families_v1(
        "ce::contract_on(support, x, y, dot); ce::relation_moments(x, r, genes); "
        "ce::sparse_update(x, ids, delta, add);");
    assert(current_core.accepted());
    assert(current_core.operations.size() == 3);
    assert(!parse_operation_families_v1("ce::edge_gate(regulation").accepted());
}
