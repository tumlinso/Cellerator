#include <Cellerator/compiler/ir/semantic/freeze_semantic_ir_module_and_symbol_scopes_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

namespace {

semantic_scope_module_definition_v1 nested_program() {
    semantic_scope_module_definition_v1 definition;
    definition.scopes = {
        {0, invalid_semantic_scope_id_v1, semantic_scope_kind_v1::program, 100, "program"},
        {1, 0, semantic_scope_kind_v1::module, 101, "biology"},
        {2, 1, semantic_scope_kind_v1::translation_unit, 102, "producer.cc"},
        {3, 2, semantic_scope_kind_v1::function, 103, "step"},
        {4, 3, semantic_scope_kind_v1::named_field, 104, "propagate"},
        {5, 4, semantic_scope_kind_v1::anonymous_field, 105, "anonymous@42"},
        {6, 1, semantic_scope_kind_v1::translation_unit, 106, "consumer.cc"},
        {7, 6, semantic_scope_kind_v1::function, 107, "run"},
    };
    definition.symbols = {
        {200, semantic_symbol_kind_v1::domain, 1, "gene"},
        {201, semantic_symbol_kind_v1::relation, 4, "regulation"},
        {202, semantic_symbol_kind_v1::state, 5, "response"},
        {203, semantic_symbol_kind_v1::function, 3, "helper"},
    };
    definition.export_authorizations = {{201, 2, 6}};
    definition.imports = {{201, 7}};
    return definition;
}

}  // namespace

int main() {
    semantic_scope_diagnostic_v1 diagnostic;
    auto frozen = freeze_semantic_ir_module_and_symbol_scopes_v1(nested_program(), &diagnostic);
    assert(frozen && diagnostic);
    assert(frozen->scopes().size() == 8);
    assert(frozen->owning_translation_unit(5) == 2);
    assert(frozen->owning_translation_unit(7) == 6);
    assert(frozen->resolve_local(5, "response")->stable_identity == 202);
    assert(frozen->resolve_local(5, "regulation")->stable_identity == 201);
    assert(frozen->resolve_imported(7, "regulation")->stable_identity == 201);
    assert(frozen->resolve_imported(5, "regulation") == nullptr);

    auto duplicate_scope = nested_program();
    duplicate_scope.scopes.push_back(
        {8, 4, semantic_scope_kind_v1::anonymous_field, 108, "anonymous@42"});
    assert(!freeze_semantic_ir_module_and_symbol_scopes_v1(
        std::move(duplicate_scope), &diagnostic));
    assert(diagnostic.code == semantic_scope_diagnostic_code_v1::duplicate_scope);
    assert(diagnostic.scope == 8);

    auto duplicate_symbol = nested_program();
    duplicate_symbol.symbols.push_back(
        {204, semantic_symbol_kind_v1::state, 5, "response"});
    assert(!freeze_semantic_ir_module_and_symbol_scopes_v1(
        std::move(duplicate_symbol), &diagnostic));
    assert(diagnostic.code == semantic_scope_diagnostic_code_v1::duplicate_symbol);

    auto unauthorized = nested_program();
    unauthorized.export_authorizations.clear();
    assert(!freeze_semantic_ir_module_and_symbol_scopes_v1(
        std::move(unauthorized), &diagnostic));
    assert(diagnostic.code == semantic_scope_diagnostic_code_v1::unauthorized_import);
    assert(diagnostic.symbol_identity == 201);

    auto illegal_nesting = nested_program();
    illegal_nesting.scopes[3].parent = 1;
    assert(!freeze_semantic_ir_module_and_symbol_scopes_v1(
        std::move(illegal_nesting), &diagnostic));
    assert(diagnostic.code == semantic_scope_diagnostic_code_v1::invalid_scope_nesting);

    std::cout << "semantic_scopes=8 authorized_cross_tu_imports=1\n";
}
