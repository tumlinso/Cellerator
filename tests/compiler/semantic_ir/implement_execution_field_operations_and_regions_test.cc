#include <Cellerator/compiler/ir/semantic/implement_execution_field_operations_and_regions_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

int main() {
    semantic_scope_diagnostic_v1 scope_diagnostic;
    semantic_scope_module_definition_v1 definition;
    definition.scopes = {
        {0, invalid_semantic_scope_id_v1, semantic_scope_kind_v1::program, 1, "program"},
        {1, 0, semantic_scope_kind_v1::module, 2, "module"},
        {2, 1, semantic_scope_kind_v1::translation_unit, 3, "field.cc"},
        {3, 2, semantic_scope_kind_v1::function, 4, "run"},
        {4, 3, semantic_scope_kind_v1::named_field, 5, "outer"},
        {5, 4, semantic_scope_kind_v1::anonymous_field, 6, "inner"},
        {6, 2, semantic_scope_kind_v1::named_field, 7, "opaque"},
    };
    auto scopes = freeze_semantic_ir_module_and_symbol_scopes_v1(
        std::move(definition), &scope_diagnostic);
    assert(scopes);

    execution_field_region_ir_v1 outer;
    outer.identity = 100;
    outer.scope = 4;
    outer.kind = execution_field_kind_ir_v1::named;
    outer.captures = {{20, false}};
    outer.results = {{21, true}};
    outer.profile_environment = {30, 31};
    outer.facts = {{"structure_persists", "experiment"}};
    outer.constraints = {{"deterministic", "required", true}};
    outer.operations = {200, 201};
    outer.observable_effects = field_effect_reads_ir_v1;

    execution_field_region_ir_v1 inner;
    inner.identity = 101;
    inner.scope = 5;
    inner.parent_field_identity = 100;
    inner.kind = execution_field_kind_ir_v1::anonymous_field;
    inner.profile_environment = {30, 31};
    inner.facts = {{"minimum_transient_memory", "preferred"}};
    inner.operations = {202};
    inner.observable_effects = field_effect_writes_ir_v1;

    execution_field_region_ir_v1 opaque;
    opaque.identity = 102;
    opaque.scope = 6;
    opaque.kind = execution_field_kind_ir_v1::named;
    opaque.boundary = execution_field_boundary_ir_v1::explicit_boundary;
    opaque.profile_environment = {40, 41};
    opaque.operations = {203};

    execution_field_ir_validation_code_v1 status;
    auto fields = freeze_execution_field_operations_and_regions_v1(
        {inner, opaque, outer}, *scopes, &status);
    assert(fields && status == execution_field_ir_validation_code_v1::success);
    const auto environment = fields->effective_environment(101);
    assert(environment && environment->facts.size() == 2);
    assert((environment->observable_effects & field_effect_reads_ir_v1) != 0);
    assert((environment->observable_effects & field_effect_writes_ir_v1) != 0);
    assert(fields->visibility(100, 101, true) ==
           execution_field_visibility_ir_v1::inline_semantics);
    assert(fields->visibility(100, 102, true) ==
           execution_field_visibility_ir_v1::call_boundary);

    inner.parent_field_identity = 102;
    assert(!freeze_execution_field_operations_and_regions_v1(
        {outer, inner, opaque}, *scopes, &status));
    assert(status == execution_field_ir_validation_code_v1::invalid_parent);

    std::cout << "fields=3 nested_inline=visible explicit_boundary=opaque\n";
}
