#include <Cellerator/compiler/sema/field/resolve_and_implement_nested_field_semantics_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

field::execution_field_semantics_v1 make_field(
    const char* name, std::uint64_t begin, std::uint64_t end) {
    field::execution_field_definition_v1 definition;
    definition.stable_source_name = "nested.cell";
    definition.explicit_field_name = name;
    definition.source = {{3, begin}, {3, end}};
    field::execution_field_semantics_v1 result;
    if (field::define_execution_field_semantic_ownership_v1(definition, &result) !=
        field::execution_field_definition_status_v1::success) {
        return {};
    }
    return result;
}

int main() {
    const auto parent = make_field("outer", 10, 200);
    const auto child = make_field("inner", 40, 100);
    field::nested_field_request_v1 request;
    request.parent = &parent;
    request.child = child;
    request.inherited_facts = {{"reuse", "100"}, {"provider", "auto"}};
    request.inherited_constraints = {{"determinism", "stable", 2}};
    request.local_fact_overlays = {{"provider", "nvidia"}};
    request.local_constraint_overlays = {{"determinism", "bitwise", 3}};

    field::resolved_nested_field_v1 resolved;
    if (field::resolve_and_implement_nested_field_semantics_v1(request, &resolved) !=
            field::nested_field_status_v1::success ||
        !resolved.separately_nameable || !resolved.planning_subproblem ||
        !resolved.optimization_boundary || !resolved.movement_barrier ||
        resolved.effective_facts.size() != 2 ||
        resolved.effective_facts[1].value != "nvidia" ||
        resolved.effective_constraints[0].strength != 3) {
        std::cerr << "nested field did not inherit and overlay policy\n";
        return 1;
    }

    request.explicitly_inline = true;
    if (field::resolve_and_implement_nested_field_semantics_v1(request, &resolved) !=
            field::nested_field_status_v1::success ||
        resolved.planning_subproblem || resolved.optimization_boundary ||
        resolved.movement_barrier) {
        std::cerr << "explicit field inlining did not remove the boundary\n";
        return 1;
    }

    request.local_constraint_overlays[0].strength = 1;
    if (field::resolve_and_implement_nested_field_semantics_v1(request, &resolved) !=
        field::nested_field_status_v1::weakened_inherited_constraint) {
        std::cerr << "nested field weakened an inherited hard constraint\n";
        return 1;
    }

    request.local_constraint_overlays[0].strength = 3;
    request.child = make_field("outside", 190, 220);
    if (field::resolve_and_implement_nested_field_semantics_v1(request, &resolved) !=
        field::nested_field_status_v1::child_outside_parent) {
        std::cerr << "cross-boundary nested field was accepted\n";
        return 1;
    }
    return 0;
}
