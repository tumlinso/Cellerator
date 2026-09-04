#include <Cellerator/compiler/sema/field/implement_field_level_reflection_identity_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

field::execution_field_semantics_v1 compile_field(std::uint64_t end_offset) {
    field::execution_field_definition_v1 definition;
    definition.stable_source_name = "src/biology/propagate.cc";
    definition.explicit_field_name = "propagate";
    definition.source = {{41, 100}, {41, end_offset}};
    field::execution_field_semantics_v1 semantics;
    if (field::define_execution_field_semantic_ownership_v1(definition, &semantics) !=
        field::execution_field_definition_status_v1::success) {
        return {};
    }
    return semantics;
}

int main() {
    const auto first_compile = compile_field(300);
    const auto unchanged_recompile = compile_field(300);
    const auto changed_field = compile_field(301);
    field::field_reflection_identity_v1 first;
    field::field_reflection_identity_v1 second;
    field::field_reflection_identity_v1 changed;
    if (field::implement_field_level_reflection_identity_v1(first_compile, &first) !=
            field::field_reflection_identity_status_v1::success ||
        field::implement_field_level_reflection_identity_v1(unchanged_recompile, &second) !=
            field::field_reflection_identity_status_v1::success ||
        field::implement_field_level_reflection_identity_v1(changed_field, &changed) !=
            field::field_reflection_identity_status_v1::success ||
        !(first == second) || first.field_identity == changed.field_identity ||
        first.stable_export_name.empty() ||
        first.stable_export_name.find("cellerator.field.v1.") != 0) {
        std::cerr << "field reflection identity was not stable and source-derived\n";
        return 1;
    }

    // The reflection record contains only schema, compile-time identity, and a
    // stable export spelling; it owns no runtime metadata or addresses.
    return 0;
}
