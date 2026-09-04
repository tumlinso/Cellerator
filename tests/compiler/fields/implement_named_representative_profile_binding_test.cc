#include <Cellerator/compiler/sema/field/implement_named_representative_profile_binding_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    field::execution_field_definition_v1 definition;
    definition.stable_source_name = "profiles.cell";
    definition.explicit_field_name = "profiled_step";
    definition.source = {{2, 10}, {2, 80}};
    field::execution_field_semantics_v1 semantics;
    if (field::define_execution_field_semantic_ownership_v1(definition, &semantics) !=
        field::execution_field_definition_status_v1::success) return 1;

    const std::vector<field::representative_profile_state_v1> profiles{
        {"baseline", 11, 1, 2, true, false},
        {"stimulated", 12, 3, 4, false, true},
        {"withheld", 13, 5, 6, false, false},
    };
    field::representative_profile_binding_v1 binding;
    if (field::implement_named_representative_profile_binding_v1(
            semantics, profiles, {{"active", "stimulated"}},
            {{20, "baseline"}, {21, "active"}}, &binding) !=
            field::profile_binding_status_v1::success ||
        binding.operations.size() != 2 || binding.operations[0].state_identity != 11 ||
        binding.operations[1].state_identity != 12 || !binding.operations[1].activated) {
        std::cerr << "baseline and activated profile alternatives did not bind\n";
        return 1;
    }
    if (field::implement_named_representative_profile_binding_v1(
            semantics, profiles, {}, {{22, "withheld"}}, &binding) !=
        field::profile_binding_status_v1::unavailable_state) {
        std::cerr << "inactive profile state was selected\n";
        return 1;
    }
    // The semantic record intentionally contains identities and digests only;
    // profile data paths are compile-driver inputs, not language semantics.
    return 0;
}
