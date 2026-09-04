#include <Cellerator/compiler/sema/field/implement_missing_profile_failure_policy_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    field::missing_profile_policy_result_v1 result;
    if (field::implement_missing_profile_failure_policy_v1(
            {field::compilation_activation_kind_v1::pure_cpp_fallthrough, false, false},
            &result) != field::missing_profile_policy_status_v1::success ||
        !result.compilation_allowed) {
        std::cerr << "pure C++ fallthrough incorrectly required a profile\n";
        return 1;
    }
    if (field::implement_missing_profile_failure_policy_v1(
            {field::compilation_activation_kind_v1::ceir_structural_only, false, false},
            &result) != field::missing_profile_policy_status_v1::success ||
        !result.compilation_allowed) {
        std::cerr << "CEIR-only structural work incorrectly required a profile\n";
        return 1;
    }
    if (field::implement_missing_profile_failure_policy_v1(
            {field::compilation_activation_kind_v1::biological_compilation, false, false},
            &result) != field::missing_profile_policy_status_v1::representative_profile_required ||
        result.compilation_allowed || result.diagnostic.empty()) {
        std::cerr << "activated biological compilation accepted a missing profile\n";
        return 1;
    }
    if (field::implement_missing_profile_failure_policy_v1(
            {field::compilation_activation_kind_v1::biological_compilation, false, true},
            &result) != field::missing_profile_policy_status_v1::success ||
        !result.compilation_allowed || !result.uses_generic_reference_profile) {
        std::cerr << "explicit generic reference profile was not accepted\n";
        return 1;
    }
    if (field::implement_missing_profile_failure_policy_v1(
            {field::compilation_activation_kind_v1::biological_compilation, true, false},
            &result) != field::missing_profile_policy_status_v1::success ||
        !result.compilation_allowed || result.uses_generic_reference_profile) {
        std::cerr << "bound representative profile was not accepted\n";
        return 1;
    }
    return 0;
}
