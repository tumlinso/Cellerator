#include <Cellerator/compiler/sema/field/implement_expected_data_state_transformation_hints_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    field::representative_profile_binding_v1 binding;
    binding.field_identity = {17, 23};
    binding.states = {
        {"baseline", 101, 1, 2, true, false},
        {"activated", 102, 3, 4, false, true},
    };

    field::expected_data_state_transformation_v1 result;
    if (field::implement_expected_data_state_transformation_hints_v1(
            binding, {1, 101, "", field::data_state_transformation_kind_v1::value_only},
            &result) != field::expected_data_state_transformation_status_v1::success ||
        result.output_profile_state_identity != 101 ||
        result.advance_components != field::state_component_value_v1 ||
        result.precision != field::profile_state_precision_v1::inferred ||
        result.costly_widening) {
        std::cerr << "value-only profile transfer was not inferred\n";
        return 1;
    }

    if (field::implement_expected_data_state_transformation_hints_v1(
            binding, {2, 101, "activated",
                      field::data_state_transformation_kind_v1::support_changing},
            &result) != field::expected_data_state_transformation_status_v1::success ||
        result.output_profile_state_identity != 102 || !result.explicitly_selected ||
        result.advance_components !=
            (field::state_component_value_v1 | field::state_component_support_v1) ||
        result.precision != field::profile_state_precision_v1::exact) {
        std::cerr << "support-changing expected state was not selected\n";
        return 1;
    }

    if (field::implement_expected_data_state_transformation_hints_v1(
            binding, {3, 102, "", field::data_state_transformation_kind_v1::topology_changing},
            &result) != field::expected_data_state_transformation_status_v1::success ||
        !result.costly_widening || result.warning.empty() || result.widening_cost != 2 ||
        (result.advance_components & field::state_component_structure_v1) == 0) {
        std::cerr << "topology-changing transfer did not warn about widening\n";
        return 1;
    }

    if (field::implement_expected_data_state_transformation_hints_v1(
            binding, {4, 102, "", field::data_state_transformation_kind_v1::unknown},
            &result) != field::expected_data_state_transformation_status_v1::success ||
        result.precision != field::profile_state_precision_v1::unknown ||
        result.widening_cost != 3 || result.output_profile_state_identity != 0 ||
        result.warning.empty()) {
        std::cerr << "unknown transformation did not conservatively widen\n";
        return 1;
    }

    return 0;
}
