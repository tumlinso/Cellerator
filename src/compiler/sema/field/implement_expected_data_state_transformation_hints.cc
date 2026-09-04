#include <Cellerator/compiler/sema/field/implement_expected_data_state_transformation_hints_v1.hh>

#include <algorithm>
#include <utility>

namespace Cellerator::compiler::sema::field {
namespace {

[[nodiscard]] bool available(const representative_profile_state_v1& state) noexcept {
    return state.baseline || state.activated;
}

}  // namespace

expected_data_state_transformation_status_v1
implement_expected_data_state_transformation_hints_v1(
    const representative_profile_binding_v1& binding,
    const expected_data_state_transformation_hint_v1& hint,
    expected_data_state_transformation_v1* transformation) noexcept {
    if (transformation == nullptr ||
        (binding.field_identity.low == 0 && binding.field_identity.high == 0) ||
        binding.states.empty()) {
        return expected_data_state_transformation_status_v1::invalid_binding;
    }
    if (hint.operation_identity == 0) {
        return expected_data_state_transformation_status_v1::invalid_operation;
    }

    const auto input = std::find_if(
        binding.states.begin(), binding.states.end(), [&hint](const auto& state) {
            return state.state_identity == hint.input_profile_state_identity;
        });
    if (input == binding.states.end() || !available(*input)) {
        return expected_data_state_transformation_status_v1::unknown_input_state;
    }

    expected_data_state_transformation_v1 result;
    result.operation_identity = hint.operation_identity;
    result.input_profile_state_identity = input->state_identity;

    switch (hint.transformation) {
        case data_state_transformation_kind_v1::value_only:
            result.advance_components = state_component_value_v1;
            result.output_profile_state_identity = input->state_identity;
            result.output_profile_state_name = input->name;
            result.precision = profile_state_precision_v1::inferred;
            break;
        case data_state_transformation_kind_v1::support_changing:
            result.advance_components = state_component_value_v1 | state_component_support_v1;
            result.precision = profile_state_precision_v1::widened;
            result.widening_cost = 1;
            break;
        case data_state_transformation_kind_v1::topology_changing:
            result.advance_components = state_component_structure_v1 |
                state_component_value_v1 | state_component_support_v1;
            result.precision = profile_state_precision_v1::widened;
            result.widening_cost = 2;
            result.costly_widening = true;
            result.warning =
                "topology-changing transformation requires costly profile-state widening";
            break;
        case data_state_transformation_kind_v1::unknown:
            result.advance_components = state_component_structure_v1 |
                state_component_value_v1 | state_component_support_v1 |
                state_component_order_v1;
            result.precision = profile_state_precision_v1::unknown;
            result.widening_cost = 3;
            result.costly_widening = true;
            result.warning =
                "unknown transformation requires costly widening of all profile-state facts";
            break;
    }

    if (!hint.expected_post_state.empty()) {
        const auto expected = std::find_if(
            binding.states.begin(), binding.states.end(), [&hint](const auto& state) {
                return state.name == hint.expected_post_state;
            });
        if (expected == binding.states.end() || !available(*expected)) {
            return expected_data_state_transformation_status_v1::unavailable_expected_state;
        }
        result.output_profile_state_identity = expected->state_identity;
        result.output_profile_state_name = expected->name;
        result.precision = profile_state_precision_v1::exact;
        result.widening_cost = 0;
        result.explicitly_selected = true;
        result.costly_widening = false;
        result.warning.clear();
    }

    *transformation = std::move(result);
    return expected_data_state_transformation_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
