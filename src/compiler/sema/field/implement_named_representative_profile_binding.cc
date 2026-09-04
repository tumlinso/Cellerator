#include <Cellerator/compiler/sema/field/implement_named_representative_profile_binding_v1.hh>

#include <algorithm>
#include <utility>

namespace Cellerator::compiler::sema::field {

profile_binding_status_v1 implement_named_representative_profile_binding_v1(
    const execution_field_semantics_v1& field,
    const std::vector<representative_profile_state_v1>& compile_supplied_states,
    const std::vector<representative_profile_alias_v1>& aliases,
    const std::vector<operation_profile_selection_v1>& selections,
    representative_profile_binding_v1* binding) noexcept {
    if (binding == nullptr || (field.identity.low == 0 && field.identity.high == 0)) {
        return profile_binding_status_v1::invalid_field;
    }
    std::size_t baseline_count = 0;
    for (std::size_t index = 0; index < compile_supplied_states.size(); ++index) {
        const auto& state = compile_supplied_states[index];
        if (state.name.empty() || state.state_identity == 0 ||
            (state.content_digest_low == 0 && state.content_digest_high == 0)) {
            return profile_binding_status_v1::invalid_state;
        }
        baseline_count += state.baseline ? 1 : 0;
        if (std::find_if(compile_supplied_states.begin(),
                         compile_supplied_states.begin() + index,
                         [&state](const auto& prior) { return prior.name == state.name; }) !=
            compile_supplied_states.begin() + index) {
            return profile_binding_status_v1::duplicate_state;
        }
    }
    if (baseline_count != 1) return profile_binding_status_v1::missing_baseline;

    for (std::size_t index = 0; index < aliases.size(); ++index) {
        const auto& alias = aliases[index];
        if (alias.alias.empty() || alias.state_name.empty() ||
            std::find_if(aliases.begin(), aliases.begin() + index,
                         [&alias](const auto& prior) { return prior.alias == alias.alias; }) !=
                aliases.begin() + index) {
            return profile_binding_status_v1::duplicate_alias;
        }
        if (std::none_of(compile_supplied_states.begin(), compile_supplied_states.end(),
                         [&alias](const auto& state) { return state.name == alias.state_name; })) {
            return profile_binding_status_v1::unresolved_alias;
        }
    }

    representative_profile_binding_v1 result;
    result.field_identity = field.identity;
    result.states = compile_supplied_states;
    result.aliases = aliases;
    for (const auto& selection : selections) {
        if (selection.operation_identity == 0) return profile_binding_status_v1::invalid_operation;
        auto selected_name = selection.state_or_alias;
        const auto alias = std::find_if(aliases.begin(), aliases.end(),
            [&selection](const auto& candidate) {
                return candidate.alias == selection.state_or_alias;
            });
        if (alias != aliases.end()) selected_name = alias->state_name;
        const auto state = std::find_if(
            compile_supplied_states.begin(), compile_supplied_states.end(),
            [&selected_name](const auto& candidate) { return candidate.name == selected_name; });
        if (state == compile_supplied_states.end() || (!state->baseline && !state->activated)) {
            return profile_binding_status_v1::unavailable_state;
        }
        result.operations.push_back(
            {selection.operation_identity, state->state_identity, state->name, state->activated});
    }
    *binding = std::move(result);
    return profile_binding_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
