#include <Cellerator/compiler/frontend/cxx/model_opaque_native_calls_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::frontend::cxx {

opaque_native_call_status_v1 model_opaque_native_calls_v1(
    std::uint32_t schema_version,
    const std::vector<overload_semantic_candidate_v1>& calls,
    const std::vector<native_call_contract_v1>& contracts,
    std::vector<opaque_native_call_v1>* models) noexcept {
    if (models == nullptr || calls.empty()) {
        return opaque_native_call_status_v1::invalid_candidate;
    }
    models->clear();
    if (schema_version != opaque_native_call_schema_version_v1) {
        return opaque_native_call_status_v1::schema_mismatch;
    }
    for (const auto& call : calls) {
        if (call.selected_declaration == nullptr || call.qualified_name.empty()) {
            return opaque_native_call_status_v1::invalid_candidate;
        }
        opaque_native_call_v1 model;
        model.selected_declaration = call.selected_declaration;
        model.qualified_name = call.qualified_name;
        const auto contract = std::find_if(
            contracts.begin(), contracts.end(),
            [&call](const native_call_contract_v1& item) {
                return item.qualified_name == call.qualified_name;
            });
        if (call.cellerator_aware) {
            model.contract_applied = true;
            model.effects = native_effect_read_v1;
        } else if (contract != contracts.end()) {
            model.contract_applied = true;
            model.effects = contract->effects;
        } else {
            model.effects = native_effect_read_v1 | native_effect_write_v1 |
                            native_effect_escape_v1 | native_effect_synchronize_v1;
            model.semantic_barrier = true;
            model.diagnostic = "uncontracted native call '" + call.qualified_name +
                "' is a conservative read/write/escape/synchronization barrier";
        }
        models->push_back(std::move(model));
    }
    return opaque_native_call_status_v1::success;
}

bool may_reorder_across_native_call_v1(
    const opaque_native_call_v1& call,
    std::uint32_t moving_effects) noexcept {
    if (call.semantic_barrier || (call.effects & native_effect_synchronize_v1) != 0 ||
        (call.effects & native_effect_escape_v1) != 0) {
        return false;
    }
    const bool either_writes =
        ((call.effects | moving_effects) & native_effect_write_v1) != 0;
    const bool either_accesses =
        ((call.effects | moving_effects) & (native_effect_read_v1 | native_effect_write_v1)) != 0;
    return !(either_writes && either_accesses);
}

}  // namespace Cellerator::compiler::frontend::cxx
