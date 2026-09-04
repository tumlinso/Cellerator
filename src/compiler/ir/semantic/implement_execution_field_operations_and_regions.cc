#include <Cellerator/compiler/ir/semantic/implement_execution_field_operations_and_regions_v1.hh>

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace Cellerator::compiler::ir::semantic {

const execution_field_region_ir_v1* frozen_execution_field_regions_v1::field(
    std::uint64_t identity) const noexcept {
    const auto found = std::lower_bound(fields_.begin(), fields_.end(), identity,
        [](const execution_field_region_ir_v1& item, std::uint64_t key) {
            return item.identity < key;
        });
    return found != fields_.end() && found->identity == identity ? &*found : nullptr;
}

std::optional<effective_field_environment_ir_v1>
frozen_execution_field_regions_v1::effective_environment(std::uint64_t identity) const {
    const auto* current = field(identity);
    if (current == nullptr) return std::nullopt;
    std::vector<const execution_field_region_ir_v1*> lineage;
    while (current != nullptr) {
        lineage.push_back(current);
        current = current->parent_field_identity == 0
            ? nullptr : field(current->parent_field_identity);
    }
    std::reverse(lineage.begin(), lineage.end());
    effective_field_environment_ir_v1 result;
    for (const auto* item : lineage) {
        if (item->profile_environment.valid()) result.profile_environment = item->profile_environment;
        result.facts.insert(result.facts.end(), item->facts.begin(), item->facts.end());
        result.constraints.insert(result.constraints.end(), item->constraints.begin(),
                                  item->constraints.end());
        result.observable_effects |= item->observable_effects;
    }
    return result;
}

execution_field_visibility_ir_v1 frozen_execution_field_regions_v1::visibility(
    std::uint64_t caller_identity,
    std::uint64_t callee_identity,
    bool request_inline_semantics) const noexcept {
    const auto* caller = field(caller_identity);
    const auto* callee = field(callee_identity);
    if (caller == nullptr || callee == nullptr || caller_identity == callee_identity)
        return execution_field_visibility_ir_v1::unavailable;
    if (callee->boundary == execution_field_boundary_ir_v1::explicit_boundary ||
        !callee->semantic_body_available)
        return execution_field_visibility_ir_v1::call_boundary;
    const bool lexical_child = callee->parent_field_identity == caller_identity;
    const bool callable_named = callee->kind == execution_field_kind_ir_v1::named;
    if ((lexical_child || callable_named) && request_inline_semantics)
        return execution_field_visibility_ir_v1::inline_semantics;
    return callable_named || lexical_child
        ? execution_field_visibility_ir_v1::call_boundary
        : execution_field_visibility_ir_v1::unavailable;
}

std::optional<frozen_execution_field_regions_v1>
freeze_execution_field_operations_and_regions_v1(
    std::vector<execution_field_region_ir_v1> fields,
    const frozen_semantic_scope_module_v1& scopes,
    execution_field_ir_validation_code_v1* status) noexcept {
    auto fail = [&](execution_field_ir_validation_code_v1 code)
        -> std::optional<frozen_execution_field_regions_v1> {
        if (status != nullptr) *status = code;
        return std::nullopt;
    };
    std::sort(fields.begin(), fields.end(), [](const auto& left, const auto& right) {
        return left.identity < right.identity;
    });
    for (std::size_t index = 0; index < fields.size(); ++index) {
        const auto& item = fields[index];
        if (item.identity == 0 || (index != 0 && fields[index - 1].identity == item.identity))
            return fail(execution_field_ir_validation_code_v1::invalid_identity);
        const auto* scope = scopes.scope(item.scope);
        if (scope == nullptr) return fail(execution_field_ir_validation_code_v1::invalid_scope);
        const bool scope_matches =
            (item.kind == execution_field_kind_ir_v1::named &&
             scope->kind == semantic_scope_kind_v1::named_field) ||
            (item.kind == execution_field_kind_ir_v1::anonymous_field &&
             scope->kind == semantic_scope_kind_v1::anonymous_field);
        if (!scope_matches) return fail(execution_field_ir_validation_code_v1::invalid_kind);
        if (item.boundary != execution_field_boundary_ir_v1::transparent &&
            item.boundary != execution_field_boundary_ir_v1::explicit_boundary)
            return fail(execution_field_ir_validation_code_v1::invalid_boundary);
        if (!item.profile_environment.valid())
            return fail(execution_field_ir_validation_code_v1::invalid_profile);
        if (item.parent_field_identity != 0) {
            const auto parent = std::find_if(fields.begin(), fields.end(), [&item](const auto& other) {
                return other.identity == item.parent_field_identity;
            });
            if (parent == fields.end() || scope->parent != parent->scope)
                return fail(execution_field_ir_validation_code_v1::invalid_parent);
        }
        std::unordered_set<std::uint64_t> bindings;
        for (const auto& binding : item.captures) {
            if (binding.symbol_identity == 0)
                return fail(execution_field_ir_validation_code_v1::invalid_binding);
            if (!bindings.insert(binding.symbol_identity).second)
                return fail(execution_field_ir_validation_code_v1::duplicate_binding);
        }
        for (const auto& binding : item.results) {
            if (binding.symbol_identity == 0)
                return fail(execution_field_ir_validation_code_v1::invalid_binding);
            if (!bindings.insert(binding.symbol_identity).second)
                return fail(execution_field_ir_validation_code_v1::duplicate_binding);
        }
        for (const auto& fact : item.facts)
            if (fact.name.empty() || fact.value.empty())
                return fail(execution_field_ir_validation_code_v1::invalid_directive);
        for (const auto& constraint : item.constraints)
            if (constraint.name.empty() || constraint.value.empty())
                return fail(execution_field_ir_validation_code_v1::invalid_directive);
        auto operations = item.operations;
        std::sort(operations.begin(), operations.end());
        if (!operations.empty() && operations.front() == 0)
            return fail(execution_field_ir_validation_code_v1::duplicate_operation);
        if (std::adjacent_find(operations.begin(), operations.end()) != operations.end())
            return fail(execution_field_ir_validation_code_v1::duplicate_operation);
    }
    frozen_execution_field_regions_v1 result;
    result.fields_ = std::move(fields);
    if (status != nullptr) *status = execution_field_ir_validation_code_v1::success;
    return result;
}

}  // namespace Cellerator::compiler::ir::semantic
