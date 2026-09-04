#include <Cellerator/compiler/sema/field/resolve_and_implement_nested_field_semantics_v1.hh>

#include <algorithm>
#include <utility>

namespace Cellerator::compiler::sema::field {
namespace {

template<class Entry>
bool has_duplicate_keys(const std::vector<Entry>& entries) noexcept {
    for (std::size_t index = 0; index < entries.size(); ++index) {
        if (entries[index].key.empty() ||
            std::find_if(entries.begin(), entries.begin() + index,
                         [&entries, index](const Entry& prior) {
                             return prior.key == entries[index].key;
                         }) != entries.begin() + index) {
            return true;
        }
    }
    return false;
}

template<class Entry>
void overlay(std::vector<Entry>& effective,
             const std::vector<Entry>& local) {
    for (const auto& entry : local) {
        const auto found = std::find_if(
            effective.begin(), effective.end(), [&entry](const Entry& inherited) {
                return inherited.key == entry.key;
            });
        if (found == effective.end()) {
            effective.push_back(entry);
        } else {
            *found = entry;
        }
    }
}

bool valid_identity(execution_field_identity_v1 identity) noexcept {
    return identity.low != 0 || identity.high != 0;
}

}  // namespace

nested_field_status_v1 resolve_and_implement_nested_field_semantics_v1(
    const nested_field_request_v1& request,
    resolved_nested_field_v1* resolved) noexcept {
    if (request.parent == nullptr || resolved == nullptr) {
        return nested_field_status_v1::missing_parent;
    }
    if (!valid_identity(request.parent->identity) || !request.parent->source.valid()) {
        return nested_field_status_v1::invalid_parent;
    }
    if (!valid_identity(request.child.identity) || !request.child.source.valid()) {
        return nested_field_status_v1::invalid_child;
    }
    if (!execution_field_owns_operation_v1(*request.parent, request.child.source)) {
        return nested_field_status_v1::child_outside_parent;
    }
    if (request.child.explicit_field_name.empty()) {
        return nested_field_status_v1::child_not_separately_named;
    }
    if (has_duplicate_keys(request.inherited_facts) ||
        has_duplicate_keys(request.inherited_constraints) ||
        has_duplicate_keys(request.local_fact_overlays) ||
        has_duplicate_keys(request.local_constraint_overlays)) {
        return nested_field_status_v1::duplicate_overlay;
    }
    for (const auto& local : request.local_constraint_overlays) {
        const auto inherited = std::find_if(
            request.inherited_constraints.begin(), request.inherited_constraints.end(),
            [&local](const hard_constraint_v1& candidate) {
                return candidate.key == local.key;
            });
        if (inherited != request.inherited_constraints.end() &&
            local.strength < inherited->strength) {
            return nested_field_status_v1::weakened_inherited_constraint;
        }
    }

    resolved_nested_field_v1 result;
    result.parent_identity = request.parent->identity;
    result.child_identity = request.child.identity;
    result.effective_facts = request.inherited_facts;
    result.effective_constraints = request.inherited_constraints;
    overlay(result.effective_facts, request.local_fact_overlays);
    overlay(result.effective_constraints, request.local_constraint_overlays);
    result.separately_nameable = true;
    result.planning_subproblem = !request.explicitly_inline;
    result.optimization_boundary = !request.explicitly_inline;
    result.movement_barrier = !request.explicitly_inline;
    *resolved = std::move(result);
    return nested_field_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
