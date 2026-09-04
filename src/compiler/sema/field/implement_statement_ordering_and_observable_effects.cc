#include <Cellerator/compiler/sema/field/implement_statement_ordering_and_observable_effects_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::sema::field {
namespace {

bool intersects(const std::vector<semantic_value_id_v1>& lhs,
                const std::vector<semantic_value_id_v1>& rhs) noexcept {
    return std::any_of(lhs.begin(), lhs.end(), [&rhs](semantic_value_id_v1 value) {
        return std::find(rhs.begin(), rhs.end(), value) != rhs.end();
    });
}

bool generation_conflict(const std::vector<generation_access_v1>& lhs,
                         const std::vector<generation_access_v1>& rhs) noexcept {
    return std::any_of(lhs.begin(), lhs.end(), [&rhs](const generation_access_v1& access) {
        return std::any_of(rhs.begin(), rhs.end(), [&access](const generation_access_v1& other) {
            return access.value == other.value && access.generation != other.generation;
        });
    });
}

}  // namespace

statement_pair_analysis_v1 implement_statement_ordering_and_observable_effects_v1(
    const field_statement_semantics_v1& before,
    const field_statement_semantics_v1& after) noexcept {
    statement_pair_analysis_v1 result;
    if (before.statement_id == 0 || after.statement_id == 0 ||
        before.statement_id == after.statement_id) {
        result.reorder_blocker = ordering_blocker_v1::invalid_statement;
    } else if (intersects(before.writes, after.reads) ||
               intersects(before.reads, after.writes) ||
               intersects(before.writes, after.writes)) {
        result.reorder_blocker = ordering_blocker_v1::data_dependency;
    } else if (before.observable_effects != field_effect_none_v1 ||
               after.observable_effects != field_effect_none_v1) {
        result.reorder_blocker = ordering_blocker_v1::observable_effect;
    } else if (generation_conflict(before.generation_writes, after.generation_reads) ||
               generation_conflict(before.generation_reads, after.generation_writes) ||
               generation_conflict(before.generation_writes, after.generation_writes)) {
        result.reorder_blocker = ordering_blocker_v1::generation_dependency;
    }

    result.fusion_blocker = result.reorder_blocker;
    if (result.fusion_blocker == ordering_blocker_v1::none &&
        before.numerical_contract_id != after.numerical_contract_id) {
        result.fusion_blocker = ordering_blocker_v1::numerical_contract;
    }
    if (result.fusion_blocker == ordering_blocker_v1::none &&
        before.field_constraint_set_id != after.field_constraint_set_id) {
        result.fusion_blocker = ordering_blocker_v1::field_constraint;
    }
    return result;
}

}  // namespace Cellerator::compiler::sema::field
