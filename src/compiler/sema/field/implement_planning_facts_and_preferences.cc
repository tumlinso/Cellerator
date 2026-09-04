#include <Cellerator/compiler/sema/field/implement_planning_facts_and_preferences_v1.hh>

#include <array>
#include <cmath>
#include <utility>

namespace Cellerator::compiler::sema::field {
namespace {

[[nodiscard]] constexpr std::size_t kind_index(
    planning_fact_or_preference_kind_v1 kind) noexcept {
    return static_cast<std::size_t>(kind) - 1;
}

}  // namespace

planning_facts_and_preferences_status_v1 implement_planning_facts_and_preferences_v1(
    const std::vector<planning_fact_or_preference_v1>& hints,
    planning_facts_and_preferences_v1* resolved) noexcept {
    if (resolved == nullptr) return planning_facts_and_preferences_status_v1::invalid_output;
    constexpr std::size_t kind_count = 8;
    std::array<std::size_t, kind_count> winners{};
    std::array<bool, kind_count> has_winner{};
    planning_facts_and_preferences_v1 result;
    result.hints.reserve(hints.size());

    for (const auto& hint : hints) {
        const auto index = kind_index(hint.kind);
        if (hint.source_identity == 0 || index >= kind_count ||
            !std::isfinite(hint.magnitude) || hint.magnitude <= 0.0) {
            return planning_facts_and_preferences_status_v1::invalid_hint;
        }

        resolved_planning_hint_v1 item;
        item.hint = hint;
        if (!hint.supported) {
            item.disposition = planning_hint_disposition_v1::ignored;
            item.diagnostic = "planning hint ignored because the target does not support it";
            ++result.ignored_count;
        } else if (!has_winner[index]) {
            item.disposition = planning_hint_disposition_v1::applied;
            winners[index] = result.hints.size();
            has_winner[index] = true;
            ++result.applied_count;
        } else {
            auto& winner = result.hints[winners[index]];
            if (hint.magnitude > winner.hint.magnitude) {
                winner.disposition = planning_hint_disposition_v1::dominated;
                winner.diagnostic = "planning hint dominated by a stronger scoped hint";
                --result.applied_count;
                ++result.dominated_count;
                item.disposition = planning_hint_disposition_v1::applied;
                winners[index] = result.hints.size();
                ++result.applied_count;
            } else {
                item.disposition = planning_hint_disposition_v1::dominated;
                item.diagnostic = "planning hint dominated by a stronger scoped hint";
                ++result.dominated_count;
            }
        }
        result.hints.push_back(std::move(item));
    }

    *resolved = std::move(result);
    return planning_facts_and_preferences_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
