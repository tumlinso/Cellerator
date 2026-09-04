#include <Cellerator/compiler/discovery/import_factor_bicluster_and_signature_proposal_strategie_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::discovery {
namespace {

bool valid_strategy_v1(experimental_proposal_strategy_v1 strategy) noexcept {
    const auto value = static_cast<std::uint8_t>(strategy);
    return value >= static_cast<std::uint8_t>(
                        experimental_proposal_strategy_v1::factor) &&
        value <= static_cast<std::uint8_t>(
                     experimental_proposal_strategy_v1::support_signature);
}

bool fraction_at_least_v1(std::uint64_t left_numerator,
                          std::uint64_t left_denominator,
                          std::uint64_t right_numerator,
                          std::uint64_t right_denominator) noexcept {
    bool reversed = false;
    for (;;) {
        const auto left_quotient = left_numerator / left_denominator;
        const auto right_quotient = right_numerator / right_denominator;
        if (left_quotient != right_quotient) {
            return reversed ? left_quotient < right_quotient
                            : left_quotient > right_quotient;
        }
        left_numerator %= left_denominator;
        right_numerator %= right_denominator;
        if (left_numerator == 0 || right_numerator == 0) {
            if (left_numerator == right_numerator) {
                return true;
            }
            return reversed ? left_numerator != 0 : right_numerator == 0;
        }
        const auto next_left_numerator = left_denominator;
        const auto next_right_numerator = right_denominator;
        left_denominator = left_numerator;
        right_denominator = right_numerator;
        left_numerator = next_left_numerator;
        right_numerator = next_right_numerator;
        reversed = !reversed;
    }
}

bool candidate_less_v1(const experimental_proposal_candidate_v1& left,
                       const experimental_proposal_candidate_v1& right) noexcept {
    return persistent_atom_identity_less_v1(left.proposal_identity,
                                            right.proposal_identity);
}

}  // namespace

experimental_proposal_status_v1 evaluate_experimental_proposal_strategies_v1(
    const std::vector<experimental_proposal_candidate_v1>& candidates,
    experimental_proposal_policy_v1 policy,
    std::vector<experimental_proposal_evaluation_v1>* output) noexcept {
    if (output == nullptr || policy.maximum_candidates == 0 ||
        policy.maximum_total_work_items == 0 || policy.minimum_members == 0 ||
        policy.minimum_confidence_denominator == 0 ||
        policy.minimum_quality_denominator == 0) {
        return experimental_proposal_status_v1::invalid_policy;
    }
    if (candidates.size() > policy.maximum_candidates) {
        return experimental_proposal_status_v1::candidate_bound_exceeded;
    }
    try {
        auto ordered = candidates;
        std::sort(ordered.begin(), ordered.end(), candidate_less_v1);
        std::vector<experimental_proposal_evaluation_v1> evaluations;
        evaluations.reserve(ordered.size());
        std::uint64_t total_work = 0;
        for (std::size_t index = 0; index < ordered.size(); ++index) {
            const auto& candidate = ordered[index];
            if (!valid_persistent_atom_identity_v1(candidate.proposal_identity) ||
                !valid_persistent_atom_identity_v1(candidate.evidence_identity) ||
                !valid_persistent_atom_identity_v1(
                    candidate.source_domain_identity) ||
                !valid_persistent_atom_identity_v1(
                    candidate.destination_domain_identity) ||
                !valid_strategy_v1(candidate.strategy) ||
                candidate.confidence_denominator == 0 || candidate.work_items == 0 ||
                candidate.member_count == 0 ||
                candidate.exact_covered_members > candidate.member_count ||
                candidate.observed_quality_denominator == 0 ||
                candidate.null_quality_denominator == 0) {
                return experimental_proposal_status_v1::invalid_candidate;
            }
            if (index != 0 && ordered[index - 1].proposal_identity ==
                                      candidate.proposal_identity) {
                return experimental_proposal_status_v1::duplicate_proposal;
            }
            if (candidate.work_items >
                policy.maximum_total_work_items - total_work) {
                return experimental_proposal_status_v1::work_bound_exceeded;
            }
            total_work += candidate.work_items;

            const bool exact_coverage =
                candidate.exact_covered_members == candidate.member_count;
            const bool enough_members =
                candidate.member_count >= policy.minimum_members;
            const bool confidence_passed = fraction_at_least_v1(
                candidate.confidence_numerator, candidate.confidence_denominator,
                policy.minimum_confidence_numerator,
                policy.minimum_confidence_denominator);
            const bool quality_passed = fraction_at_least_v1(
                candidate.observed_quality_numerator,
                candidate.observed_quality_denominator,
                policy.minimum_quality_numerator,
                policy.minimum_quality_denominator);
            const bool beats_null = !fraction_at_least_v1(
                candidate.null_quality_numerator,
                candidate.null_quality_denominator,
                candidate.observed_quality_numerator,
                candidate.observed_quality_denominator);
            const auto disposition =
                exact_coverage && enough_members && confidence_passed &&
                    quality_passed && beats_null
                ? experimental_proposal_disposition_v1::candidate_supported
                : experimental_proposal_disposition_v1::evaluated_not_promoted;
            evaluations.push_back({candidate, disposition});
        }
        *output = std::move(evaluations);
        return experimental_proposal_status_v1::success;
    } catch (...) {
        return experimental_proposal_status_v1::allocation_failure;
    }
}

}  // namespace Cellerator::compiler::discovery
