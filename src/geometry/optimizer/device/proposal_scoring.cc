#include "Cellerator/geometry/optimizer/device/proposal_scoring.h"

#include <cstdint>
#include <limits>

namespace cellerator::geometry::optimizer::device {
namespace {

bool checked_add(std::int64_t left, std::int64_t right,
                 std::int64_t* result) noexcept {
    if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
        return false;
    }
    *result = left + right;
    return true;
}

bool checked_multiply(std::int64_t left, std::int64_t right,
                      std::int64_t* result) noexcept {
    if (left == 0 || right == 0) {
        *result = 0;
        return true;
    }
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    if ((left > 0 && ((right > 0 && left > maximum / right) ||
                      (right < 0 && right < minimum / left))) ||
        (left < 0 && ((right > 0 && left < minimum / right) ||
                      (right < 0 && right < maximum / left)))) return false;
    *result = left * right;
    return true;
}

bool valid_span(const proposal_score_span_v1& span,
                std::uint64_t term_count) noexcept {
    return span.first_term <= term_count &&
           span.term_count <= term_count - span.first_term;
}

}  // namespace

proposal_scoring_report_v1 score_proposals_host_v1(
        const proposal_scoring_problem_v1& problem,
        proposal_score_result_v1* results,
        std::uint64_t result_capacity) noexcept {
    proposal_scoring_report_v1 report{};
    if ((problem.proposal_count != 0 && problem.proposals == nullptr) ||
        (problem.term_count != 0 && problem.terms == nullptr) ||
        (problem.proposal_count != 0 && results == nullptr)) {
        return report;
    }
    if (result_capacity < problem.proposal_count) {
        report.status = proposal_scoring_status::insufficient_capacity;
        return report;
    }
    report.status = proposal_scoring_status::success;
    for (std::uint64_t proposal_index = 0;
         proposal_index < problem.proposal_count; ++proposal_index) {
        const auto& span = problem.proposals[proposal_index];
        auto& output = results[proposal_index];
        output = {};
        output.stable_proposal_id = span.stable_proposal_id;
        if (!valid_span(span, problem.term_count)) {
            output.flags = proposal_score_invalid_span;
            report.status = proposal_scoring_status::invalid_span;
            report.first_invalid_proposal = proposal_index;
            return report;
        }
        for (std::uint64_t local = 0; local < span.term_count; ++local) {
            const auto& term = problem.terms[span.first_term + local];
            for (std::uint32_t component = 0;
                 component < score_component_count; ++component) {
                std::int64_t weighted = 0;
                std::int64_t next = 0;
                if (!checked_multiply(term.component_delta[component],
                                      problem.weights.component[component],
                                      &weighted) ||
                    !checked_add(output.weighted_objective_delta, weighted,
                                 &next)) {
                    output.flags |= proposal_score_arithmetic_overflow;
                    report.status = proposal_scoring_status::arithmetic_overflow;
                    report.first_invalid_proposal = proposal_index;
                    return report;
                }
                output.weighted_objective_delta = next;
            }
            if (!checked_add(output.mma_interaction_delta,
                             term.mma_interaction_delta,
                             &output.mma_interaction_delta) ||
                !checked_add(output.residual_interaction_delta,
                             term.residual_interaction_delta,
                             &output.residual_interaction_delta)) {
                output.flags |= proposal_score_arithmetic_overflow;
                report.status = proposal_scoring_status::arithmetic_overflow;
                report.first_invalid_proposal = proposal_index;
                return report;
            }
        }
        ++report.scored_proposals;
    }
    return report;
}

}  // namespace cellerator::geometry::optimizer::device
