#include "Cellerator/geometry/optimizer/device/exact_census.h"

#include <cstdint>
#include <limits>

namespace cellerator::geometry::optimizer::device {
namespace {

bool add_u64(std::uint64_t value, std::uint64_t* total) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - *total) return false;
    *total += value;
    return true;
}

bool multiply_u64(std::uint64_t left, std::uint64_t right,
                  std::uint64_t* result) noexcept {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        return false;
    }
    *result = left * right;
    return true;
}

bool signed_delta(std::uint64_t after, std::uint64_t before,
                  std::int64_t* result) noexcept {
    const auto magnitude = after >= before ? after - before : before - after;
    if (magnitude > static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())) return false;
    const auto value = static_cast<std::int64_t>(magnitude);
    *result = after >= before ? value : -value;
    return true;
}

}  // namespace

exact_census_status exact_census_host_v1(
        const exact_census_problem_v1& problem,
        exact_census_result_v1* results,
        std::uint64_t result_capacity) noexcept {
    if ((problem.proposal_count != 0 &&
         (problem.proposals == nullptr || results == nullptr)) ||
        (problem.change_count != 0 && problem.changes == nullptr)) {
        return exact_census_status::invalid_argument;
    }
    if (result_capacity < problem.proposal_count) {
        return exact_census_status::insufficient_capacity;
    }
    exact_census_status status = exact_census_status::success;
    for (std::uint64_t proposal = 0; proposal < problem.proposal_count; ++proposal) {
        const auto span = problem.proposals[proposal];
        auto& output = results[proposal];
        output = {};
        output.stable_proposal_id = span.stable_proposal_id;
        if (span.first_change > problem.change_count ||
            span.change_count > problem.change_count - span.first_change) {
            output.flags = exact_census_invalid_span;
            status = exact_census_status::invalid_census;
            continue;
        }
        std::uint64_t before_mma = 0;
        std::uint64_t before_residual = 0;
        std::uint64_t after_mma = 0;
        std::uint64_t after_residual = 0;
        for (std::uint64_t local = 0; local < span.change_count; ++local) {
            const auto& change = problem.changes[span.first_change + local];
            std::uint64_t slots = 0;
            const bool no_overflow =
                    multiply_u64(change.source_count, change.destination_count,
                                 &slots) &&
                    add_u64(change.before_mma, &before_mma) &&
                    add_u64(change.before_residual, &before_residual) &&
                    add_u64(change.after_mma, &after_mma) &&
                    add_u64(change.after_residual, &after_residual) &&
                    add_u64(slots, &output.after_physical_slots);
            if (!no_overflow) {
                output.flags |= exact_census_arithmetic_overflow;
                break;
            }
            const auto maximum = std::numeric_limits<std::uint64_t>::max();
            if (change.before_residual > maximum - change.before_mma ||
                change.after_residual > maximum - change.after_mma) {
                output.flags |= exact_census_arithmetic_overflow;
            } else if (change.before_mma + change.before_residual !=
                       change.after_mma + change.after_residual) {
                output.flags |= exact_census_nonunique_contribution;
            }
            if (change.after_mma > slots) {
                output.flags |= exact_census_rectangle_overfull;
            }
        }
        if (output.flags == exact_census_valid) {
            if (!add_u64(before_mma, &output.before_interactions) ||
                !add_u64(before_residual, &output.before_interactions) ||
                !add_u64(after_mma, &output.after_interactions) ||
                !add_u64(after_residual, &output.after_interactions) ||
                after_mma > output.after_physical_slots ||
                !signed_delta(after_mma, before_mma, &output.mma_delta) ||
                !signed_delta(after_residual, before_residual,
                              &output.residual_delta)) {
                output.flags |= exact_census_arithmetic_overflow;
            } else {
                output.after_padding_slots =
                        output.after_physical_slots - after_mma;
            }
        }
        if (output.flags != exact_census_valid) {
            status = exact_census_status::invalid_census;
        }
    }
    return status;
}

}  // namespace cellerator::geometry::optimizer::device
