#pragma once

#include <Cellerator/compute/projection_family/view_family_comparison_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::projection_family {

inline constexpr std::uint32_t max_cross_operation_candidates_v1 = 64;

enum class promotion_disposition_v1 : std::uint8_t {
    promote_generalized_family = 1,
    retain_specialized_family = 2,
    retain_measured_plurality = 3,
};

// Frozen bounded artifact. Candidate positions are compact local coordinates;
// candidates themselves retain globally stable 128-bit identities and 64-bit
// support identities/counts. The fixed bound makes pairwise dominance exact
// without permitting an atlas-scale quadratic scan.
struct cross_operation_pareto_artifact_v1 {
    operation::v2::stable_id artifact_identity{};
    operation::v2::stable_id evidence_set_identity{};
    support_family_identity_v1 family{};
    std::uint64_t expected_reuse = 0;
    std::uint32_t required_operations = 0;
    std::uint32_t candidate_count = 0;
    std::uint32_t frontier_count = 0;
    promotion_disposition_v1 disposition =
        promotion_disposition_v1::retain_measured_plurality;
    std::uint8_t reserved[3]{};
    std::uint32_t frontier_candidate_indices[
        max_cross_operation_candidates_v1]{};
};

enum class cross_operation_pareto_code_v1 : std::uint32_t {
    emitted = 0,
    invalid_artifact_identity,
    invalid_evidence_set_identity,
    empty_candidate_set,
    candidate_bound_exceeded,
    zero_expected_reuse,
    invalid_candidate,
    family_mismatch,
    operation_mismatch,
    duplicate_candidate_identity,
    arithmetic_overflow,
    empty_frontier,
    missing_output,
};

struct cross_operation_pareto_result_v1 {
    cross_operation_pareto_code_v1 code =
        cross_operation_pareto_code_v1::emitted;
    std::uint32_t candidate_index = 0;
    [[nodiscard]] constexpr bool emitted() const noexcept {
        return code == cross_operation_pareto_code_v1::emitted;
    }
};

struct measured_candidate_metrics_v1 {
    std::uint64_t total_ns = 0;
    std::uint64_t persistent_bytes = 0;
    std::uint64_t transient_bytes = 0;
    std::uint64_t launch_count = 0;
};

[[nodiscard]] constexpr bool dominates_v1(
    const measured_candidate_metrics_v1 &lhs,
    const measured_candidate_metrics_v1 &rhs) noexcept {
    const bool no_worse = lhs.total_ns <= rhs.total_ns
        && lhs.persistent_bytes <= rhs.persistent_bytes
        && lhs.transient_bytes <= rhs.transient_bytes
        && lhs.launch_count <= rhs.launch_count;
    const bool strictly_better = lhs.total_ns < rhs.total_ns
        || lhs.persistent_bytes < rhs.persistent_bytes
        || lhs.transient_bytes < rhs.transient_bytes
        || lhs.launch_count < rhs.launch_count;
    return no_worse && strictly_better;
}

[[nodiscard]] inline cross_operation_pareto_result_v1
emit_cross_operation_pareto_v1(
    operation::v2::stable_id artifact_identity,
    operation::v2::stable_id evidence_set_identity,
    const view_family_measurement_v1 *candidates,
    std::uint32_t candidate_count,
    std::uint64_t expected_reuse,
    cross_operation_pareto_artifact_v1 *output) noexcept {
    if (!operation::v2::valid_stable_id(artifact_identity)) {
        return {cross_operation_pareto_code_v1::invalid_artifact_identity};
    }
    if (!operation::v2::valid_stable_id(evidence_set_identity)) {
        return {cross_operation_pareto_code_v1::
                    invalid_evidence_set_identity};
    }
    if (candidates == nullptr || candidate_count == 0) {
        return {cross_operation_pareto_code_v1::empty_candidate_set};
    }
    if (candidate_count > max_cross_operation_candidates_v1) {
        return {cross_operation_pareto_code_v1::candidate_bound_exceeded};
    }
    if (expected_reuse == 0) {
        return {cross_operation_pareto_code_v1::zero_expected_reuse};
    }
    if (output == nullptr) {
        return {cross_operation_pareto_code_v1::missing_output};
    }
    *output = {};

    measured_candidate_metrics_v1 metrics[max_cross_operation_candidates_v1]{};
    const auto &reference = candidates[0];
    for (std::uint32_t index = 0; index < candidate_count; ++index) {
        const auto &candidate = candidates[index];
        if (!validate_view_family_measurement_v1(candidate).valid()) {
            return {cross_operation_pareto_code_v1::invalid_candidate, index};
        }
        if (!same_support_family_identity_v1(
                reference.family, candidate.family)) {
            return {cross_operation_pareto_code_v1::family_mismatch, index};
        }
        if (candidate.supported_operations
            != reference.supported_operations) {
            return {cross_operation_pareto_code_v1::operation_mismatch, index};
        }
        for (std::uint32_t prior = 0; prior < index; ++prior) {
            if (operation::v2::same_stable_id(
                    candidates[prior].candidate_identity,
                    candidate.candidate_identity)) {
                return {cross_operation_pareto_code_v1::
                            duplicate_candidate_identity,
                        index};
            }
        }
        if (!measured_total_ns_v1(
                candidate, expected_reuse, &metrics[index].total_ns)) {
            return {cross_operation_pareto_code_v1::arithmetic_overflow,
                    index};
        }
        metrics[index].persistent_bytes = candidate.persistent_bytes;
        metrics[index].transient_bytes = candidate.transient_bytes;
        metrics[index].launch_count = candidate.launch_count;
    }

    std::uint32_t frontier_count = 0;
    std::uint32_t specialized_count = 0;
    std::uint32_t generalized_count = 0;
    for (std::uint32_t candidate = 0;
         candidate < candidate_count;
         ++candidate) {
        bool dominated = false;
        for (std::uint32_t challenger = 0;
             challenger < candidate_count;
             ++challenger) {
            if (challenger != candidate
                && dominates_v1(metrics[challenger], metrics[candidate])) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            output->frontier_candidate_indices[frontier_count] = candidate;
            ++frontier_count;
            if (candidates[candidate].kind == view_family_kind_v1::specialized) {
                ++specialized_count;
            } else {
                ++generalized_count;
            }
        }
    }
    if (frontier_count == 0) {
        return {cross_operation_pareto_code_v1::empty_frontier};
    }

    promotion_disposition_v1 disposition =
        promotion_disposition_v1::retain_measured_plurality;
    if (generalized_count != 0 && specialized_count == 0) {
        disposition = promotion_disposition_v1::promote_generalized_family;
    } else if (specialized_count != 0 && generalized_count == 0) {
        disposition = promotion_disposition_v1::retain_specialized_family;
    }
    output->artifact_identity = artifact_identity;
    output->evidence_set_identity = evidence_set_identity;
    output->family = reference.family;
    output->expected_reuse = expected_reuse;
    output->required_operations = reference.supported_operations;
    output->candidate_count = candidate_count;
    output->frontier_count = frontier_count;
    output->disposition = disposition;
    return {cross_operation_pareto_code_v1::emitted, frontier_count};
}

static_assert(std::is_standard_layout_v<cross_operation_pareto_artifact_v1>);
static_assert(std::is_trivially_copyable_v<cross_operation_pareto_artifact_v1>);
static_assert(std::is_trivially_copyable_v<measured_candidate_metrics_v1>);

} // namespace cellerator::compute::projection_family
