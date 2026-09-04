#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

enum class experimental_proposal_strategy_v1 : std::uint8_t {
    factor = 1,
    bicluster,
    support_signature,
};

struct experimental_proposal_candidate_v1 {
    persistent_atom_identity_v1 proposal_identity{};
    persistent_atom_identity_v1 evidence_identity{};
    persistent_atom_identity_v1 source_domain_identity{};
    persistent_atom_identity_v1 destination_domain_identity{};
    experimental_proposal_strategy_v1 strategy =
        experimental_proposal_strategy_v1::factor;
    bool approximate = true;
    std::uint64_t confidence_numerator = 0;
    std::uint64_t confidence_denominator = 1;
    std::uint64_t work_items = 0;
    std::uint64_t member_count = 0;
    std::uint64_t exact_covered_members = 0;
    std::uint64_t observed_quality_numerator = 0;
    std::uint64_t observed_quality_denominator = 1;
    std::uint64_t null_quality_numerator = 0;
    std::uint64_t null_quality_denominator = 1;
};

enum class experimental_proposal_disposition_v1 : std::uint8_t {
    evaluated_not_promoted = 1,
    candidate_supported,
};

struct experimental_proposal_evaluation_v1 {
    experimental_proposal_candidate_v1 candidate{};
    experimental_proposal_disposition_v1 disposition =
        experimental_proposal_disposition_v1::evaluated_not_promoted;
};

struct experimental_proposal_policy_v1 {
    std::uint64_t maximum_candidates = 0;
    std::uint64_t maximum_total_work_items = 0;
    std::uint64_t minimum_members = 0;
    std::uint64_t minimum_confidence_numerator = 0;
    std::uint64_t minimum_confidence_denominator = 1;
    std::uint64_t minimum_quality_numerator = 0;
    std::uint64_t minimum_quality_denominator = 1;
};

enum class experimental_proposal_status_v1 : std::uint8_t {
    success = 0,
    invalid_policy,
    candidate_bound_exceeded,
    invalid_candidate,
    duplicate_proposal,
    work_bound_exceeded,
    allocation_failure,
};

[[nodiscard]] experimental_proposal_status_v1
evaluate_experimental_proposal_strategies_v1(
    const std::vector<experimental_proposal_candidate_v1>& candidates,
    experimental_proposal_policy_v1 policy,
    std::vector<experimental_proposal_evaluation_v1>* output) noexcept;

[[nodiscard]] constexpr bool authorizes_execution(
    const experimental_proposal_evaluation_v1&) noexcept {
    return false;
}

}  // namespace Cellerator::compiler::discovery
