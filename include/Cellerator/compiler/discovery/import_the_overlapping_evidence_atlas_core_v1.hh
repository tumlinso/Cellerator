#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstdint>
#include <optional>
#include <vector>

namespace Cellerator::compiler::discovery {

enum class negative_evidence_reason_v1 : std::uint8_t {
    none = 0,
    not_observed,
    contradicted,
    unstable,
    bounded_search_exhausted,
    candidate_cap_reached,
    complete_cost_nonpromotion,
};

enum class exact_rescan_status_v1 : std::uint8_t {
    not_performed = 1,
    complete,
    incomplete,
};

struct proposal_evidence_record_v1 {
    persistent_atom_identity_v1 evidence_identity{};
    persistent_atom_identity_v1 subject_atom_identity{};
    persistent_atom_identity_v1 provenance_identity{};
    std::uint64_t observation_generation = 0;
    std::vector<persistent_atom_identity_v1> approximate_members;
    std::uint64_t confidence_numerator = 0;
    std::uint64_t confidence_denominator = 0;
    std::uint64_t stable_resamples = 0;
    std::uint64_t total_resamples = 0;
    std::uint64_t exact_visited = 0;
    std::uint64_t exact_assigned = 0;
    negative_evidence_reason_v1 negative_reason = negative_evidence_reason_v1::none;
    exact_rescan_status_v1 exact_rescan = exact_rescan_status_v1::not_performed;
};

struct overlapping_evidence_atlas_v1 {
    persistent_atom_identity_v1 atlas_identity{};
    std::uint64_t generation = 0;
    std::vector<proposal_evidence_record_v1> proposals;
};

enum class evidence_atlas_status_v1 : std::uint8_t {
    success = 0,
    invalid_atlas_identity,
    missing_generation,
    empty_atlas,
    invalid_record_identity,
    duplicate_record,
    invalid_member,
    unordered_or_duplicate_member,
    invalid_confidence,
    invalid_stability,
    invalid_exact_rescan,
    invalid_negative_reason,
    invalid_image,
    checksum_mismatch,
};

[[nodiscard]] evidence_atlas_status_v1 validate_overlapping_evidence_atlas_v1(
    const overlapping_evidence_atlas_v1& atlas) noexcept;

[[nodiscard]] std::optional<std::vector<std::uint8_t>>
serialize_overlapping_evidence_atlas_v1(
    const overlapping_evidence_atlas_v1& atlas,
    evidence_atlas_status_v1* status = nullptr) noexcept;

[[nodiscard]] std::optional<overlapping_evidence_atlas_v1>
deserialize_overlapping_evidence_atlas_v1(
    const std::vector<std::uint8_t>& image,
    evidence_atlas_status_v1* status = nullptr) noexcept;

[[nodiscard]] bool equivalent_evidence_atlas_v1(
    const overlapping_evidence_atlas_v1& left,
    const overlapping_evidence_atlas_v1& right) noexcept;

}  // namespace Cellerator::compiler::discovery
