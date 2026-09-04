#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

enum class modality_kind_v1 : std::uint8_t {
    transcriptome = 1,
    chromatin,
    protein,
    spatial,
    sequence,
    custom,
};

struct modality_identity_binding_v1 {
    persistent_atom_identity_v1 modality_identity{};
    persistent_atom_identity_v1 observation_domain_identity{};
    persistent_atom_identity_v1 observation_axis_identity{};
    persistent_atom_identity_v1 observation_order_identity{};
    persistent_atom_identity_v1 feature_domain_identity{};
    persistent_atom_identity_v1 feature_axis_identity{};
    persistent_atom_identity_v1 feature_order_identity{};
    persistent_atom_identity_v1 observation_to_subject_relation_identity{};
    std::uint64_t value_generation = 0;
    modality_kind_v1 kind = modality_kind_v1::custom;
};

struct multimodal_identity_spine_v1 {
    persistent_atom_identity_v1 spine_identity{};
    persistent_atom_identity_v1 cohort_identity{};
    persistent_atom_identity_v1 subject_domain_identity{};
    persistent_atom_identity_v1 subject_axis_identity{};
    persistent_atom_identity_v1 subject_order_identity{};
    std::uint64_t structure_epoch = 0;
    std::vector<modality_identity_binding_v1> modalities;
};

struct modality_overlay_v1 {
    persistent_atom_identity_v1 modality_identity{};
    persistent_atom_identity_v1 domain_identity{};
    persistent_atom_identity_v1 axis_identity{};
    persistent_atom_identity_v1 order_identity{};
    persistent_atom_identity_v1 geometry_identity{};
    persistent_atom_identity_v1 value_plane_identity{};
    std::uint64_t value_generation = 0;
};

struct cross_modal_relation_proposal_v1 {
    persistent_atom_identity_v1 proposal_identity{};
    persistent_atom_identity_v1 relation_identity{};
    persistent_atom_identity_v1 evidence_identity{};
    persistent_atom_identity_v1 source_modality_identity{};
    persistent_atom_identity_v1 source_domain_identity{};
    persistent_atom_identity_v1 source_axis_identity{};
    persistent_atom_identity_v1 source_entity_identity{};
    persistent_atom_identity_v1 destination_modality_identity{};
    persistent_atom_identity_v1 destination_domain_identity{};
    persistent_atom_identity_v1 destination_axis_identity{};
    persistent_atom_identity_v1 destination_entity_identity{};
    std::int64_t confidence_numerator = 0;
    std::uint64_t confidence_denominator = 1;
    bool directed = true;
};

enum class multimodal_discovery_status_v1 : std::uint8_t {
    success = 0,
    invalid_spine,
    insufficient_modalities,
    invalid_modality,
    duplicate_modality,
    subject_identity_mismatch,
    missing_subject_relation,
    overlay_count_mismatch,
    invalid_overlay,
    stale_value_generation,
    invalid_proposal,
    unknown_modality,
    not_cross_modal,
    endpoint_identity_mismatch,
    duplicate_proposal,
    allocation_failure,
};

[[nodiscard]] multimodal_discovery_status_v1 discover_multimodal_identity_spine_v1(
    const multimodal_identity_spine_v1& spine,
    const std::vector<modality_overlay_v1>& overlays,
    const std::vector<cross_modal_relation_proposal_v1>& candidates,
    std::vector<cross_modal_relation_proposal_v1>* output) noexcept;

[[nodiscard]] constexpr bool authorizes_execution(
    const cross_modal_relation_proposal_v1&) noexcept {
    return false;
}

}  // namespace Cellerator::compiler::discovery
