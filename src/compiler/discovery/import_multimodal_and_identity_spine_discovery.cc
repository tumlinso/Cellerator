#include <Cellerator/compiler/discovery/import_multimodal_and_identity_spine_discovery_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::discovery {
namespace {

bool empty_identity_v1(persistent_atom_identity_v1 identity) noexcept {
    return identity.producer_namespace == 0 && identity.local_identity == 0;
}

bool valid_modality_kind_v1(modality_kind_v1 kind) noexcept {
    const auto value = static_cast<std::uint8_t>(kind);
    return value >= static_cast<std::uint8_t>(modality_kind_v1::transcriptome) &&
        value <= static_cast<std::uint8_t>(modality_kind_v1::custom);
}

const modality_identity_binding_v1* find_modality_v1(
    const multimodal_identity_spine_v1& spine,
    persistent_atom_identity_v1 identity) noexcept {
    const auto found = std::find_if(
        spine.modalities.begin(), spine.modalities.end(),
        [identity](const auto& modality) {
            return modality.modality_identity == identity;
        });
    return found == spine.modalities.end() ? nullptr : &*found;
}

const modality_overlay_v1* find_overlay_v1(
    const std::vector<modality_overlay_v1>& overlays,
    persistent_atom_identity_v1 identity) noexcept {
    const auto found = std::find_if(
        overlays.begin(), overlays.end(), [identity](const auto& overlay) {
            return overlay.modality_identity == identity;
        });
    return found == overlays.end() ? nullptr : &*found;
}

bool valid_proposal_identities_v1(
    const cross_modal_relation_proposal_v1& proposal) noexcept {
    return valid_persistent_atom_identity_v1(proposal.proposal_identity) &&
        valid_persistent_atom_identity_v1(proposal.relation_identity) &&
        valid_persistent_atom_identity_v1(proposal.evidence_identity) &&
        valid_persistent_atom_identity_v1(proposal.source_modality_identity) &&
        valid_persistent_atom_identity_v1(proposal.source_domain_identity) &&
        valid_persistent_atom_identity_v1(proposal.source_axis_identity) &&
        valid_persistent_atom_identity_v1(proposal.source_entity_identity) &&
        valid_persistent_atom_identity_v1(proposal.destination_modality_identity) &&
        valid_persistent_atom_identity_v1(proposal.destination_domain_identity) &&
        valid_persistent_atom_identity_v1(proposal.destination_axis_identity) &&
        valid_persistent_atom_identity_v1(proposal.destination_entity_identity) &&
        proposal.confidence_denominator != 0;
}

}  // namespace

multimodal_discovery_status_v1 discover_multimodal_identity_spine_v1(
    const multimodal_identity_spine_v1& spine,
    const std::vector<modality_overlay_v1>& overlays,
    const std::vector<cross_modal_relation_proposal_v1>& candidates,
    std::vector<cross_modal_relation_proposal_v1>* output) noexcept {
    if (output == nullptr || !valid_persistent_atom_identity_v1(spine.spine_identity) ||
        !valid_persistent_atom_identity_v1(spine.cohort_identity) ||
        !valid_persistent_atom_identity_v1(spine.subject_domain_identity) ||
        !valid_persistent_atom_identity_v1(spine.subject_axis_identity) ||
        !valid_persistent_atom_identity_v1(spine.subject_order_identity) ||
        spine.structure_epoch == 0) {
        return multimodal_discovery_status_v1::invalid_spine;
    }
    if (spine.modalities.size() < 2) {
        return multimodal_discovery_status_v1::insufficient_modalities;
    }
    if (overlays.size() != spine.modalities.size()) {
        return multimodal_discovery_status_v1::overlay_count_mismatch;
    }

    for (std::size_t index = 0; index < spine.modalities.size(); ++index) {
        const auto& modality = spine.modalities[index];
        if (!valid_persistent_atom_identity_v1(modality.modality_identity) ||
            !valid_persistent_atom_identity_v1(modality.observation_domain_identity) ||
            !valid_persistent_atom_identity_v1(modality.observation_axis_identity) ||
            !valid_persistent_atom_identity_v1(modality.observation_order_identity) ||
            !valid_persistent_atom_identity_v1(modality.feature_domain_identity) ||
            !valid_persistent_atom_identity_v1(modality.feature_axis_identity) ||
            !valid_persistent_atom_identity_v1(modality.feature_order_identity) ||
            modality.value_generation == 0 || !valid_modality_kind_v1(modality.kind)) {
            return multimodal_discovery_status_v1::invalid_modality;
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (spine.modalities[previous].modality_identity ==
                modality.modality_identity) {
                return multimodal_discovery_status_v1::duplicate_modality;
            }
        }
        if (modality.observation_domain_identity == spine.subject_domain_identity) {
            if (modality.observation_axis_identity != spine.subject_axis_identity ||
                modality.observation_order_identity != spine.subject_order_identity ||
                !empty_identity_v1(
                    modality.observation_to_subject_relation_identity)) {
                return multimodal_discovery_status_v1::subject_identity_mismatch;
            }
        } else if (!valid_persistent_atom_identity_v1(
                       modality.observation_to_subject_relation_identity)) {
            return multimodal_discovery_status_v1::missing_subject_relation;
        }

        const auto* overlay = find_overlay_v1(overlays, modality.modality_identity);
        if (overlay == nullptr ||
            !valid_persistent_atom_identity_v1(overlay->geometry_identity) ||
            !valid_persistent_atom_identity_v1(overlay->value_plane_identity) ||
            overlay->domain_identity != modality.feature_domain_identity ||
            overlay->axis_identity != modality.feature_axis_identity ||
            overlay->order_identity != modality.feature_order_identity) {
            return multimodal_discovery_status_v1::invalid_overlay;
        }
        if (overlay->value_generation != modality.value_generation) {
            return multimodal_discovery_status_v1::stale_value_generation;
        }
    }

    try {
        std::vector<cross_modal_relation_proposal_v1> proposals;
        proposals.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            if (!valid_proposal_identities_v1(candidate)) {
                return multimodal_discovery_status_v1::invalid_proposal;
            }
            const auto* source =
                find_modality_v1(spine, candidate.source_modality_identity);
            const auto* destination =
                find_modality_v1(spine, candidate.destination_modality_identity);
            if (source == nullptr || destination == nullptr) {
                return multimodal_discovery_status_v1::unknown_modality;
            }
            if (source == destination) {
                return multimodal_discovery_status_v1::not_cross_modal;
            }
            const auto* source_overlay =
                find_overlay_v1(overlays, source->modality_identity);
            const auto* destination_overlay =
                find_overlay_v1(overlays, destination->modality_identity);
            if (candidate.source_domain_identity != source_overlay->domain_identity ||
                candidate.source_axis_identity != source_overlay->axis_identity ||
                candidate.destination_domain_identity !=
                    destination_overlay->domain_identity ||
                candidate.destination_axis_identity !=
                    destination_overlay->axis_identity) {
                return multimodal_discovery_status_v1::endpoint_identity_mismatch;
            }
            proposals.push_back(candidate);
        }
        std::sort(proposals.begin(), proposals.end(), [](const auto& left, const auto& right) {
            return persistent_atom_identity_less_v1(
                left.proposal_identity, right.proposal_identity);
        });
        for (std::size_t index = 1; index < proposals.size(); ++index) {
            if (proposals[index - 1].proposal_identity ==
                proposals[index].proposal_identity) {
                return multimodal_discovery_status_v1::duplicate_proposal;
            }
        }
        *output = std::move(proposals);
        return multimodal_discovery_status_v1::success;
    } catch (...) {
        return multimodal_discovery_status_v1::allocation_failure;
    }
}

}  // namespace Cellerator::compiler::discovery
