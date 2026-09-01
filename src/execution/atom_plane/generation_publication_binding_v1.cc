#include "Cellerator/execution/atom_plane/generation_publication_binding_v1.hh"

namespace cellerator::execution::atom_plane {
namespace {

using namespace projection_value_plane;

atom_generation_publication_status_v1 failure(
    atom_generation_publication_code_v1 code,
    u64 subject = 0u,
    u32 component_index = 0u,
    relation_value_atom_plane_code_v1 atom_code =
        relation_value_atom_plane_code_v1::success) noexcept {
    return {code, atom_code, 0u, component_index, subject};
}

}  // namespace

atom_generation_publication_status_v1
validate_atom_generation_publication_binding_v1(
    const atom_generation_publication_binding_v1 &binding,
    composite_validation_workspace_v1 composite_workspace) noexcept {
    if (binding.schema_version != atom_generation_publication_schema_v1
        || binding.reserved != 0u || binding.atom == nullptr
        || binding.publication == nullptr) {
        return failure(atom_generation_publication_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(binding.plane_identity)) {
        return failure(
            atom_generation_publication_code_v1::invalid_plane_identity);
    }
    const relation_value_atom_plane_status_v1 atom_status =
        validate_relation_value_atom_plane_v1(
            *binding.atom, composite_workspace, nullptr);
    if (!atom_status) {
        return failure(atom_generation_publication_code_v1::invalid_atom,
            atom_status.subject, 0u, atom_status.code);
    }
    const projection_value_plane_v1 &values = *binding.atom->values;
    const generation_publication_v1 &publication = *binding.publication;
    if (binding.atom_generation.value == 0u
        || binding.atom_generation.value != values.generation.value) {
        return failure(
            atom_generation_publication_code_v1::stale_atom_generation,
            binding.atom_generation.value);
    }
    if (!same_structure_handle(publication.structure, values.structure)) {
        return failure(
            atom_generation_publication_code_v1::stale_publication_structure);
    }
    if (publication.structure_epoch_value.value
        != values.structure_epoch_value.value) {
        return failure(
            atom_generation_publication_code_v1::stale_publication_epoch,
            publication.structure_epoch_value.value);
    }
    if (publication.generation.value != binding.atom_generation.value) {
        return failure(
            atom_generation_publication_code_v1::stale_publication_generation,
            publication.generation.value);
    }
    if (publication.phase != generation_publication_phase_v1::published) {
        return failure(
            atom_generation_publication_code_v1::publication_not_ready,
            publication.ready_count);
    }
    if (publication.required_component_count
            != values.required_component_count
        || publication.ready_count != values.required_component_count) {
        return failure(atom_generation_publication_code_v1::
                ready_component_count_mismatch,
            publication.ready_count);
    }
    if (publication.ready_components == nullptr) {
        return failure(
            atom_generation_publication_code_v1::missing_ready_components);
    }
    for (u32 index = 0u; index < publication.required_component_count;
         ++index) {
        if (publication.ready_components[index] == 0u) {
            return failure(atom_generation_publication_code_v1::
                    incomplete_ready_components,
                index, index);
        }
    }
    return {};
}

}  // namespace cellerator::execution::atom_plane
