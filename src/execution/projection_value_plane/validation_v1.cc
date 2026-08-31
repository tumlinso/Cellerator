#include "Cellerator/execution/projection_value_plane/validation_v1.hh"

namespace cellerator::execution::projection_value_plane {
namespace {

value_plane_status_v1 failure(
    value_plane_status_code_v1 code,
    u64 subject) noexcept {
    return {code, subject};
}

}  // namespace

value_plane_status_v1 validate_frozen_projection_value_plane_v1(
    const frozen_projection_value_plane_v1 &frozen,
    composite_validation_result_v1 *composite_result) noexcept {
    if (frozen.structure == nullptr || frozen.plane == nullptr
        || frozen.publication == nullptr) {
        return failure(value_plane_status_code_v1::invalid_argument, 0u);
    }
    const value_plane_status_v1 plane_status =
        validate_projection_value_plane_v1(*frozen.structure, *frozen.plane);
    if (!plane_status) {
        return plane_status;
    }
    const value_plane_status_v1 composite_status =
        validate_composite_projection_values_v1(*frozen.plane,
            frozen.composite_workspace, composite_result);
    if (!composite_status) {
        return composite_status;
    }
    if (frozen.publication->phase
            != generation_publication_phase_v1::published
        || !same_structure_handle(frozen.plane->structure,
            frozen.publication->structure)
        || frozen.plane->structure_epoch_value.value
            != frozen.publication->structure_epoch_value.value
        || frozen.plane->generation.value
            != frozen.publication->generation.value
        || frozen.publication->required_component_count
            != frozen.plane->required_component_count
        || frozen.publication->ready_count
            != frozen.plane->required_component_count
        || frozen.publication->ready_components == nullptr) {
        return failure(value_plane_status_code_v1::not_ready,
            frozen.plane->generation.value);
    }
    for (u32 index = 0u; index < frozen.plane->required_component_count;
         ++index) {
        if (frozen.publication->ready_components[index] == 0u) {
            return failure(value_plane_status_code_v1::not_ready, index);
        }
    }
    const value_plane_status_v1 index_status =
        build_logical_value_index_v1(*frozen.plane, frozen.logical_index);
    if (!index_status) {
        return index_status;
    }
    if (frozen.portfolio != nullptr) {
        const value_plane_status_v1 portfolio_status =
            validate_value_pack_portfolio_v1(*frozen.structure,
                *frozen.portfolio);
        if (!portfolio_status) {
            return portfolio_status;
        }
    }
    return {};
}

}  // namespace cellerator::execution::projection_value_plane
