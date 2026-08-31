#pragma once

#include <Cellerator/execution/projection_value_plane/canonical_transfer_v1.hh>
#include <Cellerator/execution/projection_value_plane/composite_plane_v1.hh>
#include <Cellerator/execution/projection_value_plane/generation_publication_v1.hh>
#include <Cellerator/execution/projection_value_plane/value_pack_portfolio_v1.hh>

namespace cellerator::execution::projection_value_plane {

// Frozen validation bundle for a published projection-primary generation.
// Optional portfolio validation is included when portfolio is non-null.
struct frozen_projection_value_plane_v1 {
    const relation_structure *structure = nullptr;
    const projection_value_plane_v1 *plane = nullptr;
    const generation_publication_v1 *publication = nullptr;
    const value_pack_portfolio_v1 *portfolio = nullptr;
    logical_value_index_v1 logical_index{};
    composite_validation_workspace_v1 composite_workspace{};
};

value_plane_status_v1 validate_frozen_projection_value_plane_v1(
    const frozen_projection_value_plane_v1 &frozen,
    composite_validation_result_v1 *composite_result) noexcept;

}  // namespace cellerator::execution::projection_value_plane
