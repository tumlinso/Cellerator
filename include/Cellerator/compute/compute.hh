#pragma once

// Public CE-EXOP operation portfolio umbrella. Stable semantics remain in the
// operation core; these headers expose target-specific physical candidates to
// explicit catalog/planner selection without declaring any candidate canonical.
#include <Cellerator/compute/operation/operation_core_v2.hh>

#include <Cellerator/compute/architecture/providers/nvidia/sm70/residual/portfolio_v1.h>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_inventory_v1.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_reference_v1.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_catalog_v1.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/exact_validation_v1.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_integration_v1.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_validation_v1.hh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_value_map_v1.hh>
#include <Cellerator/compute/operation/edge/gate_update_validation_v1.hh>
#include <Cellerator/compute/operation/relation_bundle/catalog.hh>
#include <Cellerator/compute/operation/relation_chain/hierarchy.hh>
#include <Cellerator/compute/candidate/segment/portfolio_v2.hh>
#include <Cellerator/compute/operation/fusion/fusion_validation_v1.hh>
#include <Cellerator/compute/operation/fusion/prepared_stage_graph_v1.hh>
#include <Cellerator/compute/training/v2/interface.hh>

#include <cstdint>

namespace cellerator::compute {

struct ce_exop_operation_portfolio_v1 {
    std::uint64_t relation_apply_candidates = 0u;
    std::uint64_t residual_candidates = 0u;
    std::uint64_t contraction_candidates = 0u;
    std::uint64_t transpose_candidates = 0u;
    std::uint64_t gate_and_update_candidates = 0u;
    std::uint64_t bundle_and_chain_candidates = 0u;
    std::uint64_t segment_candidates = 0u;
    std::uint64_t fusion_candidates = 0u;
    std::uint64_t training_stage_kinds = 0u;
    bool all_candidates_planner_owned = false;
    bool all_experimental_candidates_require_measurement = false;
};

ce_exop_operation_portfolio_v1 query_ce_exop_operation_portfolio_v1() noexcept;

} // namespace cellerator::compute
