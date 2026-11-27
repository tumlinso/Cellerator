#include "compute_candidate_inventory.hh"

#include <Cellerator/ops/relation/cuda/sm70/contract/contract_catalog_v1.hh>
#include <Cellerator/ops/relation/cuda/sm70/relation_apply/apply_inventory_v1.hh>
#include <Cellerator/ops/relation/cuda/sm70/residual/portfolio_v1.h>
#include <Cellerator/ops/relation/cuda/sm70/transpose/transpose_candidates_v1.hh>
#include <Cellerator/ops/segment/portfolio_v2.hh>
#include <Cellerator/ops/condition/edge/gate_update_validation_v1.hh>
#include <Cellerator/ops/relation/fusion/fusion_validation_v1.hh>
#include <Cellerator/ops/relation/bundle/catalog.hh>

#include <cstddef>

namespace cellerator::execution::planning {

compute_candidate_inventory_v1 query_compute_candidate_inventory_v1() noexcept {
    namespace apply = ::cellerator::execution::planning::architecture::nvidia::sm70::relation_apply;
    namespace residual = ::cellerator::execution::planning::architecture::providers::nvidia::sm70::residual;
    namespace contract = ::cellerator::execution::planning::architecture::providers::nvidia::sm70::contract;
    namespace transpose = ::cellerator::execution::planning::architecture::providers::nvidia::sm70::transpose;
    namespace gate = ops::relation::edge;
    namespace fusion = ops::relation::fusion;

    const apply::sm70_apply_inventory_v1 apply_inventory =
        apply::built_in_sm70_apply_inventory_v1();
    const residual::residual_portfolio_view_v1 residual_inventory =
        residual::residual_portfolio_v1();
    std::size_t contraction_count = 0u;
    const contract::catalog_entry_v1 *contraction_inventory =
        contract::catalog_v1(&contraction_count);
    const transpose::transpose_candidate_catalog_v1 transpose_inventory =
        transpose::query_transpose_candidates_v1();
    std::size_t gate_count = 0u;
    const gate::registry_entry_v1 *gate_inventory = gate::registry_v1(&gate_count);
    std::size_t fusion_count = 0u;
    const fusion::registry_entry_v1 *fusion_inventory =
        fusion::fusion_registry_v1(&fusion_count);

    bool measurement_contract = true;
    for (std::size_t index = 0; index < contraction_count; ++index) {
        measurement_contract = measurement_contract
            && (!contraction_inventory[index].promoted)
            && contraction_inventory[index].requires_measurement;
    }
    for (std::uint64_t index = 0u; index < transpose_inventory.candidate_count; ++index) {
        const transpose::transpose_candidate_v1 &candidate =
            transpose_inventory.candidates[index];
        measurement_contract = measurement_contract
            && (!candidate.experimental || candidate.requires_measurement);
    }
    for (std::size_t index = 0u; index < gate_count; ++index) {
        measurement_contract = measurement_contract
            && !gate_inventory[index].promoted
            && gate_inventory[index].requires_measurement;
    }
    for (std::size_t index = 0u; index < fusion_count; ++index) {
        measurement_contract = measurement_contract
            && fusion_inventory[index].experimental
            && fusion_inventory[index].requires_measurement
            && !fusion_inventory[index].auto_promoted
            && fusion_inventory[index].unfused_stages_available;
    }

    return {apply_inventory.candidate_count,
            residual_inventory.candidate_count,
            static_cast<std::uint64_t>(contraction_count),
            transpose_inventory.candidate_count,
            static_cast<std::uint64_t>(gate_count),
            static_cast<std::uint64_t>(ops::relation::bundle::candidate_count),
            static_cast<std::uint64_t>(ops::segment::segment_candidate_count_v2()),
            static_cast<std::uint64_t>(fusion_count),
            6u,
            true,
            measurement_contract};
}

} // namespace cellerator::execution::planning
