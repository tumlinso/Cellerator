#include <Cellerator/compute/compute.hh>

#include <cstddef>

namespace cellerator::compute {

ce_exop_operation_portfolio_v1 query_ce_exop_operation_portfolio_v1() noexcept {
    namespace apply = architecture::nvidia::sm70::relation_apply;
    namespace residual = architecture::providers::nvidia::sm70::residual;
    namespace contract = architecture::providers::nvidia::sm70::contract;
    namespace transpose = architecture::providers::nvidia::sm70::transpose;

    const apply::sm70_apply_inventory_v1 apply_inventory =
        apply::built_in_sm70_apply_inventory_v1();
    const residual::residual_portfolio_view_v1 residual_inventory =
        residual::residual_portfolio_v1();
    std::size_t contraction_count = 0u;
    const contract::catalog_entry_v1 *contraction_inventory =
        contract::catalog_v1(&contraction_count);
    const transpose::transpose_candidate_catalog_v1 transpose_inventory =
        transpose::query_transpose_candidates_v1();

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

    return {apply_inventory.candidate_count,
            residual_inventory.candidate_count,
            static_cast<std::uint64_t>(contraction_count),
            transpose_inventory.candidate_count,
            true,
            measurement_contract};
}

} // namespace cellerator::compute
