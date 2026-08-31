#include <Cellerator/geometry/geometry.hh>

namespace cellerator::geometry::compiler {

optimizer_portfolio_readiness_v1
validate_integrated_optimizer_portfolio_v1() noexcept {
    optimizer_portfolio_readiness_v1 result{};
    const auto contract = optimizer::oracle::built_in_optimizer_portfolio_v1();
    const auto validation =
            optimizer::oracle::validate_optimizer_portfolio_contract_v1(contract);
    if (validation.status != optimizer::oracle::optimizer_portfolio_status::success
        || !validation.deterministic_registry_order
        || !validation.no_promoted_strategy) {
        return result;
    }

    const auto disposition =
            optimizer::device::built_in_device_assisted_disposition_v1();
    if (!optimizer::device::validate_device_assisted_disposition_v1(disposition)) {
        return result;
    }
    result.contract_fingerprint = validation.contract_fingerprint;
    result.validated_strategies = validation.validated_strategies;
    result.device_assisted_available = true;
    result.device_assisted_experimental = disposition.requires_measurement
            && !disposition.production_promoted
            && !disposition.steady_state_allowed;
    return result;
}

}  // namespace cellerator::geometry::compiler
