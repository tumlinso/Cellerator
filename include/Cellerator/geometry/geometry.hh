#pragma once

#include <Cellerator/geometry/optimizer/device/device_assisted_disposition.h>
#include <Cellerator/geometry/optimizer/portfolio_v1.hh>

#include <cstdint>

namespace cellerator::geometry::compiler {

// Cold integration receipt only. It proves that the frozen optimizer
// availability contract and the non-promoted device disposition are linked
// and valid; it does not select a strategy or move planner authority.
struct optimizer_portfolio_readiness_v1 {
    std::uint64_t contract_fingerprint = 0;
    std::uint32_t validated_strategies = 0;
    bool device_assisted_available = false;
    bool device_assisted_experimental = false;
};

optimizer_portfolio_readiness_v1
validate_integrated_optimizer_portfolio_v1() noexcept;

}  // namespace cellerator::geometry::compiler
