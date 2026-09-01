#pragma once

#include <Cellerator/planner/external_cost/vector_v1.hh>

namespace cellerator::planner::external_cost {

struct local_cost_resource_vector_v1 {
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    std::uint64_t transfer_bytes = 0u;
    std::uint64_t communication_bytes = 0u;
    std::uint64_t launch_count = 0u;
};

struct external_complete_cost_v1 {
    double local_complete_ns = 0.0;
    double external_charge_ns = 0.0;
    double applied_reuse_credit_ns = 0.0;
    double complete_ns = 0.0;
};

enum class external_complete_cost_status_v1 : std::uint8_t {
    success = 0u,
    null_output,
    invalid_external_cost,
    invalid_local_cost,
    arithmetic_overflow,
};

external_complete_cost_status_v1 inject_external_complete_cost_v1(
    double local_complete_ns,
    const local_cost_resource_vector_v1 &resources,
    const external_cost_vector_v1 &external_cost,
    external_complete_cost_v1 *output) noexcept;

} // namespace cellerator::planner::external_cost
