#include <Cellerator/planner/external_cost/complete_cost_v1.hh>

#include <algorithm>
#include <cmath>

namespace cellerator::planner::external_cost {

external_complete_cost_status_v1 inject_external_complete_cost_v1(
    double local_complete_ns,
    const local_cost_resource_vector_v1 &resources,
    const external_cost_vector_v1 &cost,
    external_complete_cost_v1 *output) noexcept {
    using status = external_complete_cost_status_v1;
    if (output == nullptr)
        return status::null_output;
    *output = {};
    if (validate_external_cost_vector_v1(cost)
        != external_cost_vector_status_v1::valid)
        return status::invalid_external_cost;
    if (!std::isfinite(local_complete_ns) || local_complete_ns < 0.0)
        return status::invalid_local_cost;
    const double reuse = static_cast<double>(cost.expected_reuse);
    const double charge = cost.fixed_ns / reuse
        + static_cast<double>(resources.persistent_bytes)
            * cost.persistent_byte_ns / reuse
        + static_cast<double>(resources.transient_bytes)
            * cost.transient_byte_ns
        + static_cast<double>(resources.transfer_bytes)
            * cost.transfer_byte_ns
        + static_cast<double>(resources.communication_bytes)
            * cost.communication_byte_ns
        + static_cast<double>(resources.launch_count) * cost.launch_ns
        + cost.synchronization_ns;
    if (!std::isfinite(charge))
        return status::arithmetic_overflow;
    const double before_credit = local_complete_ns + charge;
    if (!std::isfinite(before_credit))
        return status::arithmetic_overflow;
    const double credit = std::min(cost.reuse_credit_ns, before_credit);
    *output = {local_complete_ns, charge, credit, before_credit - credit};
    return status::success;
}

} // namespace cellerator::planner::external_cost
