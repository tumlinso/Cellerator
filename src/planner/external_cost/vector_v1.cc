#include <Cellerator/planner/external_cost/vector_v1.hh>

#include <cmath>

namespace cellerator::planner::external_cost {

external_cost_vector_status_v1 validate_external_cost_vector_v1(
    const external_cost_vector_v1 &cost) noexcept {
    using status = external_cost_vector_status_v1;
    if (cost.schema_version != external_cost_vector_schema_v1)
        return status::unsupported_schema;
    if (cost.record_bytes != sizeof(external_cost_vector_v1))
        return status::invalid_record_bytes;
    if (cost.contract_id == 0u)
        return status::invalid_contract;
    if (cost.pricing_epoch == 0u)
        return status::invalid_pricing_epoch;
    const double components[] = {cost.fixed_ns, cost.persistent_byte_ns,
        cost.transient_byte_ns, cost.transfer_byte_ns,
        cost.communication_byte_ns, cost.launch_ns, cost.synchronization_ns,
        cost.reuse_credit_ns};
    for (double value : components) {
        if (!std::isfinite(value) || value < 0.0)
            return status::invalid_component;
    }
    if (cost.expected_reuse == 0u)
        return status::invalid_reuse;
    return status::valid;
}

} // namespace cellerator::planner::external_cost
