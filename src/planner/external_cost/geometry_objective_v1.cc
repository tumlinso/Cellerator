#include <Cellerator/planner/external_cost/geometry_objective_v1.hh>

#include <algorithm>
#include <cmath>

namespace cellerator::planner::external_cost {

geometry_objective_status_v1 price_geometry_objective_v1(
    const geometry_objective_terms_v1 &terms,
    const external_cost_vector_v1 &cost,
    priced_geometry_objective_v1 *output) noexcept {
    using status = geometry_objective_status_v1;
    if (output == nullptr)
        return status::null_output;
    *output = {};
    if (validate_external_cost_vector_v1(cost)
        != external_cost_vector_status_v1::valid)
        return status::invalid_external_cost;
    if (!std::isfinite(terms.local_objective_ns)
        || terms.local_objective_ns < 0.0
        || !std::isfinite(terms.construction_ns)
        || terms.construction_ns < 0.0
        || terms.global_expected_reuse == 0u)
        return status::invalid_terms;
    const double reuse = static_cast<double>(terms.global_expected_reuse);
    const double amortized = (terms.construction_ns + cost.fixed_ns
        + static_cast<double>(terms.persistent_bytes)
            * cost.persistent_byte_ns) / reuse;
    const double movement = (static_cast<double>(terms.input_movement_bytes)
        + static_cast<double>(terms.output_movement_bytes))
        * cost.transfer_byte_ns;
    const double communication = static_cast<double>(
        terms.communication_bytes) * cost.communication_byte_ns;
    const double before_credit = terms.local_objective_ns + amortized
        + movement + communication;
    if (!std::isfinite(before_credit))
        return status::arithmetic_overflow;
    const double credit = std::min(cost.reuse_credit_ns, before_credit);
    *output = {terms.local_objective_ns, amortized, movement, communication,
        credit, before_credit - credit};
    return status::success;
}

} // namespace cellerator::planner::external_cost
