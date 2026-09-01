#include <Cellerator/planner/external_cost/complete_cost_v1.hh>

#include <cassert>

namespace external = cellerator::planner::external_cost;

int main() {
    external::external_cost_vector_v1 cost{};
    cost.contract_id = 1u;
    cost.pricing_epoch = 1u;
    cost.fixed_ns = 20.0;
    cost.persistent_byte_ns = 2.0;
    cost.transient_byte_ns = 3.0;
    cost.transfer_byte_ns = 4.0;
    cost.communication_byte_ns = 5.0;
    cost.launch_ns = 6.0;
    cost.synchronization_ns = 7.0;
    cost.reuse_credit_ns = 8.0;
    cost.expected_reuse = 2u;
    const external::local_cost_resource_vector_v1 resources{
        10u, 2u, 3u, 4u, 2u};
    external::external_complete_cost_v1 result{};
    assert(external::inject_external_complete_cost_v1(
        100.0, resources, cost, &result)
        == external::external_complete_cost_status_v1::success);
    assert(result.external_charge_ns == 77.0);
    assert(result.applied_reuse_credit_ns == 8.0);
    assert(result.complete_ns == 169.0);

    cost.reuse_credit_ns = 1000.0;
    assert(external::inject_external_complete_cost_v1(
        1.0, {}, cost, &result)
        == external::external_complete_cost_status_v1::success);
    assert(result.complete_ns == 0.0);
}
