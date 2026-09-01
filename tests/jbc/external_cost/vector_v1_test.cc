#include <Cellerator/planner/external_cost/vector_v1.hh>

#include <cassert>
#include <limits>

namespace external = cellerator::planner::external_cost;

int main() {
    external::external_cost_vector_v1 cost{};
    cost.contract_id = 1u;
    cost.pricing_epoch = 2u;
    cost.fixed_ns = 10.0;
    cost.persistent_byte_ns = 0.25;
    cost.expected_reuse = 4u;
    assert(external::validate_external_cost_vector_v1(cost)
        == external::external_cost_vector_status_v1::valid);
    cost.transfer_byte_ns = std::numeric_limits<double>::infinity();
    assert(external::validate_external_cost_vector_v1(cost)
        == external::external_cost_vector_status_v1::invalid_component);
    cost.transfer_byte_ns = 0.0;
    cost.expected_reuse = 0u;
    assert(external::validate_external_cost_vector_v1(cost)
        == external::external_cost_vector_status_v1::invalid_reuse);
}
