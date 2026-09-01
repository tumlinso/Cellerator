#include <Cellerator/planner/external_cost/geometry_objective_v1.hh>

#include <cassert>

namespace external = cellerator::planner::external_cost;

int main() {
    external::external_cost_vector_v1 cost{};
    cost.contract_id = 1u;
    cost.pricing_epoch = 1u;
    cost.fixed_ns = 20.0;
    cost.persistent_byte_ns = 2.0;
    cost.transfer_byte_ns = 3.0;
    cost.communication_byte_ns = 4.0;
    cost.reuse_credit_ns = 5.0;
    external::geometry_objective_terms_v1 terms{};
    terms.local_objective_ns = 100.0;
    terms.construction_ns = 40.0;
    terms.persistent_bytes = 10u;
    terms.input_movement_bytes = 3u;
    terms.output_movement_bytes = 2u;
    terms.communication_bytes = 5u;
    terms.global_expected_reuse = 4u;
    external::priced_geometry_objective_v1 output{};
    assert(external::price_geometry_objective_v1(terms, cost, &output)
        == external::geometry_objective_status_v1::success);
    assert(output.amortized_geometry_ns == 20.0);
    assert(output.movement_ns == 15.0);
    assert(output.communication_ns == 20.0);
    assert(output.complete_objective_ns == 150.0);
    terms.global_expected_reuse = 0u;
    assert(external::price_geometry_objective_v1(terms, cost, &output)
        == external::geometry_objective_status_v1::invalid_terms);
}
