#include <Cellerator/compiler/planning/represent_partial_result_trees_and_algebra_selection_v1.hh>

#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

namespace planning = Cellerator::compiler::planning;

int main() {
    planning::partial_result_plan_v1 plan{};
    plan.algebra = {1u, planning::partial_algebra_kind_v1::sum,
                    planning::associative_algebra_v1 |
                        planning::commutative_algebra_v1 |
                        planning::deterministic_tree_algebra_v1,
                    sizeof(double), alignof(double), 2u, 3u};
    plan.output_order_identity = 9u;
    plan.required_output_order_identity = 9u;
    plan.deterministic_tree_identity = 10u;
    plan.workspace_limit_bytes = 7u * sizeof(double);
    plan.deterministic_required = true;
    plan.nodes = {
        {1u, UINT32_MAX, UINT32_MAX, 0u},
        {2u, UINT32_MAX, UINT32_MAX, 1u},
        {3u, UINT32_MAX, UINT32_MAX, 2u},
        {4u, UINT32_MAX, UINT32_MAX, 3u},
        {5u, 0u, 1u, UINT32_MAX},
        {6u, 2u, 3u, UINT32_MAX},
        {7u, 4u, 5u, UINT32_MAX},
    };
    plan.root = 6u;
    assert(planning::validate_partial_result_plan_v1(plan, 4u) ==
           planning::partial_result_plan_validation_code_v1::ok);

    const std::vector<double> partials{1.25, -0.5, 3.0, 8.25};
    double reconstructed = 0.0;
    assert(planning::reconstruct_partial_results_v1(plan, partials, &reconstructed));
    const double unsplit = std::accumulate(partials.begin(), partials.end(), 0.0);
    assert(std::abs(reconstructed - unsplit) < 1.0e-12);

    auto duplicate = plan;
    duplicate.nodes[1].leaf_input = 0u;
    assert(planning::validate_partial_result_plan_v1(duplicate, 4u) ==
           planning::partial_result_plan_validation_code_v1::duplicate_leaf);

    auto insufficient = plan;
    insufficient.workspace_limit_bytes = sizeof(double);
    assert(planning::validate_partial_result_plan_v1(insufficient, 4u) ==
           planning::partial_result_plan_validation_code_v1::resource_limit_exceeded);
}
