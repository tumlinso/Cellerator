#include <Cellerator/compiler/planning/represent_partial_result_trees_and_algebra_selection_v1.hh>

#include <algorithm>
#include <limits>
#include <vector>

namespace Cellerator::compiler::planning {
namespace {

constexpr bool power_of_two(std::uint64_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

double merge(partial_algebra_kind_v1 kind, double left, double right) noexcept {
    switch (kind) {
    case partial_algebra_kind_v1::sum: return left + right;
    case partial_algebra_kind_v1::product: return left * right;
    case partial_algebra_kind_v1::minimum: return std::min(left, right);
    case partial_algebra_kind_v1::maximum: return std::max(left, right);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

partial_result_plan_validation_code_v1 validate_partial_result_plan_v1(
    const partial_result_plan_v1& plan,
    std::uint32_t expected_leaf_count) noexcept {
    const auto properties = plan.algebra.properties;
    if (plan.algebra.algebra_identity == 0u || plan.algebra.state_bytes == 0u ||
        !power_of_two(plan.algebra.state_alignment) ||
        plan.algebra.merge_operation_identity == 0u ||
        plan.algebra.finalize_operation_identity == 0u ||
        (properties & associative_algebra_v1) == 0u) {
        return partial_result_plan_validation_code_v1::invalid_algebra;
    }
    if (plan.output_order_identity == 0u ||
        plan.output_order_identity != plan.required_output_order_identity) {
        return partial_result_plan_validation_code_v1::invalid_order;
    }
    if (plan.nodes.empty() || plan.root >= plan.nodes.size() || expected_leaf_count == 0u) {
        return partial_result_plan_validation_code_v1::invalid_tree;
    }
    if (plan.algebra.state_bytes > plan.workspace_limit_bytes ||
        plan.nodes.size() > plan.workspace_limit_bytes / plan.algebra.state_bytes) {
        return partial_result_plan_validation_code_v1::resource_limit_exceeded;
    }
    if ((plan.deterministic_required ||
         (properties & deterministic_tree_algebra_v1) != 0u) &&
        plan.deterministic_tree_identity == 0u) {
        return partial_result_plan_validation_code_v1::nondeterministic_tree;
    }

    std::vector<bool> leaves(expected_leaf_count, false);
    for (std::uint32_t index = 0u; index < plan.nodes.size(); ++index) {
        const auto& node = plan.nodes[index];
        if (node.node_identity == 0u) return partial_result_plan_validation_code_v1::invalid_tree;
        const bool leaf = node.leaf_input != UINT32_MAX;
        if (leaf) {
            if (node.left != UINT32_MAX || node.right != UINT32_MAX ||
                node.leaf_input >= expected_leaf_count) {
                return partial_result_plan_validation_code_v1::invalid_tree;
            }
            if (leaves[node.leaf_input]) {
                return partial_result_plan_validation_code_v1::duplicate_leaf;
            }
            leaves[node.leaf_input] = true;
        } else if (node.left >= index || node.right >= index || node.left == node.right) {
            return partial_result_plan_validation_code_v1::invalid_tree;
        } else if ((properties & commutative_algebra_v1) == 0u && node.left > node.right) {
            return partial_result_plan_validation_code_v1::invalid_tree;
        }
    }
    return std::all_of(leaves.begin(), leaves.end(), [](bool present) { return present; })
        ? partial_result_plan_validation_code_v1::ok
        : partial_result_plan_validation_code_v1::incomplete_leaf_coverage;
}

bool reconstruct_partial_results_v1(
    const partial_result_plan_v1& plan,
    const std::vector<double>& partials,
    double* output) noexcept {
    if (output == nullptr ||
        validate_partial_result_plan_v1(plan, static_cast<std::uint32_t>(partials.size())) !=
            partial_result_plan_validation_code_v1::ok) {
        return false;
    }
    std::vector<double> values(plan.nodes.size(), 0.0);
    for (std::uint32_t index = 0u; index < plan.nodes.size(); ++index) {
        const auto& node = plan.nodes[index];
        values[index] = node.leaf_input != UINT32_MAX
            ? partials[node.leaf_input]
            : merge(plan.algebra.kind, values[node.left], values[node.right]);
    }
    *output = values[plan.root];
    return true;
}

}  // namespace Cellerator::compiler::planning
