#include <Cellerator/compiler/ir/planning/implement_partial_result_algebra_nodes_v1.hh>

#include <cassert>

int main() {
    namespace planning = cellerator::compiler::ir::planning::v1;
    namespace decomp = cellerator::compute::decomposition;
    namespace execution = cellerator::execution;
    decomp::partial_result_algebra_v1 algebra{};
    algebra.algebra_identity = {1u, 1u};
    algebra.state_layout_identity = {1u, 2u};
    algebra.neutral_element_identity = {1u, 3u};
    algebra.merge_operation_identity = {1u, 4u};
    algebra.finalize_operation_identity = {1u, 5u};
    algebra.state_bytes = 16u;
    algebra.state_alignment = 16u;
    algebra.flags = decomp::associative_v1 | decomp::commutative_v1 |
                    decomp::deterministic_tree_required_v1;
    algebra.deterministic_tree_identity = {2u, 1u};
    algebra.numerical.relation_storage = execution::numeric_type::f16;
    algebra.numerical.state_storage = execution::numeric_type::f32;
    algebra.numerical.multiply = execution::numeric_type::f32;
    algebra.numerical.accumulation = execution::numeric_type::f32;
    algebra.numerical.output_storage = execution::numeric_type::f32;
    algebra.numerical.scalar = execution::numeric_type::f32;
    const planning::merge_tree_edge_v1 tree[] = {{0u, 1u, 4u, 0u},
                                                  {2u, 3u, 5u, 0u},
                                                  {4u, 5u, 6u, 0u}};
    planning::partial_result_algebra_node_v1 node{{3u, 1u}, algebra, tree, 3u, 4u};
    assert(planning::validate_partial_result_algebra_node_v1(node) ==
           planning::partial_algebra_node_status_v1::ok);
    node.reference_tree_edge_count = 2u;
    assert(planning::validate_partial_result_algebra_node_v1(node) ==
           planning::partial_algebra_node_status_v1::missing_tree);
    node.reference_tree_edge_count = 3u;
    const planning::merge_tree_edge_v1 invalid[] = {{0u, 0u, 4u, 0u},
                                                     {2u, 3u, 5u, 0u},
                                                     {4u, 5u, 6u, 0u}};
    node.reference_tree = invalid;
    assert(planning::validate_partial_result_algebra_node_v1(node) ==
           planning::partial_algebra_node_status_v1::invalid_tree_edge);
}
