#include <Cellerator/compiler/ir/semantic/implement_edge_map_gate_support_mask_and_sparse_update_o_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

int main() {
    edge_transform_operation_ir_v1 edge;
    edge.identity = {1, 2};
    edge.logical_edge_identity = {3, 4};
    edge.logical_edge_order = {5, 6};
    edge.logical_edge_count = 4;
    edge.kind = edge_transform_kind_ir_v1::multiplicative_gate;
    const std::vector<double> values{1, 2, 3, 4};
    const std::vector<double> gates{2, 0, 0.5, 1};
    std::vector<double> result;
    assert(apply_edge_transform_ir_v1(edge, values, gates, {}, &result) ==
           edge_sparse_operation_status_ir_v1::success);
    assert((result == std::vector<double>{2, 0, 1.5, 4}));

    edge.kind = edge_transform_kind_ir_v1::support_mask;
    edge.consumed_support_generation = 7;
    edge.produced_support_generation = 8;
    assert(apply_edge_transform_ir_v1(edge, values, {}, {1, 0, 1, 0}, &result) ==
           edge_sparse_operation_status_ir_v1::success);
    assert((result == std::vector<double>{1, 0, 3, 0}));
    assert(lower_edge_transform_kind_ir_v1(edge.kind) ==
           cellerator::compute::operation::v2::edge_operation::active_support_mask);

    sparse_axis_update_operation_ir_v1 sparse;
    sparse.identity = {9, 10};
    sparse.target_axis.identity = {11, 12};
    sparse.target_axis.domain = {{13, 14}, "gene"};
    sparse.target_axis.order = {{15, 16}, {13, 14}, true};
    sparse.target_axis.geometry = {{17, 18}, {13, 14}};
    sparse.target_axis.partition = {{19, 20}, {13, 14}, {21, 22}};
    sparse.target_axis.extent = {extent_knowledge_kind_v1::exact, 4, 4};
    sparse.update = cellerator::compute::operation::v2::sparse_update_operation::add;
    sparse.indices_unique = true;
    sparse.indices_in_persistent_order = true;
    std::vector<double> target{1, 1, 1, 1};
    assert(apply_sparse_axis_update_ir_v1(sparse, {0, 2}, {3, 4}, &target) ==
           edge_sparse_operation_status_ir_v1::success);
    assert((target == std::vector<double>{4, 1, 5, 1}));
    assert(apply_sparse_axis_update_ir_v1(sparse, {1, 1}, {2, 3}, &target) ==
           edge_sparse_operation_status_ir_v1::duplicate_index);

    std::cout << "logical_edges=4 support_generation=8 sparse_updates=2\n";
}
