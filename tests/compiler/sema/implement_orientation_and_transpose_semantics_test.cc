#include <Cellerator/compiler/sema/implement_orientation_and_transpose_semantics_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    relation_endpoint_semantics relation{};
    relation.source.domain.identity = {1, 2};
    relation.destination.domain.identity = {3, 4};
    relation.logical_edge_identity = {5, 6};
    const auto forward = infer_oriented_operation(relation, relation_orientation::forward);
    const auto transpose = infer_oriented_operation(relation, relation_orientation::transpose);
    assert(forward.input_axis == &relation.source && forward.output_axis == &relation.destination);
    assert(transpose.input_axis == &relation.destination && transpose.output_axis == &relation.source);
    assert(transpose.logical_edge_identity.low == forward.logical_edge_identity.low);
    assert(orientation_agrees_with_runtime(transpose,
        cellerator::compute::operation::v2::relation_orientation::transpose,
        cellerator::compute::operation::v2::operation_kind::relation_apply_transpose));
    assert(!orientation_agrees_with_runtime(transpose,
        cellerator::compute::operation::v2::relation_orientation::forward,
        cellerator::compute::operation::v2::operation_kind::relation_apply));
}
