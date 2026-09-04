#include <Cellerator/compiler/sema/implement_orientation_and_transpose_semantics_v1.hh>

namespace cellerator::compiler::sema::v1 {

oriented_relation_operation infer_oriented_operation(
    const relation_endpoint_semantics &relation,
    relation_orientation orientation) noexcept {
    if (orientation == relation_orientation::forward) {
        return {&relation.source, &relation.destination,
                relation.logical_edge_identity, orientation,
                compute::operation::v2::operation_kind::relation_apply};
    }
    return {&relation.destination, &relation.source,
            relation.logical_edge_identity, orientation,
            compute::operation::v2::operation_kind::relation_apply_transpose};
}

bool orientation_agrees_with_runtime(
    const oriented_relation_operation &operation,
    compute::operation::v2::relation_orientation orientation,
    compute::operation::v2::operation_kind kind) noexcept {
    const bool forward = operation.orientation == relation_orientation::forward;
    return operation.input_axis != nullptr && operation.output_axis != nullptr
        && operation.logical_edge_identity.low != 0
        && operation.runtime_kind == kind
        && ((forward
             && orientation == compute::operation::v2::relation_orientation::forward
             && kind == compute::operation::v2::operation_kind::relation_apply)
            || (!forward
             && orientation == compute::operation::v2::relation_orientation::transpose
             && kind == compute::operation::v2::operation_kind::relation_apply_transpose));
}

}  // namespace cellerator::compiler::sema::v1
