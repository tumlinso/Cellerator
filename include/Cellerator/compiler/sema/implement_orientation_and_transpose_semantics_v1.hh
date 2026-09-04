#pragma once

#include <Cellerator/compiler/sema/implement_relation_endpoint_semantics_v1.hh>

namespace cellerator::compiler::sema::v1 {

struct oriented_relation_operation {
    const axis_type *input_axis = nullptr;
    const axis_type *output_axis = nullptr;
    semantic_identity logical_edge_identity{};
    relation_orientation orientation = relation_orientation::forward;
    compute::operation::v2::operation_kind runtime_kind =
        compute::operation::v2::operation_kind::relation_apply;
};

oriented_relation_operation infer_oriented_operation(
    const relation_endpoint_semantics &relation,
    relation_orientation orientation) noexcept;
bool orientation_agrees_with_runtime(
    const oriented_relation_operation &operation,
    compute::operation::v2::relation_orientation orientation,
    compute::operation::v2::operation_kind kind) noexcept;

}  // namespace cellerator::compiler::sema::v1
