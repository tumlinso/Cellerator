#include <Cellerator/compiler/ir/semantic/implement_edge_map_gate_support_mask_and_sparse_update_o_v1.hh>

#include <algorithm>
#include <unordered_set>

namespace Cellerator::compiler::ir::semantic {

edge_sparse_operation_status_ir_v1 apply_edge_transform_ir_v1(
    const edge_transform_operation_ir_v1& operation,
    const std::vector<double>& values,
    const std::vector<double>& gates,
    const std::vector<std::uint8_t>& support,
    std::vector<double>* result) noexcept {
    if (!operation.identity.valid() || !operation.logical_edge_identity.valid() ||
        !operation.logical_edge_order.valid())
        return edge_sparse_operation_status_ir_v1::invalid_identity;
    if (!operation.projection_independent)
        return edge_sparse_operation_status_ir_v1::projection_dependent;
    if (result == nullptr || values.size() != operation.logical_edge_count)
        return edge_sparse_operation_status_ir_v1::invalid_input;
    if ((operation.kind == edge_transform_kind_ir_v1::multiplicative_gate ||
         operation.kind == edge_transform_kind_ir_v1::predicate_gate) &&
        gates.size() != values.size()) return edge_sparse_operation_status_ir_v1::invalid_input;
    if (operation.kind == edge_transform_kind_ir_v1::support_mask) {
        if (support.size() != values.size()) return edge_sparse_operation_status_ir_v1::invalid_input;
        if (operation.consumed_support_generation == 0 ||
            operation.produced_support_generation <= operation.consumed_support_generation)
            return edge_sparse_operation_status_ir_v1::invalid_generation;
    }
    result->resize(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        switch (operation.kind) {
        case edge_transform_kind_ir_v1::map_affine:
            (*result)[index] = operation.scale * values[index] + operation.bias;
            break;
        case edge_transform_kind_ir_v1::multiplicative_gate:
            (*result)[index] = values[index] * gates[index];
            break;
        case edge_transform_kind_ir_v1::predicate_gate:
            (*result)[index] = gates[index] != 0.0 ? values[index] : 0.0;
            break;
        case edge_transform_kind_ir_v1::support_mask:
            (*result)[index] = support[index] != 0 ? values[index] : 0.0;
            break;
        }
    }
    return edge_sparse_operation_status_ir_v1::success;
}

edge_sparse_operation_status_ir_v1 apply_sparse_axis_update_ir_v1(
    const sparse_axis_update_operation_ir_v1& operation,
    const std::vector<std::uint64_t>& indices,
    const std::vector<double>& updates,
    std::vector<double>* target) noexcept {
    if (!operation.identity.valid()) return edge_sparse_operation_status_ir_v1::invalid_identity;
    if (validate_axis_ir_type_v1(operation.target_axis) != axis_ir_validation_code_v1::success)
        return edge_sparse_operation_status_ir_v1::invalid_axis;
    if (target == nullptr || indices.size() != updates.size())
        return edge_sparse_operation_status_ir_v1::invalid_input;
    if (operation.indices_unique) {
        std::unordered_set<std::uint64_t> seen;
        for (const auto index : indices)
            if (!seen.insert(index).second)
                return edge_sparse_operation_status_ir_v1::duplicate_index;
    }
    for (std::size_t item = 0; item < indices.size(); ++item) {
        if (indices[item] >= target->size()) return edge_sparse_operation_status_ir_v1::invalid_input;
        auto& value = (*target)[indices[item]];
        switch (operation.update) {
        case cellerator::compute::operation::v2::sparse_update_operation::assign:
            value = updates[item]; break;
        case cellerator::compute::operation::v2::sparse_update_operation::add:
            value += updates[item]; break;
        case cellerator::compute::operation::v2::sparse_update_operation::subtract:
            value -= updates[item]; break;
        case cellerator::compute::operation::v2::sparse_update_operation::multiply:
            value *= updates[item]; break;
        case cellerator::compute::operation::v2::sparse_update_operation::maximum:
            value = std::max(value, updates[item]); break;
        }
    }
    return edge_sparse_operation_status_ir_v1::success;
}

cellerator::compute::operation::v2::edge_operation
lower_edge_transform_kind_ir_v1(edge_transform_kind_ir_v1 kind) noexcept {
    using result = cellerator::compute::operation::v2::edge_operation;
    switch (kind) {
    case edge_transform_kind_ir_v1::map_affine: return result::arbitrary_map;
    case edge_transform_kind_ir_v1::multiplicative_gate: return result::multiplicative_gate;
    case edge_transform_kind_ir_v1::predicate_gate: return result::predicate_gate;
    case edge_transform_kind_ir_v1::support_mask: return result::active_support_mask;
    }
    return result::none;
}

}  // namespace Cellerator::compiler::ir::semantic
