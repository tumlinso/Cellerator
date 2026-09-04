#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/composition.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

enum class edge_transform_kind_ir_v1 : std::uint8_t {
    map_affine = 1,
    multiplicative_gate,
    predicate_gate,
    support_mask,
};

struct edge_transform_operation_ir_v1 {
    semantic_identity_v1 identity{};
    edge_transform_kind_ir_v1 kind = edge_transform_kind_ir_v1::map_affine;
    semantic_identity_v1 logical_edge_identity{};
    semantic_identity_v1 logical_edge_order{};
    std::uint64_t logical_edge_count = 0;
    std::uint64_t consumed_support_generation = 0;
    std::uint64_t produced_support_generation = 0;
    double scale = 1.0;
    double bias = 0.0;
    bool projection_independent = true;
};

struct sparse_axis_update_operation_ir_v1 {
    semantic_identity_v1 identity{};
    axis_ir_type_v1 target_axis;
    cellerator::compute::operation::v2::sparse_update_operation update =
        cellerator::compute::operation::v2::sparse_update_operation::assign;
    bool indices_unique = false;
    bool indices_in_persistent_order = false;
    bool preserve_canonical_identity = true;
    bool input_output_aliasing_legal = false;
};

enum class edge_sparse_operation_status_ir_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_edge_contract,
    invalid_generation,
    projection_dependent,
    invalid_axis,
    invalid_input,
    duplicate_index,
};

[[nodiscard]] edge_sparse_operation_status_ir_v1
apply_edge_transform_ir_v1(
    const edge_transform_operation_ir_v1& operation,
    const std::vector<double>& values,
    const std::vector<double>& gates,
    const std::vector<std::uint8_t>& support,
    std::vector<double>* result) noexcept;

[[nodiscard]] edge_sparse_operation_status_ir_v1
apply_sparse_axis_update_ir_v1(
    const sparse_axis_update_operation_ir_v1& operation,
    const std::vector<std::uint64_t>& indices,
    const std::vector<double>& updates,
    std::vector<double>* target) noexcept;

[[nodiscard]] cellerator::compute::operation::v2::edge_operation
lower_edge_transform_kind_ir_v1(edge_transform_kind_ir_v1 kind) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
