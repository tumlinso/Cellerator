#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_state_and_value_plane_ir_types_v1.hh>

#include <cstdint>
#include <optional>

namespace Cellerator::compiler::ir::semantic {

enum class relation_orientation_ir_v1 : std::uint8_t {
    forward = 1,
    transpose,
};

enum class relation_mutation_policy_v1 : std::uint8_t {
    immutable_structure_mutable_values = 1,
    immutable_structure_and_values,
};

struct relation_ir_type_v1 {
    axis_ir_type_v1 source_axis;
    axis_ir_type_v1 destination_axis;
    semantic_identity_v1 structure_identity{};
    std::uint64_t structure_epoch = 0;
    semantic_identity_v1 logical_edge_identity{};
    semantic_identity_v1 logical_edge_order{};
    std::uint64_t logical_edge_count = 0;
    semantic_identity_v1 support_identity{};
    semantic_identity_v1 value_plane_identity{};
    std::uint64_t value_generation = 0;
    std::uint64_t active_support_generation = 0;
    relation_orientation_ir_v1 orientation = relation_orientation_ir_v1::forward;
    relation_mutation_policy_v1 mutation =
        relation_mutation_policy_v1::immutable_structure_mutable_values;
};

struct relation_ir_binding_v1 {
    semantic_identity_v1 logical_edge_identity{};
    semantic_identity_v1 support_identity{};
    semantic_identity_v1 value_plane_identity{};
    std::uint64_t value_generation = 0;
    std::uint64_t active_support_generation = 0;
    relation_orientation_ir_v1 orientation = relation_orientation_ir_v1::forward;
    relation_mutation_policy_v1 mutation =
        relation_mutation_policy_v1::immutable_structure_mutable_values;
};

enum class relation_ir_validation_code_v1 : std::uint8_t {
    success = 0,
    invalid_source_axis,
    invalid_destination_axis,
    invalid_structure,
    invalid_logical_edges,
    invalid_support,
    invalid_value_plane,
    invalid_generation,
    invalid_orientation,
    invalid_mutation_policy,
};

[[nodiscard]] relation_ir_validation_code_v1
validate_relation_ir_type_v1(const relation_ir_type_v1& relation) noexcept;

[[nodiscard]] std::optional<relation_ir_type_v1>
relation_ir_from_typed_relation_v1(
    const cellerator::compute::operation::v2::typed_relation& relation,
    axis_ir_type_v1 source_axis,
    axis_ir_type_v1 destination_axis,
    relation_ir_binding_v1 binding) noexcept;

[[nodiscard]] std::optional<cellerator::compute::operation::v2::typed_relation>
typed_relation_from_relation_ir_v1(const relation_ir_type_v1& relation) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
