#pragma once

#include <Cellerator/compiler/ir/semantic/implement_relation_ir_types_v1.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

#include <cstdint>

namespace Cellerator::compiler::ir::semantic {

enum relation_apply_effect_ir_v1 : std::uint32_t {
    relation_apply_reads_source_v1 = 1u << 0,
    relation_apply_reads_values_v1 = 1u << 1,
    relation_apply_writes_result_v1 = 1u << 2,
    relation_apply_advances_result_generation_v1 = 1u << 3,
};

struct relation_apply_operation_ir_v1 {
    semantic_identity_v1 identity{};
    relation_ir_type_v1 relation;
    state_ir_type_v1 source;
    state_ir_type_v1 result;
    cellerator::compute::operation::v2::destination_update update =
        cellerator::compute::operation::v2::destination_update::overwrite;
    bool deterministic = true;
    std::uint32_t effects = relation_apply_reads_source_v1 |
        relation_apply_reads_values_v1 | relation_apply_writes_result_v1 |
        relation_apply_advances_result_generation_v1;
};

enum class relation_apply_ir_validation_code_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_relation,
    invalid_source,
    invalid_result,
    axis_mismatch,
    width_mismatch,
    numeric_mismatch,
    invalid_update,
    invalid_effects,
};

struct lowered_relation_apply_v1 {
    cellerator::compute::operation::v2::typed_relation relation{};
    cellerator::compute::operation::v2::relation_binding_contract binding{};
    cellerator::compute::operation::v2::relation_value_binding_contract value_binding{};
    cellerator::compute::operation::v2::operation_problem operation{};
    cellerator::compute::operation::v2::relation_algebra_problem algebra{};

    lowered_relation_apply_v1() noexcept;
    lowered_relation_apply_v1(const lowered_relation_apply_v1& other) noexcept;
    lowered_relation_apply_v1& operator=(const lowered_relation_apply_v1& other) noexcept;
    lowered_relation_apply_v1(lowered_relation_apply_v1&& other) noexcept;
    lowered_relation_apply_v1& operator=(lowered_relation_apply_v1&& other) noexcept;
    void refresh_views() noexcept;
};

[[nodiscard]] relation_apply_ir_validation_code_v1
validate_relation_apply_operation_ir_v1(const relation_apply_operation_ir_v1& operation) noexcept;

[[nodiscard]] relation_apply_ir_validation_code_v1
lower_relation_apply_operation_v1(
    const relation_apply_operation_ir_v1& operation,
    lowered_relation_apply_v1* lowered) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
