#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

struct numeric_tuple_ir_v1 {
    cellerator::execution::numeric_type storage = cellerator::execution::numeric_type::invalid;
    cellerator::execution::numeric_type compute = cellerator::execution::numeric_type::invalid;
    cellerator::execution::numeric_type accumulation = cellerator::execution::numeric_type::invalid;
    cellerator::execution::numeric_type output = cellerator::execution::numeric_type::invalid;
};

enum class value_mutability_v1 : std::uint8_t {
    immutable = 1,
    mutable_values,
};

enum class address_intent_v1 : std::uint8_t {
    unconstrained = 1,
    host,
    device,
    managed,
    peer_device,
};

struct value_generation_ir_v1 {
    std::uint64_t value = 0;
    bool dynamic_at_launch = false;
};

struct alias_contract_ir_v1 {
    std::uint64_t alias_class = 0;
    bool may_alias_input = false;
};

struct state_ir_type_v1 {
    semantic_identity_v1 identity{};
    std::vector<semantic_identity_v1> axes;
    std::uint32_t dense_width = 0;
    numeric_tuple_ir_v1 numeric{};
    semantic_identity_v1 order{};
    value_generation_ir_v1 generation{};
    value_mutability_v1 mutability = value_mutability_v1::immutable;
    address_intent_v1 address_intent = address_intent_v1::unconstrained;
    alias_contract_ir_v1 alias{};
};

struct value_plane_ir_type_v1 {
    semantic_identity_v1 identity{};
    semantic_identity_v1 structure{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t element_count = 0;
    numeric_tuple_ir_v1 numeric{};
    semantic_identity_v1 order{};
    value_generation_ir_v1 generation{};
    value_mutability_v1 mutability = value_mutability_v1::mutable_values;
    address_intent_v1 address_intent = address_intent_v1::unconstrained;
    alias_contract_ir_v1 alias{};
};

enum class state_value_ir_validation_code_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_axis,
    invalid_width,
    invalid_numeric_tuple,
    invalid_order,
    invalid_generation,
    invalid_mutability,
    invalid_address_intent,
    invalid_alias_contract,
    invalid_structure,
};

[[nodiscard]] state_value_ir_validation_code_v1
validate_numeric_tuple_ir_v1(const numeric_tuple_ir_v1& numeric) noexcept;

[[nodiscard]] state_value_ir_validation_code_v1
validate_state_ir_type_v1(const state_ir_type_v1& state) noexcept;

[[nodiscard]] state_value_ir_validation_code_v1
validate_value_plane_ir_type_v1(const value_plane_ir_type_v1& plane) noexcept;

[[nodiscard]] cellerator::compute::operation::v2::numerical_policy
to_operation_numeric_policy_v1(const numeric_tuple_ir_v1& numeric) noexcept;

[[nodiscard]] numeric_tuple_ir_v1 from_operation_numeric_policy_v1(
    const cellerator::compute::operation::v2::numerical_policy& numeric) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
