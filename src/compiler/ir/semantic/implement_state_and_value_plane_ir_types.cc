#include <Cellerator/compiler/ir/semantic/implement_state_and_value_plane_ir_types_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::ir::semantic {
namespace {

bool valid_numeric(cellerator::execution::numeric_type type) noexcept {
    using cellerator::execution::numeric_type;
    return type >= numeric_type::bit && type <= numeric_type::f64;
}

bool valid_address(address_intent_v1 intent) noexcept {
    return intent >= address_intent_v1::unconstrained &&
        intent <= address_intent_v1::peer_device;
}

state_value_ir_validation_code_v1 validate_common(
    semantic_identity_v1 identity,
    const numeric_tuple_ir_v1& numeric,
    semantic_identity_v1 order,
    value_generation_ir_v1 generation,
    value_mutability_v1 mutability,
    address_intent_v1 address,
    alias_contract_ir_v1 alias) noexcept {
    if (!identity.valid()) return state_value_ir_validation_code_v1::invalid_identity;
    const auto numeric_status = validate_numeric_tuple_ir_v1(numeric);
    if (numeric_status != state_value_ir_validation_code_v1::success) return numeric_status;
    if (!order.valid()) return state_value_ir_validation_code_v1::invalid_order;
    if (generation.value == 0) return state_value_ir_validation_code_v1::invalid_generation;
    if (mutability != value_mutability_v1::immutable &&
        mutability != value_mutability_v1::mutable_values)
        return state_value_ir_validation_code_v1::invalid_mutability;
    if (!valid_address(address)) return state_value_ir_validation_code_v1::invalid_address_intent;
    if (alias.may_alias_input && alias.alias_class == 0)
        return state_value_ir_validation_code_v1::invalid_alias_contract;
    return state_value_ir_validation_code_v1::success;
}

}  // namespace

state_value_ir_validation_code_v1
validate_numeric_tuple_ir_v1(const numeric_tuple_ir_v1& numeric) noexcept {
    if (!valid_numeric(numeric.storage) || !valid_numeric(numeric.compute) ||
        !valid_numeric(numeric.accumulation) || !valid_numeric(numeric.output))
        return state_value_ir_validation_code_v1::invalid_numeric_tuple;
    if (numeric.compute == cellerator::execution::numeric_type::bit ||
        numeric.accumulation == cellerator::execution::numeric_type::bit)
        return state_value_ir_validation_code_v1::invalid_numeric_tuple;
    return state_value_ir_validation_code_v1::success;
}

state_value_ir_validation_code_v1
validate_state_ir_type_v1(const state_ir_type_v1& state) noexcept {
    const auto common = validate_common(state.identity, state.numeric, state.order,
                                        state.generation, state.mutability,
                                        state.address_intent, state.alias);
    if (common != state_value_ir_validation_code_v1::success) return common;
    if (state.axes.empty() ||
        std::any_of(state.axes.begin(), state.axes.end(),
                    [](semantic_identity_v1 axis) { return !axis.valid(); }))
        return state_value_ir_validation_code_v1::invalid_axis;
    if (state.dense_width == 0) return state_value_ir_validation_code_v1::invalid_width;
    return state_value_ir_validation_code_v1::success;
}

state_value_ir_validation_code_v1
validate_value_plane_ir_type_v1(const value_plane_ir_type_v1& plane) noexcept {
    const auto common = validate_common(plane.identity, plane.numeric, plane.order,
                                        plane.generation, plane.mutability,
                                        plane.address_intent, plane.alias);
    if (common != state_value_ir_validation_code_v1::success) return common;
    if (!plane.structure.valid() || plane.structure_epoch == 0)
        return state_value_ir_validation_code_v1::invalid_structure;
    return state_value_ir_validation_code_v1::success;
}

cellerator::compute::operation::v2::numerical_policy
to_operation_numeric_policy_v1(const numeric_tuple_ir_v1& numeric) noexcept {
    cellerator::compute::operation::v2::numerical_policy result;
    result.relation_storage = numeric.storage;
    result.state_storage = numeric.storage;
    result.multiply = numeric.compute;
    result.accumulation = numeric.accumulation;
    result.output_storage = numeric.output;
    result.scalar = numeric.compute;
    return result;
}

numeric_tuple_ir_v1 from_operation_numeric_policy_v1(
    const cellerator::compute::operation::v2::numerical_policy& numeric) noexcept {
    return {numeric.state_storage, numeric.multiply, numeric.accumulation,
            numeric.output_storage};
}

}  // namespace Cellerator::compiler::ir::semantic
