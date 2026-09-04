#include <Cellerator/compiler/ir/semantic/implement_native_and_opaque_c_call_operations_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::ir::semantic {

native_call_status_ir_v1 validate_native_call_operation_ir_v1(
    const native_call_operation_ir_v1& call) noexcept {
    if (!call.identity.valid() || call.resolved_symbol_identity == 0)
        return native_call_status_ir_v1::invalid_identity;
    for (const auto& operand : call.operands) {
        if (!operand.typed_value.valid() || operand.alias_class == 0 ||
            operand.access < native_call_access_ir_v1::read ||
            operand.access > native_call_access_ir_v1::read_write)
            return native_call_status_ir_v1::invalid_operand;
    }
    if (std::any_of(call.results.begin(), call.results.end(),
                    [](semantic_identity_v1 result) { return !result.valid(); }))
        return native_call_status_ir_v1::invalid_result;
    constexpr std::uint32_t conservative = native_effect_opaque_barrier_v1 |
        native_effect_reads_memory_v1 | native_effect_writes_memory_v1 |
        native_effect_may_throw_v1;
    if (!call.explicit_effect_contract && (call.effects & conservative) != conservative)
        return native_call_status_ir_v1::invalid_effect_contract;
    if (call.explicit_effect_contract && (call.effects & native_effect_opaque_barrier_v1) != 0)
        return native_call_status_ir_v1::invalid_effect_contract;
    if (call.provenance.source_identity == 0 || call.provenance.source_file.empty())
        return native_call_status_ir_v1::invalid_provenance;
    return native_call_status_ir_v1::success;
}

bool native_calls_may_reorder_v1(const native_call_operation_ir_v1& first,
                                 const native_call_operation_ir_v1& second) noexcept {
    if (validate_native_call_operation_ir_v1(first) != native_call_status_ir_v1::success ||
        validate_native_call_operation_ir_v1(second) != native_call_status_ir_v1::success)
        return false;
    constexpr std::uint32_t boundary = native_effect_opaque_barrier_v1 |
        native_effect_synchronizes_v1 | native_effect_io_v1 |
        native_effect_atomic_v1 | native_effect_may_throw_v1;
    if ((first.effects & boundary) != 0 || (second.effects & boundary) != 0) return false;
    for (const auto& left : first.operands) {
        for (const auto& right : second.operands) {
            if (left.alias_class != right.alias_class) continue;
            const bool left_writes = left.access != native_call_access_ir_v1::read;
            const bool right_writes = right.access != native_call_access_ir_v1::read;
            if (left_writes || right_writes) return false;
        }
    }
    return true;
}

}  // namespace Cellerator::compiler::ir::semantic
