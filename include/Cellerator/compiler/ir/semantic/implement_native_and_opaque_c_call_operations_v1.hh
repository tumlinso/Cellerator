#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

enum class native_call_access_ir_v1 : std::uint8_t { read = 1, write, read_write };
enum native_call_effect_ir_v1 : std::uint32_t {
    native_effect_none_v1 = 0,
    native_effect_reads_memory_v1 = 1u << 0,
    native_effect_writes_memory_v1 = 1u << 1,
    native_effect_synchronizes_v1 = 1u << 2,
    native_effect_io_v1 = 1u << 3,
    native_effect_atomic_v1 = 1u << 4,
    native_effect_may_throw_v1 = 1u << 5,
    native_effect_opaque_barrier_v1 = 1u << 6,
};

struct native_call_operand_ir_v1 {
    semantic_identity_v1 typed_value{};
    std::uint64_t alias_class = 0;
    native_call_access_ir_v1 access = native_call_access_ir_v1::read;
};

struct native_call_provenance_ir_v1 {
    std::uint64_t source_identity = 0;
    std::string source_file;
    std::uint64_t byte_offset = 0;
};

struct native_call_operation_ir_v1 {
    semantic_identity_v1 identity{};
    std::uint64_t resolved_symbol_identity = 0;
    std::vector<native_call_operand_ir_v1> operands;
    std::vector<semantic_identity_v1> results;
    std::uint32_t effects = native_effect_opaque_barrier_v1 |
        native_effect_reads_memory_v1 | native_effect_writes_memory_v1 |
        native_effect_may_throw_v1;
    bool explicit_effect_contract = false;
    bool deterministic = false;
    native_call_provenance_ir_v1 provenance;
};

enum class native_call_status_ir_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_operand,
    invalid_result,
    invalid_effect_contract,
    invalid_provenance,
};

[[nodiscard]] native_call_status_ir_v1
validate_native_call_operation_ir_v1(const native_call_operation_ir_v1& call) noexcept;

[[nodiscard]] bool native_calls_may_reorder_v1(
    const native_call_operation_ir_v1& first,
    const native_call_operation_ir_v1& second) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
