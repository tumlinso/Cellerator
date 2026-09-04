#include <Cellerator/compiler/ir/semantic/implement_native_and_opaque_c_call_operations_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

native_call_operation_ir_v1 contracted(std::uint64_t identity,
                                       std::uint64_t alias,
                                       native_call_access_ir_v1 access) {
    native_call_operation_ir_v1 call;
    call.identity = {identity, identity + 1};
    call.resolved_symbol_identity = identity + 100;
    call.operands = {{{identity + 2, identity + 3}, alias, access}};
    call.results = {{identity + 4, identity + 5}};
    call.effects = access == native_call_access_ir_v1::read
        ? native_effect_reads_memory_v1
        : native_effect_reads_memory_v1 | native_effect_writes_memory_v1;
    call.explicit_effect_contract = true;
    call.deterministic = true;
    call.provenance = {identity + 200, "native.cc", identity};
    return call;
}

int main() {
    const auto reader = contracted(1, 10, native_call_access_ir_v1::read);
    const auto independent_writer = contracted(20, 11, native_call_access_ir_v1::write);
    const auto aliased_writer = contracted(40, 10, native_call_access_ir_v1::write);
    assert(validate_native_call_operation_ir_v1(reader) == native_call_status_ir_v1::success);
    assert(native_calls_may_reorder_v1(reader, independent_writer));
    assert(!native_calls_may_reorder_v1(reader, aliased_writer));

    auto opaque = contracted(60, 12, native_call_access_ir_v1::read_write);
    opaque.explicit_effect_contract = false;
    opaque.effects = native_effect_opaque_barrier_v1 | native_effect_reads_memory_v1 |
        native_effect_writes_memory_v1 | native_effect_may_throw_v1;
    opaque.deterministic = false;
    assert(validate_native_call_operation_ir_v1(opaque) == native_call_status_ir_v1::success);
    assert(!native_calls_may_reorder_v1(independent_writer, opaque));

    opaque.effects &= ~native_effect_may_throw_v1;
    assert(validate_native_call_operation_ir_v1(opaque) ==
           native_call_status_ir_v1::invalid_effect_contract);

    std::cout << "contracted_reorder=true alias_hazard=ordered opaque=barrier\n";
}
