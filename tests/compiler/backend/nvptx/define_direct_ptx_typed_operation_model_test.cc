#include <Cellerator/compiler/backend/nvptx/define_direct_ptx_typed_operation_model_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::backend::nvptx;

int main() {
    direct_ptx_kernel_v1 kernel;
    kernel.realization_kernel_identity = 101u;
    kernel.symbol = "typed_add";
    kernel.parameters = {{"output", direct_ptx_type_v1::b64,
                          direct_ptx_address_space_v1::global, 8u}};
    kernel.registers = {{1u, direct_ptx_type_v1::predicate},
                        {2u, direct_ptx_type_v1::f32},
                        {3u, direct_ptx_type_v1::b64}};

    direct_ptx_operation_v1 entry;
    entry.realization_node_identity = 201u;
    entry.kind = direct_ptx_node_kind_v1::label;
    entry.label = "entry";

    direct_ptx_operation_v1 store;
    store.realization_node_identity = 202u;
    store.opcode = "st_global_f32";
    store.operand_registers = {3u, 2u};
    store.predicate_register = 1u;
    store.address_space = direct_ptx_address_space_v1::global;
    store.memory_effects = direct_ptx_memory_write_v1;
    store.requirement = {7u, 0u, "base_store"};

    direct_ptx_operation_v1 barrier;
    barrier.realization_node_identity = 203u;
    barrier.kind = direct_ptx_node_kind_v1::barrier;
    barrier.barrier = direct_ptx_barrier_kind_v1::synchronize;
    barrier.collective_scope = direct_ptx_collective_scope_v1::cooperative_thread_array;
    barrier.collective_threads = 128u;
    barrier.memory_effects = direct_ptx_memory_read_v1 | direct_ptx_memory_write_v1;

    direct_ptx_operation_v1 vote;
    vote.realization_node_identity = 204u;
    vote.kind = direct_ptx_node_kind_v1::collective;
    vote.result_register = 1u;
    vote.operand_registers = {1u};
    vote.collective = direct_ptx_collective_kind_v1::vote_all;
    vote.collective_scope = direct_ptx_collective_scope_v1::warp;
    vote.collective_threads = 32u;

    kernel.operations = {entry, store, barrier, vote};
    std::string diagnostic;
    assert(validate_direct_ptx_kernel_v1(kernel, 7u, 0u, &diagnostic) ==
           direct_ptx_model_status_v1::success);

    const auto text = print_direct_ptx_kernel_model_v1(kernel);
    direct_ptx_kernel_v1 parsed;
    assert(parse_direct_ptx_kernel_model_v1(text, &parsed, &diagnostic) ==
           direct_ptx_model_status_v1::success);
    assert(print_direct_ptx_kernel_model_v1(parsed) == text);
    assert(validate_direct_ptx_kernel_v1(parsed, 7u, 0u, &diagnostic) ==
           direct_ptx_model_status_v1::success);

    parsed.operations[1].operand_registers[0] = 999u;
    assert(validate_direct_ptx_kernel_v1(parsed, 7u, 0u) ==
           direct_ptx_model_status_v1::invalid_reference);
    parsed = kernel;
    parsed.operations[1].requirement.minimum_sm_major = 8u;
    assert(validate_direct_ptx_kernel_v1(parsed, 7u, 0u) ==
           direct_ptx_model_status_v1::unsupported_requirement);

    std::cout << "typed PTX realization model round-trip and validation passed\n";
}
