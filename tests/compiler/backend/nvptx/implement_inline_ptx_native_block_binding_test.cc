#include <Cellerator/compiler/backend/nvptx/implement_inline_ptx_native_block_binding_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::backend::nvptx;
using Cellerator::compiler::frontend::parser::native_backend_kind_v1;

namespace {

inline_native_block_request_v1 base_request() {
    inline_native_block_request_v1 request;
    request.fragment.backend = native_backend_kind_v1::ptx;
    request.fragment.target = "sm_70";
    request.fragment.inputs = {"input"};
    request.fragment.outputs = {"output"};
    request.fragment.clobbers = {"memory", "predicate"};
    request.fragment.effects = {"reads(input)", "writes(output)"};
    request.fragment.fallback = "cuda_exact";
    request.fragment.payload = "ld.global.f32; add.f32; st.global.f32;";
    request.inputs = {{"input", 2u, direct_ptx_type_v1::b64}};
    request.outputs = {{"output", 3u, direct_ptx_type_v1::b64}};
    request.declared_effects = native_block_effect_read_v1 | native_block_effect_write_v1;
    request.provenance = {"kernels/add.ce", 40u, 96u,
                          "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"};
    return request;
}

direct_ptx_kernel_v1 typed_kernel() {
    direct_ptx_kernel_v1 kernel;
    kernel.realization_kernel_identity = 11u;
    kernel.symbol = "inline_add";
    kernel.registers = {{1u, direct_ptx_type_v1::f32}};
    direct_ptx_operation_v1 operation;
    operation.realization_node_identity = 12u;
    operation.opcode = "add_f32";
    operation.result_register = 1u;
    operation.operand_registers = {1u};
    operation.requirement = {7u, 0u, "base_f32"};
    kernel.operations = {operation};
    return kernel;
}

}  // namespace

int main() {
    auto model = typed_kernel();
    auto safe_request = base_request();
    safe_request.typed_ptx_kernel = &model;
    const auto safe = bind_inline_native_block_v1(safe_request);
    assert(safe && !safe.validation_bypassed && safe.target_sm_major == 7u &&
           safe.exact_fallback == "cuda_exact");

    auto trusted_request = base_request();
    trusted_request.trust = native_block_trust_v1::trusted;
    const auto trusted = bind_inline_native_block_v1(trusted_request);
    assert(trusted && !trusted.validation_bypassed);

    auto unsafe_request = base_request();
    unsafe_request.trust = native_block_trust_v1::unsafe;
    unsafe_request.unsafe_acknowledged = true;
    unsafe_request.fragment.clobbers = {"expert_unknown_register_bank"};
    const auto unsafe = bind_inline_native_block_v1(unsafe_request);
    assert(unsafe && unsafe.validation_bypassed);

    auto invalid = base_request();
    invalid.fragment.fallback.clear();
    assert(bind_inline_native_block_v1(invalid).status ==
           inline_native_block_status_v1::invalid_fragment);
    invalid = base_request();
    invalid.trust = native_block_trust_v1::unsafe;
    assert(bind_inline_native_block_v1(invalid).status ==
           inline_native_block_status_v1::unsafe_not_acknowledged);

    std::cout << "safe trusted unsafe and invalid inline native bindings validated\n";
}
