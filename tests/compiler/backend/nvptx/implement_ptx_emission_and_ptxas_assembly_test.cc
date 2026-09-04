#include <Cellerator/compiler/backend/nvptx/implement_ptx_emission_and_ptxas_assembly_v1.hh>

#include <cassert>
#include <filesystem>
#include <iostream>

using namespace Cellerator::compiler::backend::nvptx;

int main(int argc, char** argv) {
    assert(argc == 3);
    direct_ptx_kernel_v1 kernel;
    kernel.realization_kernel_identity = 301u;
    kernel.symbol = "minimal_kernel";
    direct_ptx_operation_v1 entry;
    entry.realization_node_identity = 302u;
    entry.kind = direct_ptx_node_kind_v1::label;
    entry.label = "entry";
    direct_ptx_operation_v1 ret;
    ret.realization_node_identity = 303u;
    ret.opcode = "ret";
    kernel.operations = {entry, ret};

    const auto emitted = emit_deterministic_ptx_v1(kernel, 7u, 0u);
    assert(emitted && emitted.ptx.find(".target sm_70") != std::string::npos &&
           emitted.ptx.find(".visible .entry minimal_kernel") != std::string::npos);
    assert(emit_deterministic_ptx_v1(kernel, 7u, 0u).ptx == emitted.ptx);

    const std::filesystem::path scratch = argv[2];
    std::filesystem::create_directories(scratch);
    ptxas_assembly_request_v1 request;
    request.ptxas_executable = argv[1];
    request.ptx_path = (scratch / "minimal.ptx").string();
    request.cubin_path = (scratch / "minimal.cubin").string();
    request.diagnostic_path = (scratch / "ptxas.log").string();
    request.ptx = emitted.ptx;
    request.target_sm_major = 7u;
    request.retain_ptx = true;
    const auto assembled = assemble_ptx_with_ptxas_v1(request);
    assert(assembled && assembled.exit_code == 0 && !assembled.retained_ptx_path.empty());
    assert(std::filesystem::file_size(assembled.cubin_path) > 0u);
    assert(assembled.diagnostics.find("Compiling entry function 'minimal_kernel'") !=
           std::string::npos);

    std::cout << "deterministic PTX assembled for sm_70 cubin_bytes="
              << std::filesystem::file_size(assembled.cubin_path)
              << " registers=" << assembled.resources.registers << '\n';
}
