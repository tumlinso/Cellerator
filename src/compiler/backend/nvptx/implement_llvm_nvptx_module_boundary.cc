#include <Cellerator/compiler/backend/nvptx/implement_llvm_nvptx_module_boundary_v1.hh>

#include <cctype>
#include <sstream>

namespace Cellerator::compiler::backend::nvptx {
namespace {

bool valid_symbol(const std::string& symbol) {
    if (symbol.empty() || !(std::isalpha(static_cast<unsigned char>(symbol[0])) ||
                            symbol[0] == '_')) return false;
    for (const char value : symbol) {
        if (!(std::isalnum(static_cast<unsigned char>(value)) || value == '_')) return false;
    }
    return true;
}

}  // namespace

nvptx_module_result_v1 lower_realization_subset_to_llvm_nvptx_v1(
    const nvptx_module_request_v1& request) {
    if (request.realization_module_identity == 0u || request.compute_major == 0u ||
        request.compute_minor > 99u || request.operations.empty()) {
        return {nvptx_module_status_v1::invalid_module, {}};
    }

    std::ostringstream ir;
    ir << "; Cellerator realization module " << request.realization_module_identity << '\n'
       << "target triple = \"nvptx64-nvidia-cuda\"\n\n";
    for (const auto& operation : request.operations) {
        if (operation.realization_node_identity == 0u || !valid_symbol(operation.symbol)) {
            return {nvptx_module_status_v1::invalid_module, {}};
        }
        if (operation.operation != nvptx_scalar_operation_v1::add_f32) {
            return {nvptx_module_status_v1::unsupported_operation, {}};
        }
        ir << "; realization-node " << operation.realization_node_identity << '\n'
           << "define ptx_kernel void @" << operation.symbol
           << "(ptr addrspace(1) %input, ptr addrspace(1) %output, i64 %index) {\n"
           << "entry:\n"
           << "  %input_ptr = getelementptr float, ptr addrspace(1) %input, i64 %index\n"
           << "  %value = load float, ptr addrspace(1) %input_ptr, align 4\n"
           << "  %result = fadd float %value, 1.000000e+00\n"
           << "  %output_ptr = getelementptr float, ptr addrspace(1) %output, i64 %index\n"
           << "  store float %result, ptr addrspace(1) %output_ptr, align 4\n"
           << "  ret void\n"
           << "}\n\n";
    }
    return {nvptx_module_status_v1::success, ir.str()};
}

}  // namespace Cellerator::compiler::backend::nvptx
