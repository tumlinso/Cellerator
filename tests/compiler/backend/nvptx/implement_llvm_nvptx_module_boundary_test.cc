#include <Cellerator/compiler/backend/nvptx/implement_llvm_nvptx_module_boundary_v1.hh>

#include <cassert>
#include <iostream>
#include <string_view>

using namespace Cellerator::compiler::backend::nvptx;

int main(int argc, char** argv) {
    nvptx_module_request_v1 request;
    request.realization_module_identity = 71u;
    request.compute_major = 7u;
    request.operations = {{81u, nvptx_scalar_operation_v1::add_f32, "add_one"}};
    const auto result = lower_realization_subset_to_llvm_nvptx_v1(request);
    assert(result);
    assert(result.llvm_ir.find("target triple = \"nvptx64-nvidia-cuda\"") !=
           std::string::npos);
    assert(result.llvm_ir.find("define ptx_kernel void @add_one") != std::string::npos);
    assert(result.llvm_ir.find("; realization-node 81") != std::string::npos);

    request.operations[0].symbol = "bad-symbol";
    assert(lower_realization_subset_to_llvm_nvptx_v1(request).status ==
           nvptx_module_status_v1::invalid_module);

    if (argc == 2 && std::string_view(argv[1]) == "--emit") {
        std::cout << result.llvm_ir;
    }
}
