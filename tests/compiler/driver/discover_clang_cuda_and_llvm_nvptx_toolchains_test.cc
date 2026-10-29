#include <Cellerator/compiler/driver/discover_clang_cuda_and_llvm_nvptx_toolchains_v1.hh>
#include <cstdlib>
#include <iostream>
#include <set>
using namespace cellerator::compiler::driver;
int main() {
    const clang_cuda_discovery_input_v1 in{"/clang", "/llvm", "/cuda", "nvptx64-nvidia-cuda"};
    std::set<std::string> files{"/clang/bin/clang++"};
    auto probe = [&](std::string_view p){ return files.count(std::string(p)); };
    auto out = discover_clang_cuda_and_nvptx_v1(in, probe);
    if (!out.host_available || out.cuda_route_available || out.nvptx_route_available || out.diagnostic.find("host route remains usable") == std::string::npos) return EXIT_FAILURE;
    files.insert("/cuda/nvvm/libdevice/libdevice.10.bc"); files.insert("/cuda/bin/ptxas");
    out = discover_clang_cuda_and_nvptx_v1(in, probe);
    if (!out.cuda_route_available || out.nvptx_route_available) return EXIT_FAILURE;
    files.insert("/llvm/bin/llvm-config"); out = discover_clang_cuda_and_nvptx_v1(in, probe);
    if (!out.nvptx_route_available || !out.diagnostic.empty()) return EXIT_FAILURE;
    std::cout << "validated optional Clang CUDA/NVPTX routes without host-only breakage\n";
}
