#include <Cellerator/compiler/driver/discover_clang_cuda_and_llvm_nvptx_toolchains_v1.hh>
namespace cellerator::compiler::driver {
clang_cuda_toolchain_v1 discover_clang_cuda_and_nvptx_v1(const clang_cuda_discovery_input_v1& in, const clang_cuda_probe_v1& exists) {
    clang_cuda_toolchain_v1 out;
    out.clang_cxx = in.clang_root + "/bin/clang++"; out.host_available = exists(out.clang_cxx);
    out.llvm_config = in.llvm_root + "/bin/llvm-config";
    out.cuda_resource = in.cuda_root; out.libdevice = in.cuda_root + "/nvvm/libdevice/libdevice.10.bc";
    out.ptxas = in.cuda_root + "/bin/ptxas";
    const bool cuda_resources = exists(out.libdevice) && exists(out.ptxas);
    out.cuda_route_available = out.host_available && cuda_resources;
    out.nvptx_route_available = out.cuda_route_available && exists(out.llvm_config) && !in.target.empty();
    if (!out.host_available) out.diagnostic = "optional Clang host route unavailable";
    else if (!out.cuda_route_available) out.diagnostic = "optional Clang CUDA route unavailable; host route remains usable";
    else if (!out.nvptx_route_available) out.diagnostic = "optional LLVM/NVPTX route unavailable; host and Clang CUDA routes remain usable";
    return out;
}
}  // namespace cellerator::compiler::driver
