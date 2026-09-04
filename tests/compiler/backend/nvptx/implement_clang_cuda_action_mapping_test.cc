#include <Cellerator/compiler/backend/nvptx/implement_clang_cuda_action_mapping_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::backend::nvptx;

int main() {
    clang_cuda_toolchain_v1 toolchain{
        "/usr/bin/clang++-18", "/usr/bin/clang-offload-bundler-18",
        "/usr/local/cuda", "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"};
    clang_cuda_mapping_request_v1 request;
    request.source_path = "generated.cu";
    request.output_stem = "generated";
    request.compute_major = 7u;
    request.include_paths = {"generated/include"};
    request.libraries = {"-lcuda"};

    const auto plan = map_clang_cuda_actions_v1(toolchain, request);
    assert(plan && plan.actions.size() == 4u);
    assert(plan.actions[0].kind == clang_cuda_action_kind_v1::device_compile);
    assert(plan.actions[0].arguments[0] == "--cuda-gpu-arch=sm_70");
    assert(plan.actions[1].kind == clang_cuda_action_kind_v1::host_compile);
    assert(plan.actions[2].kind == clang_cuda_action_kind_v1::offload_bundle);
    assert(plan.actions[3].kind == clang_cuda_action_kind_v1::link);
    assert(plan.actions[3].arguments[1] == "-L/usr/local/cuda/lib64");

    request.compute_major = 0u;
    assert(map_clang_cuda_actions_v1(toolchain, request).status ==
           clang_cuda_mapping_status_v1::invalid_request);
}
