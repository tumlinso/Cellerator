#include <Cellerator/compiler/backend/nvptx/implement_clang_cuda_action_mapping_v1.hh>

namespace Cellerator::compiler::backend::nvptx {

clang_cuda_action_plan_v1 map_clang_cuda_actions_v1(
    const clang_cuda_toolchain_v1& toolchain,
    const clang_cuda_mapping_request_v1& request) {
    if (toolchain.clang_path.empty() || toolchain.bundler_path.empty() ||
        toolchain.cuda_root.empty() || toolchain.libdevice_path.empty()) {
        return {clang_cuda_mapping_status_v1::invalid_toolchain, {}};
    }
    if (request.source_path.empty() || request.output_stem.empty() ||
        request.compute_major == 0u || request.compute_minor > 99u) {
        return {clang_cuda_mapping_status_v1::invalid_request, {}};
    }

    const auto architecture = "--cuda-gpu-arch=sm_" +
        std::to_string(request.compute_major) + std::to_string(request.compute_minor);
    const auto cuda_path = "--cuda-path=" + toolchain.cuda_root;
    const auto device_object = request.output_stem + ".device.o";
    const auto host_object = request.output_stem + ".host.o";
    const auto bundled_object = request.output_stem + ".cuda.o";
    const auto executable = request.output_stem;

    std::vector<std::string> common{architecture, cuda_path,
                                    "-mlink-builtin-bitcode", toolchain.libdevice_path};
    for (const auto& path : request.include_paths) common.push_back("-I" + path);

    auto device_arguments = common;
    device_arguments.insert(device_arguments.end(),
                            {"--cuda-device-only", "-c", request.source_path,
                             "-o", device_object});
    auto host_arguments = common;
    host_arguments.insert(host_arguments.end(),
                          {"--cuda-host-only", "-c", request.source_path,
                           "-o", host_object});

    std::vector<clang_cuda_action_v1> actions;
    actions.push_back({clang_cuda_action_kind_v1::device_compile,
                       toolchain.clang_path, std::move(device_arguments), device_object});
    actions.push_back({clang_cuda_action_kind_v1::host_compile,
                       toolchain.clang_path, std::move(host_arguments), host_object});
    actions.push_back({clang_cuda_action_kind_v1::offload_bundle,
                       toolchain.bundler_path,
                       {"-type=o", "-targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda",
                        "-inputs=" + host_object + "," + device_object,
                        "-outputs=" + bundled_object},
                       bundled_object});

    std::vector<std::string> link_arguments{
        bundled_object, "-L" + toolchain.cuda_root + "/lib64", "-lcudart"};
    link_arguments.insert(link_arguments.end(), request.libraries.begin(),
                          request.libraries.end());
    link_arguments.insert(link_arguments.end(), {"-o", executable});
    actions.push_back({clang_cuda_action_kind_v1::link, toolchain.clang_path,
                       std::move(link_arguments), executable});
    return {clang_cuda_mapping_status_v1::success, std::move(actions)};
}

}  // namespace Cellerator::compiler::backend::nvptx
