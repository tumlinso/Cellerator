#include <Cellerator/compiler/backend/nvptx/deliver_a_direct_ptx_hot_path_demonstration_v1.hh>
#include <Cellerator/compiler/backend/nvptx/implement_ptx_emission_and_ptxas_assembly_v1.hh>

#include <cuda.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

using namespace Cellerator::compiler::backend::nvptx;

namespace {

void cuda_ok(const CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(status, &name);
        std::cerr << "CUDA driver failure: " << (name == nullptr ? "unknown" : name) << '\n';
        std::abort();
    }
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 3);
    constexpr std::uint32_t rows = 257u;
    constexpr std::uint32_t inputs = 67u;
    const auto lowered = lower_unit_degree_relation_apply_directly_to_ptx_v1(
        {rows, inputs, 7u, 0u});
    assert(lowered && lowered.restrictions.size() == 5u);

    const std::filesystem::path scratch = argv[2];
    std::filesystem::create_directories(scratch);
    ptxas_assembly_request_v1 assembly;
    assembly.ptxas_executable = argv[1];
    assembly.ptx_path = (scratch / "relation.ptx").string();
    assembly.cubin_path = (scratch / "relation.cubin").string();
    assembly.diagnostic_path = (scratch / "ptxas.log").string();
    assembly.ptx = lowered.ptx;
    assembly.target_sm_major = 7u;
    const auto assembled = assemble_ptx_with_ptxas_v1(assembly);
    assert(assembled);

    std::ifstream cubin_input(assembled.cubin_path, std::ios::binary);
    const std::vector<unsigned char> cubin((std::istreambuf_iterator<char>(cubin_input)), {});
    assert(!cubin.empty());

    std::vector<std::uint32_t> columns(rows);
    std::vector<float> weights(rows);
    std::vector<float> input(inputs);
    std::vector<float> output(rows, 0.0f);
    for (std::uint32_t index = 0u; index < inputs; ++index) input[index] = 0.25f * index - 3.0f;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        columns[row] = (row * 17u + 3u) % inputs;
        weights[row] = 0.5f + 0.01f * static_cast<float>(row % 31u);
    }

    cuda_ok(cuInit(0));
    CUdevice device;
    cuda_ok(cuDeviceGet(&device, 0));
    CUcontext context;
    cuda_ok(cuCtxCreate(&context, 0u, device));
    CUmodule module;
    cuda_ok(cuModuleLoadData(&module, cubin.data()));
    CUfunction function;
    cuda_ok(cuModuleGetFunction(&function, module, lowered.kernel_symbol.c_str()));
    CUdeviceptr device_columns = 0u;
    CUdeviceptr device_weights = 0u;
    CUdeviceptr device_input = 0u;
    CUdeviceptr device_output = 0u;
    cuda_ok(cuMemAlloc(&device_columns, sizeof(std::uint32_t) * rows));
    cuda_ok(cuMemAlloc(&device_weights, sizeof(float) * rows));
    cuda_ok(cuMemAlloc(&device_input, sizeof(float) * inputs));
    cuda_ok(cuMemAlloc(&device_output, sizeof(float) * rows));
    cuda_ok(cuMemcpyHtoD(device_columns, columns.data(), sizeof(std::uint32_t) * rows));
    cuda_ok(cuMemcpyHtoD(device_weights, weights.data(), sizeof(float) * rows));
    cuda_ok(cuMemcpyHtoD(device_input, input.data(), sizeof(float) * inputs));
    void* arguments[] = {&device_columns, &device_weights, &device_input, &device_output,
                         const_cast<std::uint32_t*>(&rows)};
    cuda_ok(cuLaunchKernel(function, (rows + 127u) / 128u, 1u, 1u, 128u, 1u, 1u,
                           0u, nullptr, arguments, nullptr));
    cuda_ok(cuCtxSynchronize());
    cuda_ok(cuMemcpyDtoH(output.data(), device_output, sizeof(float) * rows));

    for (std::uint32_t row = 0u; row < rows; ++row) {
        const float reference = weights[row] * input[columns[row]];
        assert(std::abs(output[row] - reference) <= 1.0e-6f);
    }
    cuda_ok(cuMemFree(device_output));
    cuda_ok(cuMemFree(device_input));
    cuda_ok(cuMemFree(device_weights));
    cuda_ok(cuMemFree(device_columns));
    cuda_ok(cuModuleUnload(module));
    cuda_ok(cuCtxDestroy(context));

    std::cout << "direct PTX unit-degree relation apply differential passed rows=" << rows
              << " cubin_bytes=" << cubin.size() << '\n';
}
