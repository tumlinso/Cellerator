#include <Cellerator/compiler/backend/nvptx/compare_nvcc_clang_cuda_and_direct_ptx_routes_v1.hh>
#include <Cellerator/compiler/backend/nvptx/deliver_a_direct_ptx_hot_path_demonstration_v1.hh>

#include <bench/benchmark_mutex.hh>
#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

using namespace Cellerator::compiler::backend::nvptx;

namespace {

std::string quoted(const std::string& value) { return "'" + value + "'"; }

std::string read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), {}};
}

std::uint32_t registers(const std::string& diagnostic) {
    const auto marker = diagnostic.find(" registers");
    if (marker == std::string::npos) return 0u;
    auto begin = marker;
    while (begin != 0u && diagnostic[begin - 1u] == ' ') --begin;
    auto digits = begin;
    while (digits != 0u && std::isdigit(static_cast<unsigned char>(diagnostic[digits - 1u]))) --digits;
    return digits == begin ? 0u : static_cast<std::uint32_t>(std::stoul(diagnostic.substr(digits, begin - digits)));
}

std::uint64_t compile_route(const std::string& command) {
    const auto begin = std::chrono::steady_clock::now();
    const int status = std::system(command.c_str());
    assert(status == 0);
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count());
}

void cuda_ok(const CUresult status) { assert(status == CUDA_SUCCESS); }

std::uint64_t execute(const std::vector<unsigned char>& cubin) {
    constexpr unsigned rows = 4096u;
    constexpr unsigned inputs = 257u;
    std::vector<unsigned> columns(rows);
    std::vector<float> weights(rows), input(inputs), output(rows);
    for (unsigned i = 0; i < inputs; ++i) input[i] = 0.125f * i - 2.0f;
    for (unsigned i = 0; i < rows; ++i) {
        columns[i] = (i * 29u + 7u) % inputs;
        weights[i] = 0.25f + 0.01f * (i % 43u);
    }
    CUmodule module;
    CUfunction function;
    cuda_ok(cuModuleLoadData(&module, cubin.data()));
    cuda_ok(cuModuleGetFunction(&function, module, "ce_unit_degree_relation_apply_sm70"));
    CUdeviceptr dc = 0, dw = 0, di = 0, dout = 0;
    cuda_ok(cuMemAlloc(&dc, sizeof(unsigned) * rows));
    cuda_ok(cuMemAlloc(&dw, sizeof(float) * rows));
    cuda_ok(cuMemAlloc(&di, sizeof(float) * inputs));
    cuda_ok(cuMemAlloc(&dout, sizeof(float) * rows));
    cuda_ok(cuMemcpyHtoD(dc, columns.data(), sizeof(unsigned) * rows));
    cuda_ok(cuMemcpyHtoD(dw, weights.data(), sizeof(float) * rows));
    cuda_ok(cuMemcpyHtoD(di, input.data(), sizeof(float) * inputs));
    void* arguments[] = {&dc, &dw, &di, &dout, const_cast<unsigned*>(&rows)};
    for (unsigned warmup = 0; warmup < 3u; ++warmup)
        cuda_ok(cuLaunchKernel(function, rows / 128u, 1, 1, 128, 1, 1, 0, nullptr, arguments, nullptr));
    cuda_ok(cuCtxSynchronize());
    std::vector<std::uint64_t> samples;
    for (unsigned repeat = 0; repeat < 11u; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        cuda_ok(cuLaunchKernel(function, rows / 128u, 1, 1, 128, 1, 1, 0, nullptr, arguments, nullptr));
        cuda_ok(cuCtxSynchronize());
        samples.push_back(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count()));
    }
    cuda_ok(cuMemcpyDtoH(output.data(), dout, sizeof(float) * rows));
    for (unsigned row = 0; row < rows; ++row)
        assert(std::abs(output[row] - weights[row] * input[columns[row]]) <= 1.0e-6f);
    cuda_ok(cuMemFree(dout)); cuda_ok(cuMemFree(di)); cuda_ok(cuMemFree(dw)); cuda_ok(cuMemFree(dc));
    cuda_ok(cuModuleUnload(module));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2u];
}

}  // namespace

int main(int argc, char** argv) {
    assert(argc == 7);
    const cellerator::bench::benchmark_mutex_guard mutex("ce-ccp1-f04-route-comparison", 0);
    const std::string nvcc = argv[1], clang = argv[2], ptxas = argv[3], fixture = argv[4];
    const std::filesystem::path scratch = argv[5];
    const std::string gcc_install = argv[6];
    std::filesystem::create_directories(scratch);
    const auto direct = lower_unit_degree_relation_apply_directly_to_ptx_v1({4096u, 257u, 7u, 0u});
    assert(direct);
    { std::ofstream out(scratch / "direct.ptx"); out << direct.ptx; }

    struct route_file { nvptx_route_v1 route; std::string identity; std::string ptx; };
    const std::vector<route_file> routes = {
        {nvptx_route_v1::nvcc, "nvcc 12.0.140 + ptxas 12.9", "nvcc.ptx"},
        {nvptx_route_v1::clang_cuda, "clang 18.1.3 CUDA device-only + ptxas 12.9", "clang.ptx"},
        {nvptx_route_v1::direct_ptx, "Cellerator direct PTX schema v1 + ptxas 12.9", "direct.ptx"},
    };
    std::vector<nvptx_route_measurement_v1> measurements;
    CUdevice device; CUcontext context;
    cuda_ok(cuInit(0)); cuda_ok(cuDeviceGet(&device, 0)); cuda_ok(cuCtxCreate(&context, 0, device));
    for (const auto& route : routes) {
        const auto ptx_path = scratch / route.ptx;
        std::uint64_t compile_ns = 0u;
        if (route.route == nvptx_route_v1::nvcc) {
            compile_ns += compile_route(quoted(nvcc) + " -ccbin /usr/bin/g++-12 -arch=sm_70 -ptx " +
                                        quoted(fixture) + " -o " + quoted(ptx_path.string()));
        } else if (route.route == nvptx_route_v1::clang_cuda) {
            compile_ns += compile_route(quoted(clang) + " --cuda-device-only --cuda-gpu-arch=sm_70"
                " -nocudainc -nocudalib --gcc-install-dir=" + quoted(gcc_install) +
                " -I/usr/include -include __clang_cuda_runtime_wrapper.h -S " + quoted(fixture) +
                " -o " + quoted(ptx_path.string()));
        }
        const auto cubin = scratch / (route.ptx + ".cubin");
        const auto log = scratch / (route.ptx + ".log");
        compile_ns += compile_route(quoted(ptxas) + " -v -arch=sm_70 " + quoted(ptx_path.string()) +
                                    " -o " + quoted(cubin.string()) + " >" + quoted(log.string()) + " 2>&1");
        const auto image_text = read(cubin);
        const std::vector<unsigned char> image(image_text.begin(), image_text.end());
        const auto diagnostics = read(log);
        measurements.push_back({route.route, route.identity, compile_ns,
            static_cast<std::uint64_t>(image.size()), registers(diagnostics), 0u, 0u,
            execute(image), route.route == nvptx_route_v1::direct_ptx ? 2u : 3u,
            route.route == nvptx_route_v1::direct_ptx ? 3u : 1u, true, true, false});
    }
    cuda_ok(cuCtxDestroy(context));
    const auto comparison = compare_nvptx_routes_v1(
        measurements, "V100 sm_70 unit-degree f32 relation rows=4096 inputs=257 warmup=3 repeats=11", 1.05);
    assert(comparison.disposition != nvptx_route_promotion_v1::invalid_evidence);
    for (const auto& value : comparison.measurements)
        std::cout << "route=" << static_cast<unsigned>(value.route)
                  << " compile_ns=" << value.compile_nanoseconds << " bytes=" << value.object_bytes
                  << " registers=" << value.registers << " median_execution_ns="
                  << value.median_execution_nanoseconds << '\n';
    std::cout << "disposition=" << static_cast<unsigned>(comparison.disposition)
              << " selected=" << static_cast<unsigned>(comparison.selected_route)
              << " reason=" << comparison.reason << '\n';
}
