#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

int parse_device_env(int count) {
    const char *raw = std::getenv("CELLERATOR_CUDA_PROBE_DEVICE");
    if (raw == nullptr || raw[0] == '\0') return 0;
    char *end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed < 0 || parsed >= count) return 0;
    return static_cast<int>(parsed);
}

int bool_int(bool value) {
    return value ? 1 : 0;
}

} // namespace

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        std::cout << "cuda_probe_ok=0\n";
        std::cout << "gpu_count=0\n";
        std::cout << "cuda_error=" << cudaGetErrorString(err) << "\n";
        return 0;
    }

    const int device = parse_device_env(count);
    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cout << "cuda_probe_ok=0\n";
        std::cout << "gpu_count=" << count << "\n";
        std::cout << "selected_device=" << device << "\n";
        std::cout << "cuda_error=" << cudaGetErrorString(err) << "\n";
        return 0;
    }

    const int sm = prop.major * 10 + prop.minor;
    std::cout << "cuda_probe_ok=1\n";
    std::cout << "gpu_count=" << count << "\n";
    std::cout << "selected_device=" << device << "\n";
    std::cout << "name=" << prop.name << "\n";
    std::cout << "compute_major=" << prop.major << "\n";
    std::cout << "compute_minor=" << prop.minor << "\n";
    std::cout << "sm=" << sm << "\n";
    std::cout << "total_global_mem_mib=" << static_cast<unsigned long long>(prop.totalGlobalMem / (1024ull * 1024ull)) << "\n";
    std::cout << "multiprocessors=" << prop.multiProcessorCount << "\n";
    std::cout << "warp_size=" << prop.warpSize << "\n";
    std::cout << "has_fp16_tensor_cores=" << bool_int(sm >= 70) << "\n";
    std::cout << "has_bf16_tensor_cores=" << bool_int(sm >= 80) << "\n";
    std::cout << "has_tf32_tensor_cores=" << bool_int(sm >= 80) << "\n";
    std::cout << "has_fp8_tensor_cores=" << bool_int(sm >= 89) << "\n";
    return 0;
}
