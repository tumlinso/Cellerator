#pragma once

#include <cstdio>

#include <cuda_runtime.h>

static inline int cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static inline int choose_gpu_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 1;
    if (count < 1) return 1;
    if (count > 4) count = 4;
    return count;
}

static inline int bind_task_gpu(unsigned int task_id, int gpu_count) {
    const int device = gpu_count > 1 ? (int) (task_id % (unsigned int) gpu_count) : 0;
    return cuda_check(cudaSetDevice(device), "cudaSetDevice");
}
