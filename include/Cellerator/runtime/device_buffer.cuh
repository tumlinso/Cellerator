#pragma once

#include "error.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cellerator::runtime {

template<typename T>
struct device_buffer {
    std::shared_ptr<void> owner;
    T *data = nullptr;
    std::size_t count = 0;
};

template<typename T>
inline device_buffer<T> allocate_device_buffer(std::size_t count) {
    device_buffer<T> out;
    out.count = count;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("allocate_device_buffer byte extent overflow");
    }
    if (count == 0) return out;

    int device_ordinal = 0;
    cuda_require(cudaGetDevice(&device_ordinal), "cudaGetDevice(allocate_device_buffer)");

    T *ptr = nullptr;
    cuda_require(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)), "cudaMalloc(core runtime)");
    out.owner = std::shared_ptr<void>(
        ptr,
        [device_ordinal](void *storage) {
            if (storage == nullptr) return;
            int previous = 0;
            if (cudaGetDevice(&previous) != cudaSuccess) return;
            const bool changed = previous != device_ordinal;
            if (changed && cudaSetDevice(device_ordinal) != cudaSuccess) return;
            cudaFree(storage);
            if (changed) cudaSetDevice(previous);
        });
    out.data = ptr;
    return out;
}

template<typename T>
inline void upload_device_buffer(device_buffer<T> *dst, const T *src, std::size_t count) {
    if (dst == nullptr) throw std::invalid_argument("upload_device_buffer requires a destination");
    if (count > dst->count) throw std::out_of_range("upload_device_buffer exceeds allocation");
    if (count == 0) return;
    cuda_require(
        cudaMemcpy(dst->data, src, count * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D core runtime)");
}

template<typename T>
inline void download_device_buffer(const device_buffer<T> &src, T *dst, std::size_t count) {
    if (count > src.count) throw std::out_of_range("download_device_buffer exceeds allocation");
    if (count == 0) return;
    cuda_require(cudaDeviceSynchronize(), "cudaDeviceSynchronize(download_device_buffer)");
    cuda_require(
        cudaMemcpy(dst, src.data, count * sizeof(T), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H core runtime)");
}

} // namespace cellerator::runtime
