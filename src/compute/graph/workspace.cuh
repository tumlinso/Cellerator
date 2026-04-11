#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace cellerator::compute::graph {

inline void cuda_require(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return;
    throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(err));
}

template<typename T>
class device_buffer {
public:
    device_buffer() = default;

    explicit device_buffer(std::size_t size) {
        resize(size);
    }

    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;

    device_buffer(device_buffer &&other) noexcept
        : data_(other.data_),
          size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    device_buffer &operator=(device_buffer &&other) noexcept {
        if (this == &other) return *this;
        reset();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    ~device_buffer() {
        reset();
    }

    void resize(std::size_t next_size) {
        if (next_size == size_) return;
        reset();
        if (next_size == 0) return;
        cuda_require(cudaMalloc(reinterpret_cast<void **>(&data_), next_size * sizeof(T)), "cudaMalloc(device_buffer)");
        size_ = next_size;
    }

    void reset() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    void upload(const T *host_data, std::size_t count) {
        if (count > size_) throw std::out_of_range("device_buffer upload exceeds allocation");
        if (count == 0) return;
        cuda_require(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
    }

    void download(T *host_data, std::size_t count) const {
        if (count > size_) throw std::out_of_range("device_buffer download exceeds allocation");
        if (count == 0) return;
        cuda_require(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
    }

    T *data() { return data_; }
    const T *data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace cellerator::compute::graph
