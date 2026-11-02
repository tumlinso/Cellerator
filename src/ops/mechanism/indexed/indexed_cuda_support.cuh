#pragma once

#include <Cellerator/memory/session_memory.cuh>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cellerator::ops::mechanism::indexed::detail {

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

inline void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct device_scope {
    int previous = -1;
    explicit device_scope(int device) {
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(device));
    }
    ~device_scope() { if (previous >= 0) cudaSetDevice(previous); }
};

struct session_storage {
    runtime::execution_session session{};
    explicit session_storage(int device) {
        runtime::execution_session_options options;
        options.device = device;
        require(runtime::init_session(&session, options) == runtime::session_status::success,
                "mechanism session initialization failed");
    }
    ~session_storage() { runtime::clear_session(&session); }
    void* reserve(std::size_t bytes) {
        memory::allocation_request request;
        request.bytes = std::max<std::size_t>(bytes, 64);
        request.where = {memory::domain::device, static_cast<std::int16_t>(session.device), -1, 0};
        memory::allocation allocation;
        require(memory::reserve_session_allocation(&session, runtime::persistent_lifetime::graph_stable,
                    request, 1, &allocation) == memory::status::success,
                "mechanism capacity allocation failed");
        return allocation.base;
    }
};

inline std::size_t checked_product(std::size_t a, std::size_t b) {
    require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
            "mechanism capacity overflow");
    return a * b;
}

inline std::size_t checked_sum(std::size_t a, std::size_t b) {
    require(a <= std::numeric_limits<std::size_t>::max() - b, "mechanism capacity overflow");
    return a + b;
}

inline std::size_t checked_align(std::size_t n) {
    return checked_sum(n, 63u) & ~std::size_t(63u);
}

inline std::uint64_t checked_elements(std::uint64_t a, std::uint64_t b) {
    require(b == 0 || a <= std::numeric_limits<std::uint64_t>::max() / b,
            "mechanism element count overflow");
    return a * b;
}

inline bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    if (!a || !b || !a_bytes || !b_bytes) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    const auto maximum = std::numeric_limits<std::uintptr_t>::max();
    if (a_bytes > maximum - x || b_bytes > maximum - y) return true;
    return x < y + b_bytes && y < x + a_bytes;
}

inline void validate_stream_device(void* stream, int expected) {
    CUcontext current = nullptr;
    require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr,
            "current CUDA context is unavailable");
    CUdevice current_device = -1;
    require(cuCtxGetDevice(&current_device) == CUDA_SUCCESS,
            "current CUDA context device query failed");
    require(static_cast<int>(current_device) == expected,
            "current CUDA context does not match mechanism owner");
    CUcontext stream_context = nullptr;
    require(cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &stream_context) == CUDA_SUCCESS,
            "CUDA stream context query failed");
    require(stream_context == current, "CUDA stream context does not match mechanism owner");
}

struct event {
    cudaEvent_t value = nullptr;
    event() { check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~event() { if (value) cudaEventDestroy(value); }
    event(const event&) = delete;
    event& operator=(const event&) = delete;
};

inline unsigned blocks(std::uint64_t count) {
    if (count == 0) return 0;
    return static_cast<unsigned>(std::min<std::uint64_t>(count / 256 + (count % 256 != 0), 65535));
}

} // namespace cellerator::ops::mechanism::indexed::detail
