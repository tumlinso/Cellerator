#pragma once

#include <Cellerator/compute/operation/device_elementwise.hh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::compute::relation::device_elementwise_detail {
namespace {

template<class T>
__global__ void multiply_kernel(const T* lhs, const T* rhs, T* output,
                                std::uint64_t count) {
    const auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = lhs[i] * rhs[i];
}

template<class T>
__global__ void axpby_kernel(T alpha, const T* x, T beta, const T* y,
                             T* output, std::uint64_t count) {
    const auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = alpha * x[i] + beta * y[i];
}

status cuda_status(cudaError_t error) noexcept {
    return error == cudaSuccess ? status{}
        : status{status_code::cuda_failure, cudaGetErrorString(error)};
}

bool ranges_overlap(std::uintptr_t a, std::uint64_t a_bytes,
                    std::uintptr_t b, std::uint64_t b_bytes) noexcept {
    if (!a_bytes || !b_bytes) return false;
    return a < b + b_bytes && b < a + a_bytes;
}

template<class T>
status check_range(const T* pointer, std::uint64_t bytes, int device) noexcept {
    if (!bytes) return {};
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (!pointer || address % alignof(T) != 0
        || bytes > std::numeric_limits<std::uintptr_t>::max() - address)
        return {status_code::invalid_argument, "null, misaligned or overflowing device range"};

    cudaPointerAttributes attributes{};
    auto error = cudaPointerGetAttributes(&attributes, pointer);
    if (error != cudaSuccess) {
        (void)cudaGetLastError();
        return {status_code::invalid_argument, "pointer is not accessible CUDA storage"};
    }
    bool managed = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 6000
    managed = attributes.type == cudaMemoryTypeManaged;
#endif
    if (attributes.type != cudaMemoryTypeDevice && !managed)
        return {status_code::unsupported_semantics, "allocation kind has no qualified capacity query"};
    if (attributes.device != device)
        return {status_code::incompatible_device, "pointer residency differs from execution device"};

    using address_range_fn = CUresult (CUDAAPI *)(CUdeviceptr*, std::size_t*, CUdeviceptr);
    using pointer_attribute_fn = CUresult (CUDAAPI *)(void*, CUpointer_attribute, CUdeviceptr);
    struct driver_capacity_apis {
        address_range_fn address_range = nullptr;
        pointer_attribute_fn pointer_attribute = nullptr;
    };
    static const driver_capacity_apis capacity_apis = []() noexcept {
        driver_capacity_apis result{};
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11030
        auto lookup = [](const char* name, unsigned int version) noexcept -> void* {
            void* symbol = nullptr;
#if CUDART_VERSION >= 12060
            cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
            if (cudaGetDriverEntryPointByVersion(name, &symbol, version,
                    cudaEnableDefault, &query) != cudaSuccess
                || query != cudaDriverEntryPointSuccess)
                return nullptr;
#else
            (void)version;
            if (cudaGetDriverEntryPoint(name, &symbol, cudaEnableDefault) != cudaSuccess)
                return nullptr;
#endif
            return symbol;
        };
        result.address_range = reinterpret_cast<address_range_fn>(
            lookup("cuMemGetAddressRange", 3020));
        result.pointer_attribute = reinterpret_cast<pointer_attribute_fn>(
            lookup("cuPointerGetAttribute", 4000));
#endif
        return result;
    }();

    void* allocation_base_pointer = nullptr;
    std::size_t allocation_bytes = 0;
    CUresult range_status = CUDA_ERROR_NOT_SUPPORTED;
    if (!managed) {
        if (!capacity_apis.address_range)
            return {status_code::unsupported_semantics, "device allocation capacity query is unavailable"};
        CUdeviceptr allocation_base = 0;
        range_status = capacity_apis.address_range(&allocation_base, &allocation_bytes,
                                                    static_cast<CUdeviceptr>(address));
        allocation_base_pointer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(allocation_base));
    } else {
        if (!capacity_apis.pointer_attribute)
            return {status_code::unsupported_semantics, "managed allocation capacity query is unavailable"};
        range_status = capacity_apis.pointer_attribute(&allocation_base_pointer,
            CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, static_cast<CUdeviceptr>(address));
        if (range_status == CUDA_SUCCESS)
            range_status = capacity_apis.pointer_attribute(&allocation_bytes,
                CU_POINTER_ATTRIBUTE_RANGE_SIZE, static_cast<CUdeviceptr>(address));
    }
    if (range_status != CUDA_SUCCESS)
        return {status_code::unsupported_semantics, "cannot establish CUDA allocation capacity"};

    const auto base = reinterpret_cast<std::uintptr_t>(allocation_base_pointer);
    if (allocation_bytes > std::numeric_limits<std::uintptr_t>::max() - base
        || address < base || address > base + allocation_bytes
        || bytes > base + allocation_bytes - address)
        return {status_code::insufficient_capacity, "device allocation does not cover element range"};
    return {};
}

template<class T>
status preflight(const runtime::execution_context& context,
                 const T* first, const T* second, T* output,
                 std::uint64_t count) noexcept {
    if (count == 0) return {};
    if (context.device < 0)
        return {status_code::invalid_argument, "execution context must name a device"};
    int current_device = -1;
    auto s = cuda_status(cudaGetDevice(&current_device));
    if (!s) return s;
    if (current_device != context.device)
        return {status_code::incompatible_device, "current device differs from execution context"};

    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(T)
        || count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return {status_code::insufficient_capacity, "element byte count overflows"};
    const auto bytes = count * sizeof(T);
    s = check_range(first, bytes, context.device); if (!s) return s;
    s = check_range(second, bytes, context.device); if (!s) return s;
    s = check_range(output, bytes, context.device); if (!s) return s;

    const auto out_address = reinterpret_cast<std::uintptr_t>(output);
    const auto first_address = reinterpret_cast<std::uintptr_t>(first);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second);
    if ((ranges_overlap(out_address, bytes, first_address, bytes) && out_address != first_address)
        || (ranges_overlap(out_address, bytes, second_address, bytes) && out_address != second_address))
        return {status_code::invalid_argument, "output partially overlaps an input range"};
    return {};
}

status launch_grid(std::uint64_t count, unsigned int& blocks) noexcept {
    constexpr std::uint64_t threads = 256;
    constexpr std::uint64_t maximum_grid_x = 0x7fffffffULL;
    const auto needed = count / threads + (count % threads != 0);
    if (needed > maximum_grid_x)
        return {status_code::unsupported_width, "elementwise launch grid exceeds CUDA x dimension"};
    blocks = static_cast<unsigned int>(needed);
    return {};
}

} // namespace

template<class T>
status multiply_impl(const runtime::execution_context& context, const T* lhs,
                     const T* rhs, T* output, std::uint64_t count) noexcept {
    if (!count) return {};
    auto s = preflight(context, lhs, rhs, output, count); if (!s) return s;
    unsigned int blocks = 0;
    s = launch_grid(count, blocks); if (!s) return s;
    multiply_kernel<T><<<blocks, 256, 0, context.stream>>>(lhs, rhs, output, count);
    return cuda_status(cudaPeekAtLastError());
}

template<class T>
status axpby_impl(const runtime::execution_context& context, T alpha,
                  const T* x, T beta, const T* y, T* output,
                  std::uint64_t count) noexcept {
    if (!std::isfinite(alpha) || !std::isfinite(beta))
        return {status_code::invalid_argument, "affine coefficients must be finite"};
    if (!count) return {};
    auto s = preflight(context, x, y, output, count); if (!s) return s;
    unsigned int blocks = 0;
    s = launch_grid(count, blocks); if (!s) return s;
    axpby_kernel<T><<<blocks, 256, 0, context.stream>>>(alpha, x, beta, y, output, count);
    return cuda_status(cudaPeekAtLastError());
}

template status multiply_impl<float>(const runtime::execution_context&, const float*, const float*, float*, std::uint64_t) noexcept;
template status multiply_impl<double>(const runtime::execution_context&, const double*, const double*, double*, std::uint64_t) noexcept;
template status axpby_impl<float>(const runtime::execution_context&, float, const float*, float, const float*, float*, std::uint64_t) noexcept;
template status axpby_impl<double>(const runtime::execution_context&, double, const double*, double, const double*, double*, std::uint64_t) noexcept;

} // namespace cellerator::compute::relation::device_elementwise_detail
