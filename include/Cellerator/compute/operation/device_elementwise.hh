#pragma once

#include <Cellerator/compute/operation/relation_semantics.hh>
#include <Cellerator/runtime/stream.cuh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::relation {
namespace device_elementwise_detail {

template<class T>
status multiply_impl(const runtime::execution_context&, const T*, const T*, T*,
                     std::uint64_t) noexcept;

template<class T>
status axpby_impl(const runtime::execution_context&, T, const T*, T, const T*, T*,
                  std::uint64_t) noexcept;

} // namespace device_elementwise_detail

// Elementwise operations use homogeneous float or double storage in legacy
// cudaMalloc device allocations or cudaMallocManaged allocations. Other CUDA
// allocation kinds and toolkits without a supported capacity query fail closed.
// Capacity lookup uses the CUDA runtime driver-entrypoint API (versioned lookup
// from CUDA 12.6, current-symbol lookup from CUDA 11.3); older toolkits fail closed.
// The implementation is linked from the prepared-relation CUDA library.
template<class T>
status multiply(const runtime::execution_context& context, const T* lhs,
                const T* rhs, T* output, std::uint64_t count) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "device relation elementwise operations support float and double");
    return device_elementwise_detail::multiply_impl(context, lhs, rhs, output, count);
}

template<class T>
status square(const runtime::execution_context& context, const T* input,
              T* output, std::uint64_t count) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "device relation elementwise operations support float and double");
    return multiply(context, input, input, output, count);
}

// Computes output[i] = alpha * x[i] + beta * y[i]. Exact output/input aliasing
// is supported; partial byte-range overlap is rejected before launch.
template<class T>
status axpby(const runtime::execution_context& context, T alpha, const T* x,
             T beta, const T* y, T* output, std::uint64_t count) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "device relation elementwise operations support float and double");
    return device_elementwise_detail::axpby_impl(context, alpha, x, beta, y, output, count);
}

} // namespace cellerator::compute::relation
