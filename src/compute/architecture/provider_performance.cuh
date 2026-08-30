#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture {

#if defined(__CUDACC__)
#define CELLERATOR_PROVIDER_HOST_DEVICE __host__ __device__
#else
#define CELLERATOR_PROVIDER_HOST_DEVICE
#endif

struct provider_launch_shape_v1 {
    std::uint32_t blocks = 0u;
    std::uint32_t threads_per_block = 0u;
    std::uint32_t dynamic_shared_memory_bytes = 0u;
};

CELLERATOR_PROVIDER_HOST_DEVICE constexpr bool valid_provider_launch_shape_v1(
    provider_launch_shape_v1 shape) noexcept {
    return shape.blocks != 0u && shape.threads_per_block != 0u
        && shape.threads_per_block <= 1024u;
}

CELLERATOR_PROVIDER_HOST_DEVICE constexpr std::uint32_t
provider_ceil_div_u32_v1(
    std::uint32_t numerator,
    std::uint32_t denominator) noexcept {
    return denominator == 0u
        ? 0u
        : numerator / denominator
            + static_cast<std::uint32_t>(numerator % denominator != 0u);
}

CELLERATOR_PROVIDER_HOST_DEVICE constexpr bool provider_checked_round_up_u64_v1(
    std::uint64_t value,
    std::uint64_t alignment,
    std::uint64_t *rounded) noexcept {
    if (rounded == nullptr || alignment == 0u
        || (alignment & (alignment - 1u)) != 0u)
        return false;
    const std::uint64_t mask = alignment - 1u;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask)
        return false;
    *rounded = (value + mask) & ~mask;
    return true;
}

CELLERATOR_PROVIDER_HOST_DEVICE constexpr bool provider_pointer_aligned_v1(
    const void *pointer,
    std::size_t alignment) noexcept {
    return pointer != nullptr && alignment != 0u
        && (alignment & (alignment - 1u)) == 0u
        && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

CELLERATOR_PROVIDER_HOST_DEVICE constexpr std::uint32_t
provider_warp_count_v1(
    std::uint32_t threads,
    std::uint32_t warp_size = 32u) noexcept {
    return provider_ceil_div_u32_v1(threads, warp_size);
}

#undef CELLERATOR_PROVIDER_HOST_DEVICE

} // namespace cellerator::compute::architecture
