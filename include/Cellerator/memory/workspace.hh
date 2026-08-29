#pragma once

#include "domain.hh"
#include "status.hh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::memory {

struct workspace_requirement {
    std::size_t bytes = 0;
    std::uint32_t alignment = 1u;
    placement where{};
};

struct workspace {
    unsigned char *base = nullptr;
    std::size_t bytes = 0;
    std::size_t cursor = 0;
    placement where{};
};

constexpr status reset(workspace *value) noexcept {
    if (value == nullptr) return status::invalid_argument;
    value->cursor = 0u;
    return status::success;
}

inline status take_bytes(
    workspace *value,
    std::size_t bytes,
    std::size_t alignment,
    void **out) noexcept {
    if (out != nullptr) *out = nullptr;
    if (value == nullptr || out == nullptr) return status::invalid_argument;
    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u)
        return status::invalid_alignment;
    if (value->cursor > value->bytes) return status::capacity_exceeded;
    if (bytes == 0u) return status::success;
    if (value->base == nullptr) return status::invalid_argument;

    const std::size_t mask = alignment - 1u;
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(value->base);
    if (value->cursor > std::numeric_limits<std::uintptr_t>::max() - base)
        return status::arithmetic_overflow;
    const std::uintptr_t current = base + value->cursor;
    if (current > std::numeric_limits<std::uintptr_t>::max() - mask)
        return status::arithmetic_overflow;
    const std::uintptr_t aligned_address = (current + mask) & ~mask;
    const std::size_t aligned = static_cast<std::size_t>(aligned_address - base);
    if (aligned > value->bytes || bytes > value->bytes - aligned)
        return status::capacity_exceeded;

    *out = value->base + aligned;
    value->cursor = aligned + bytes;
    return status::success;
}

template<class T>
inline status take(
    workspace *value,
    std::size_t count,
    std::size_t alignment,
    T **out) noexcept {
    if (out != nullptr) *out = nullptr;
    if (out == nullptr) return status::invalid_argument;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return status::arithmetic_overflow;
    void *storage = nullptr;
    const status result = take_bytes(
        value, count * sizeof(T), alignment, &storage);
    if (result == status::success) *out = static_cast<T *>(storage);
    return result;
}

template<class T>
inline status take(workspace *value, std::size_t count, T **out) noexcept {
    return take(value, count, alignof(T), out);
}

static_assert(std::is_trivially_copyable<workspace_requirement>::value,
    "workspace requirements must remain device-copyable");
static_assert(std::is_trivially_copyable<workspace>::value,
    "workspace views must remain device-copyable");

} // namespace cellerator::memory
