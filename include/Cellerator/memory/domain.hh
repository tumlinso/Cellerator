#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::memory {

enum class domain : std::uint8_t {
    host = 0,
    host_numa,
    host_pinned,
    host_pinned_write_combined,
    device,
    managed,
    external
};

struct placement {
    domain kind = domain::external;
    std::int16_t device_ordinal = -1;
    std::int16_t numa_node = -1;
    std::uint32_t flags = 0;
};

constexpr bool operator==(placement left, placement right) noexcept {
    return left.kind == right.kind
        && left.device_ordinal == right.device_ordinal
        && left.numa_node == right.numa_node
        && left.flags == right.flags;
}

constexpr bool operator!=(placement left, placement right) noexcept {
    return !(left == right);
}

static_assert(std::is_trivially_copyable<placement>::value,
    "memory placement must remain device-copyable");

} // namespace cellerator::memory
