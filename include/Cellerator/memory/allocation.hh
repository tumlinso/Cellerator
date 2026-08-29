#pragma once

#include "domain.hh"
#include "status.hh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::memory {

inline constexpr std::uint32_t default_allocation_alignment = 64u;

struct allocation {
    void *base = nullptr;
    std::size_t bytes = 0;
    std::uint32_t alignment = 0;
    placement where{};
    std::uint32_t generation = 0;
};

struct allocation_request {
    std::size_t bytes = 0;
    std::uint32_t alignment = default_allocation_alignment;
    placement where{};
};

constexpr bool valid_alignment(std::size_t alignment) noexcept {
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

status validate_allocation_request(
    const allocation_request &request) noexcept;

// Records caller-owned storage without taking ownership. Allocation and
// release remain responsibilities of the caller or the execution session.
status bind_external_allocation(
    void *base,
    std::size_t bytes,
    std::uint32_t alignment,
    std::uint32_t generation,
    allocation *out) noexcept;

status reset_allocation_record(allocation *record) noexcept;

static_assert(std::is_trivially_copyable<allocation>::value,
    "allocation records must remain trivially copyable");

} // namespace cellerator::memory
