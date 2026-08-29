#pragma once

#include <cstdint>

namespace cellerator::memory {

enum class status : std::uint8_t {
    success = 0,
    invalid_argument,
    invalid_state,
    invalid_placement,
    invalid_alignment,
    capacity_exceeded,
    arithmetic_overflow,
    unsupported_domain,
    allocation_failed,
    generation_clear_required,
    cuda_failure
};

constexpr bool succeeded(status value) noexcept {
    return value == status::success;
}

} // namespace cellerator::memory
