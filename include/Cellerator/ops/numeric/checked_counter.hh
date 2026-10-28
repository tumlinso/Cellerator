#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace ce_moon {
using u64 = std::uint64_t;
using u64 = std::uint64_t;
inline u64 checked_add(u64 a, u64 b) {
    if (b > std::numeric_limits<u64>::max()-a) throw std::overflow_error("counter overflow");
    return a+b;
}
} // namespace ce_moon
