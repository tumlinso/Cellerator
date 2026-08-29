#pragma once

#include "domain.hh"
#include "status.hh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::memory {

struct generation_marks {
    std::uint32_t *marks = nullptr;
    std::size_t count = 0;
    std::uint32_t generation = 1u;
    placement where{};
};

// Advancing is O(1). A wrap reports that the owner must explicitly clear the
// declared memory domain before calling reset_generation() with generation 1.
status advance_generation(generation_marks *table) noexcept;

status reset_generation_marks_host(generation_marks *table) noexcept;

constexpr bool contains(
    const generation_marks &table,
    std::size_t index) noexcept {
    return index < table.count && table.marks != nullptr
        && table.marks[index] == table.generation;
}

inline status insert(generation_marks *table, std::size_t index) noexcept {
    if (table == nullptr || table->marks == nullptr || index >= table->count)
        return status::invalid_argument;
    table->marks[index] = table->generation;
    return status::success;
}

static_assert(std::is_trivially_copyable<generation_marks>::value,
    "generation-mark views must remain device-copyable");

} // namespace cellerator::memory
