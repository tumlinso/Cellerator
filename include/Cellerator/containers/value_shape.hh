#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cellerator::containers {

// Checked row-major value extent shared by value owners and operations. This
// validates an extent calculation only; it does not impose a tensor/storage API.
inline std::size_t checked_value_count(std::size_t rows, std::size_t features) {
    if (features == 0) throw std::invalid_argument("feature count must be positive");
    if (rows > std::numeric_limits<std::size_t>::max() / features)
        throw std::length_error("value shape overflow");
    return rows * features;
}

} // namespace cellerator::containers
