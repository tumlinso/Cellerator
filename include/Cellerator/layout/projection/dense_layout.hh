#pragma once

#include <cstdint>

namespace cellerator::compute::math {

// Namespace and enum values preserve the existing packed-operand contract.
enum class dense_layout_kind : std::uint32_t {
    row_major = 1u,
    column_major = 2u
};

} // namespace cellerator::compute::math
