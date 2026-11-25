#pragma once

#include <Cellerator/common/compiler_annotations.hh>

namespace cellerator::memory {

// Inline row-major storage. Tile placement and axis identities are supplied by
// the caller. An array of tiles has a fixed 512-byte tile stride.
template<class T>
class alignas(32) Tile {
public:
    static constexpr unsigned rows = 16, row_bytes = 32;
    static constexpr unsigned columns = row_bytes / sizeof(T);
    static constexpr unsigned bytes = rows*row_bytes;

    static_assert(row_bytes % sizeof(T) == 0 && __is_trivially_copyable(T));

    CELLERATOR_HOST_DEVICE T *data() noexcept { return values_; }
    CELLERATOR_HOST_DEVICE const T *data() const noexcept { return values_; }

private:
    T values_[rows*columns];
};

} // namespace cellerator::memory
