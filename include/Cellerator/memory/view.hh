#pragma once

#include "domain.hh"

#include <cstddef>
#include <type_traits>

namespace cellerator::memory {

template<class T>
struct array_view {
    T *data = nullptr;
    std::size_t count = 0;
    placement where{};
};

template<class T>
struct const_array_view {
    const T *data = nullptr;
    std::size_t count = 0;
    placement where{};
};

template<class T>
struct matrix_view {
    T *data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t row_stride = 0;
    placement where{};
};

template<class T>
struct const_matrix_view {
    const T *data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t row_stride = 0;
    placement where{};
};

template<class T>
constexpr const_array_view<T> as_const(array_view<T> view) noexcept {
    return {view.data, view.count, view.where};
}

template<class T>
constexpr const_matrix_view<T> as_const(matrix_view<T> view) noexcept {
    return {view.data, view.rows, view.columns, view.row_stride, view.where};
}

static_assert(std::is_trivially_copyable<array_view<unsigned char>>::value,
    "array views must remain device-copyable");
static_assert(std::is_trivially_copyable<matrix_view<unsigned char>>::value,
    "matrix views must remain device-copyable");

} // namespace cellerator::memory
