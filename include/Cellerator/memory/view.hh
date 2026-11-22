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

// Both strides count elements. Use ptrdiff_t for views that can run backward.
template<class T, class Stride = std::size_t>
struct matrix_view {
    T *data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
    Stride row_stride = 0;
    placement where{};
    Stride column_stride = 1;
};

template<class T, class Stride = std::size_t>
struct const_matrix_view {
    const T *data = nullptr;
    std::size_t rows = 0;
    std::size_t columns = 0;
    Stride row_stride = 0;
    placement where{};
    Stride column_stride = 1;
};

template<class T>
constexpr const_array_view<T> as_const(array_view<T> view) noexcept {
    return {view.data, view.count, view.where};
}

template<class T, class Stride>
constexpr const_matrix_view<T, Stride> as_const(matrix_view<T, Stride> view) noexcept {
    return {view.data, view.rows, view.columns, view.row_stride, view.where, view.column_stride};
}

static_assert(std::is_trivially_copyable<array_view<unsigned char>>::value,
    "array views must remain device-copyable");
static_assert(std::is_trivially_copyable<matrix_view<unsigned char>>::value,
    "matrix views must remain device-copyable");

} // namespace cellerator::memory
