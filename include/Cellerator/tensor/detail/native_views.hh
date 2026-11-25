#pragma once

#include <stddef.h>
#include <stdint.h>
#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

namespace cellerator::tensor::detail {

enum class DType : uint8_t {
    boolean, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64, invalid,
    float16
};
enum class Layout : uint8_t { dense, csr, csc, invalid };

template<class T, class U>
inline constexpr bool scalar_is = __is_same(T, U) || __is_same(T, const U);

template<class T> constexpr DType dtype() noexcept {
    if constexpr (scalar_is<T, bool>) return DType::boolean;
    if constexpr (__is_integral(T)) {
        return sizeof(T) == 1 ? (__is_signed(T) ? DType::int8 : DType::uint8)
             : sizeof(T) == 2 ? (__is_signed(T) ? DType::int16 : DType::uint16)
             : sizeof(T) == 4 ? (__is_signed(T) ? DType::int32 : DType::uint32)
             : sizeof(T) == 8 ? (__is_signed(T) ? DType::int64 : DType::uint64)
             : DType::invalid;
    }
    return scalar_is<T, float> ? DType::float32 : scalar_is<T, double> ? DType::float64 : DType::invalid;
}

template<class T> struct ValueByte { using type = unsigned char; };
template<class T> struct ValueByte<const T> { using type = const unsigned char; };

// The compiled method bodies dispatch once on the scalar types. These records
// carry only addresses and dimensions already present in the caller's views.
template<class Byte> struct DenseBytes {
    Byte* address = nullptr;
    size_t rows = 0, cols = 0, width = 0;
    ptrdiff_t row_bytes = 0, col_bytes = 0;

    Byte* data() const noexcept { return address; }
    size_t rank() const noexcept { return width ? 2 : 0; }
    size_t extent(size_t axis) const noexcept { return axis ? cols : rows; }
    ptrdiff_t stride_bytes(size_t axis) const noexcept { return axis ? col_bytes : row_bytes; }
    size_t element_size() const noexcept { return width; }
};

template<class Byte> struct SparseBytes {
    Byte* value_address = nullptr;
    const unsigned char* index_address = nullptr;
    const unsigned char* offset_address = nullptr;
    size_t row_count = 0, col_count = 0, entry_count = 0;
    size_t value_width = 0, index_width = 0, offset_width = 0;
    bool signed_indices = false, signed_offsets = false;

    Byte* values() const noexcept { return value_address; }
    const unsigned char* indices() const noexcept { return index_address; }
    const unsigned char* offsets() const noexcept { return offset_address; }
    size_t rows() const noexcept { return row_count; }
    size_t columns() const noexcept { return col_count; }
    size_t nnz() const noexcept { return entry_count; }
    size_t value_size() const noexcept { return value_width; }
    size_t index_size() const noexcept { return index_width; }
    size_t offset_size() const noexcept { return offset_width; }
};

template<class T, class S>
DenseBytes<typename ValueByte<T>::type> erase_dense(T* data, size_t rows, size_t cols,
        S row_stride, S col_stride, cellerator::memory::placement where) noexcept {
    using Byte = typename ValueByte<T>::type;
    ptrdiff_t rb, cb;
    if (where.kind == cellerator::memory::domain::device ||
        __builtin_mul_overflow(row_stride, sizeof(T), &rb) ||
        __builtin_mul_overflow(col_stride, sizeof(T), &cb)) return {};
    if (rows && cols) {
        ptrdiff_t r, c, low, high;
        if (rows-1 > size_t(PTRDIFF_MAX) || cols-1 > size_t(PTRDIFF_MAX) ||
            __builtin_mul_overflow(rows-1, rb, &r) || __builtin_mul_overflow(cols-1, cb, &c) ||
            __builtin_add_overflow(r < 0 ? r : 0, c < 0 ? c : 0, &low) ||
            __builtin_add_overflow(r > 0 ? r : 0, c > 0 ? c : 0, &high) ||
            high > PTRDIFF_MAX - ptrdiff_t(sizeof(T)-1)) return {};
    }
    return {reinterpret_cast<Byte*>(data), rows, cols, sizeof(T), rb, cb};
}

template<class T, class S>
auto erase(cellerator::memory::const_matrix_view<T, S> view) noexcept {
    return erase_dense(view.data, view.rows, view.columns, view.row_stride, view.column_stride, view.where);
}

template<class T, class S>
auto erase(cellerator::memory::matrix_view<T, S> view) noexcept {
    return erase_dense(view.data, view.rows, view.columns, view.row_stride, view.column_stride, view.where);
}

template<class I, class T, class O>
constexpr Layout layout(cellerator::tensor::base::CompressedView<I, T, O> view) noexcept {
    using Axis = cellerator::tensor::base::CompressedAxis;
    return view.structure.axis == Axis::row ? Layout::csr
         : view.structure.axis == Axis::column ? Layout::csc : Layout::invalid;
}

template<class I, class T, class O>
SparseBytes<typename ValueByte<T>::type> erase(cellerator::tensor::base::CompressedView<I, T, O> view) noexcept {
    using Byte = typename ValueByte<T>::type;
    if constexpr (!__is_integral(I) || !__is_integral(O) || scalar_is<I, bool> || scalar_is<O, bool>)
        return {};
    const auto& s = view.structure;
    uint64_t major = s.axis == cellerator::tensor::base::CompressedAxis::row ? s.rows : s.cols;
    if (s.rows > SIZE_MAX || s.cols > SIZE_MAX || major >= uint64_t(PTRDIFF_MAX)/sizeof(O) ||
        s.nnz > uint64_t(PTRDIFF_MAX)/sizeof(T) || s.nnz > uint64_t(PTRDIFF_MAX)/sizeof(I)) return {};
    return {reinterpret_cast<Byte*>(view.values), reinterpret_cast<const unsigned char*>(s.minor_idx),
        reinterpret_cast<const unsigned char*>(s.major_ptr), size_t(s.rows), size_t(s.cols), size_t(s.nnz),
        sizeof(T), sizeof(I), sizeof(O), __is_signed(I), __is_signed(O)};
}

} // namespace cellerator::tensor::detail
