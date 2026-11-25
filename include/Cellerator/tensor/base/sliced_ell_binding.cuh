#pragma once

#include <Cellerator/memory/status.hh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <stddef.h>
#include <stdint.h>

namespace cellerator::tensor::base::device {

struct SlicedEllShape {
    unsigned rows = 0, cols = 0, nnz = 0, slice_count = 0, slice_rows = 0;
    size_t slots = 0; // Includes padding.
};

// One byte span for the existing row-packed SlicedEllView. The row offsets
// include the terminal entry. The other two metadata arrays have slice_count
// entries. The caller supplies the packed contents and retains the span.
class SlicedEllLayout {
public:
    static constexpr size_t alignment = 32;

    static memory::status prepare(SlicedEllShape shape, SlicedEllLayout *out) noexcept {
        using memory::status;
        if (!out || shape.nnz > shape.slots || (!shape.slice_count && shape.slots) ||
            (shape.nnz && (!shape.rows || !shape.cols))) return status::invalid_argument;

        SlicedEllLayout layout;
        layout.shape_ = shape;
        size_t row_offset_count, cursor;
        if (__builtin_add_overflow(size_t(shape.slice_count), size_t(1), &row_offset_count) ||
            __builtin_mul_overflow(row_offset_count, sizeof(unsigned), &cursor) ||
            !append<unsigned>(shape.slice_count, alignof(unsigned), &cursor, &layout.widths_) ||
            !append<unsigned>(shape.slice_count, alignof(unsigned), &cursor, &layout.slot_offsets_) ||
            !append<unsigned>(shape.slots, alignment, &cursor, &layout.columns_) ||
            !append<real::f16_t>(shape.slots, alignment, &cursor, &layout.values_) ||
            !align_up(cursor, alignment, &layout.bytes_)) return status::arithmetic_overflow;
        *out = layout;
        return status::success;
    }

    constexpr size_t bytes() const noexcept { return bytes_; }

    // Binding checks the span without reading or initializing device data.
    memory::status bind(void *base, size_t capacity, SlicedEllView *out) const noexcept {
        using memory::status;
        if (!out) return status::invalid_argument;
        if (!bytes_) return status::invalid_state;
        if (bytes_ > capacity) return status::capacity_exceeded;
        if (!base) return status::invalid_argument;
        if (reinterpret_cast<uintptr_t>(base) % alignment) return status::invalid_alignment;
        auto *p = static_cast<unsigned char *>(base);
        *out = {shape_.rows, shape_.cols, shape_.nnz, shape_.slice_count, shape_.slice_rows,
                reinterpret_cast<unsigned *>(p),
                shape_.slice_count ? reinterpret_cast<unsigned *>(p + widths_) : nullptr,
                shape_.slice_count ? reinterpret_cast<unsigned *>(p + slot_offsets_) : nullptr,
                shape_.slots ? reinterpret_cast<unsigned *>(p + columns_) : nullptr,
                shape_.slots ? reinterpret_cast<real::f16_t *>(p + values_) : nullptr};
        return status::success;
    }

private:
    static bool align_up(size_t value, size_t alignment, size_t *out) noexcept {
        size_t padded;
        if (__builtin_add_overflow(value, alignment - 1u, &padded)) return false;
        *out = padded & ~(alignment - 1u);
        return true;
    }

    template<class T>
    static bool append(size_t count, size_t alignment, size_t *cursor, size_t *offset) noexcept {
        size_t bytes;
        return align_up(*cursor, alignment, offset) &&
            !__builtin_mul_overflow(count, sizeof(T), &bytes) &&
            !__builtin_add_overflow(*offset, bytes, cursor);
    }

    SlicedEllShape shape_{};
    size_t widths_ = 0, slot_offsets_ = 0, columns_ = 0, values_ = 0, bytes_ = 0;
};

} // namespace cellerator::tensor::base::device
